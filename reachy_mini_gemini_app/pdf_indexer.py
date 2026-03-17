"""PDF Indexer for efficient large-document retrieval.

Builds a per-page content index using Gemini, then retrieves only
the relevant pages when answering queries. Designed for manuals
and technical documents of 400+ pages.
"""

import asyncio
import io
import json
import logging
import os
import hashlib
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Directory to cache indices so we don't re-index on every restart
INDEX_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "reachy-mini-gemini" / "pdf-indices"

# How many pages to summarize in a single Gemini call
INDEX_BATCH_SIZE = 20

# Max pages to retrieve per query
MAX_RETRIEVAL_PAGES = 12


class PDFIndexer:
    """Indexes PDF pages and retrieves relevant subsets for queries."""

    def __init__(self, client: genai.Client, model: str = "gemini-2.5-flash"):
        self.client = client
        self.model = model
        # {file_path: {"hash": str, "pages": {page_num: summary}, "total_pages": int}}
        self.indices: dict[str, dict[str, Any]] = {}
        # Raw page texts keyed by file path
        self._page_texts: dict[str, dict[int, str]] = {}
        # PDF bytes cache for page extraction
        self._pdf_bytes: dict[str, bytes] = {}

    def _file_hash(self, file_path: str) -> str:
        """Compute a hash to detect if the PDF changed since last index."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def _load_cached_index(self, file_path: str, file_hash: str) -> dict[str, Any] | None:
        """Try to load a previously computed index from disk."""
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = INDEX_CACHE_DIR / f"{file_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                logger.info(f"Loaded cached index for {file_path} ({data.get('total_pages', '?')} pages)")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}")
        return None

    def _save_cached_index(self, file_hash: str, index_data: dict[str, Any]) -> None:
        """Persist the index to disk."""
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = INDEX_CACHE_DIR / f"{file_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(index_data, f, ensure_ascii=False)
            logger.info(f"Saved index cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save index cache: {e}")

    def _extract_page_texts(self, file_path: str) -> dict[int, str]:
        """Extract text content from every page of a PDF using PyMuPDF."""
        import fitz  # PyMuPDF

        page_texts: dict[int, str] = {}
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            # Store 1-indexed page numbers for human readability
            if text:
                page_texts[page_num + 1] = text
            else:
                # Page might be image-only; note it but don't index empty content
                page_texts[page_num + 1] = "[Image-only page or no extractable text]"

        doc.close()
        logger.info(f"Extracted text from {len(page_texts)} pages in {file_path}")
        return page_texts

    async def _summarize_batch(self, pages: dict[int, str]) -> dict[int, str]:
        """Send a batch of pages to Gemini to produce per-page content summaries."""

        # Build the prompt with page contents
        page_entries = []
        for page_num, text in sorted(pages.items()):
            # Truncate very long pages to avoid token limits
            truncated = text[:3000] if len(text) > 3000 else text
            page_entries.append(f"--- PAGE {page_num} ---\n{truncated}")

        combined = "\n\n".join(page_entries)

        prompt = (
            "You are a document indexer. For each page below, produce a concise summary (1-2 sentences) "
            "describing the key topics, sections, procedures, part numbers, or concepts on that page. "
            "Focus on content that someone might search for later.\n\n"
            "Respond ONLY with a valid JSON object mapping page numbers (as strings) to their summaries. "
            "Example: {\"1\": \"Table of contents listing chapters 1-12.\", \"2\": \"Safety warnings and PPE requirements.\"}\n\n"
            "Do not include any text outside the JSON object. No markdown fences.\n\n"
            f"{combined}"
        )

        def _call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low creativity for factual indexing
                ),
            )
            return response.text

        try:
            result = await asyncio.to_thread(_call)
            if not result:
                return {}

            # Clean potential markdown fences
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            # Convert string keys back to int
            return {int(k): v for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse index batch response: {e}")
            logger.debug(f"Raw response: {result[:500] if result else 'None'}")
            # Fallback: use first 100 chars of each page as summary
            return {pn: text[:100] for pn, text in pages.items()}

    async def index_pdf(self, file_path: str) -> dict[str, Any]:
        """Build a complete page-level index for a PDF document.

        Returns the index dict: {"hash": ..., "pages": {page_num: summary}, "total_pages": int}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        file_hash = self._file_hash(file_path)

        # Check disk cache first
        cached = self._load_cached_index(file_path, file_hash)
        if cached:
            self.indices[file_path] = cached
            # Still need page texts for retrieval
            self._page_texts[file_path] = self._extract_page_texts(file_path)
            with open(file_path, "rb") as f:
                self._pdf_bytes[file_path] = f.read()
            return cached

        logger.info(f"Building index for {file_path}...")

        # Extract all page texts
        page_texts = self._extract_page_texts(file_path)
        self._page_texts[file_path] = page_texts

        # Cache raw PDF bytes for page extraction later
        with open(file_path, "rb") as f:
            self._pdf_bytes[file_path] = f.read()

        # Batch pages for summarization
        page_nums = sorted(page_texts.keys())
        all_summaries: dict[int, str] = {}

        for i in range(0, len(page_nums), INDEX_BATCH_SIZE):
            batch_nums = page_nums[i : i + INDEX_BATCH_SIZE]
            batch = {pn: page_texts[pn] for pn in batch_nums}

            batch_start = batch_nums[0]
            batch_end = batch_nums[-1]
            logger.info(f"Indexing pages {batch_start}-{batch_end} of {len(page_nums)}...")

            summaries = await self._summarize_batch(batch)
            all_summaries.update(summaries)

            # Small delay to avoid rate limits
            if i + INDEX_BATCH_SIZE < len(page_nums):
                await asyncio.sleep(0.5)

        index_data = {
            "hash": file_hash,
            "total_pages": len(page_texts),
            "pages": {str(k): v for k, v in sorted(all_summaries.items())},
        }

        self.indices[file_path] = index_data
        self._save_cached_index(file_hash, index_data)

        logger.info(f"Index complete: {len(all_summaries)} pages indexed for {file_path}")
        return index_data

    async def _identify_relevant_pages(self, query: str, file_path: str) -> list[int]:
        """Use Gemini to determine which pages are relevant for a query."""
        index = self.indices.get(file_path)
        if not index:
            return []

        # Build a compact index representation
        index_lines = []
        for page_num, summary in sorted(index["pages"].items(), key=lambda x: int(x[0])):
            index_lines.append(f"p{page_num}: {summary}")

        index_text = "\n".join(index_lines)

        prompt = (
            f"You are a document retrieval system. Given the query and a page index, "
            f"identify which pages contain information relevant to answering the query.\n\n"
            f"Query: {query}\n\n"
            f"Document index ({index['total_pages']} pages total):\n{index_text}\n\n"
            f"Return ONLY a JSON array of page numbers (integers) that are most relevant, "
            f"ordered by relevance. Return at most {MAX_RETRIEVAL_PAGES} pages. "
            f"If no pages seem relevant, return an empty array [].\n"
            f"No markdown fences, no explanation — just the JSON array."
        )

        def _call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            return response.text

        try:
            result = await asyncio.to_thread(_call)
            if not result:
                return []

            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            pages = json.loads(cleaned)
            if isinstance(pages, list):
                return [int(p) for p in pages[:MAX_RETRIEVAL_PAGES]]
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse page identification response: {e}")
            return []

    def _extract_pdf_pages(self, file_path: str, page_nums: list[int]) -> bytes:
        """Extract specific pages from a PDF and return as a new PDF bytes."""
        import fitz

        pdf_bytes = self._pdf_bytes.get(file_path)
        if not pdf_bytes:
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            self._pdf_bytes[file_path] = pdf_bytes

        src_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out_doc = fitz.open()  # Empty new document

        for page_num in sorted(page_nums):
            # Convert 1-indexed to 0-indexed
            idx = page_num - 1
            if 0 <= idx < len(src_doc):
                out_doc.insert_pdf(src_doc, from_page=idx, to_page=idx)

        result = out_doc.tobytes()
        out_doc.close()
        src_doc.close()
        return result

    def _get_page_texts_for_query(self, file_path: str, page_nums: list[int]) -> str:
        """Get the raw text of specific pages, formatted for the LLM."""
        page_texts = self._page_texts.get(file_path, {})
        sections = []
        for pn in sorted(page_nums):
            text = page_texts.get(pn, "")
            if text:
                sections.append(f"=== PAGE {pn} ===\n{text}")
        return "\n\n".join(sections)

    async def query(self, query: str, file_path: str | None = None) -> str:
        """Run the full retrieval pipeline: identify pages → extract → answer.

        If file_path is None, searches across all indexed files.
        Returns the detailed answer text.
        """
        # Determine which files to search
        target_files = [file_path] if file_path else list(self.indices.keys())

        if not target_files:
            return "Error: No documents have been indexed."

        all_relevant: list[tuple[str, list[int]]] = []

        for fp in target_files:
            if fp not in self.indices:
                continue
            pages = await self._identify_relevant_pages(query, fp)
            if pages:
                all_relevant.append((fp, pages))
                logger.info(f"Identified {len(pages)} relevant pages in {fp}: {pages}")

        if not all_relevant:
            return "No relevant pages found in the indexed documents for this query."

        # Build context from all relevant pages across files
        context_parts = []
        for fp, pages in all_relevant:
            fname = os.path.basename(fp)
            page_text = self._get_page_texts_for_query(fp, pages)
            context_parts.append(
                f"[Source: {fname}, Pages: {', '.join(str(p) for p in pages)}]\n{page_text}"
            )

        combined_context = "\n\n---\n\n".join(context_parts)

        # Final answer generation with focused context
        prompt = (
            "You are a rigorous technical data extractor. Based ONLY on the document pages below, "
            "answer the following query with full technical detail. Include all specific measurements, "
            "part numbers, screw types, torque specifications, and step-by-step instructions exactly "
            "as they appear in the source text. If the information spans multiple pages, synthesize "
            "it coherently. Always cite the page number(s) you are drawing from.\n\n"
            f"Query: {query}\n\n"
            f"Document Context:\n{combined_context}"
        )

        def _call():
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            return response.text

        try:
            result = await asyncio.to_thread(_call)
            return result if result else "No answer could be generated from the retrieved pages."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"

    async def index_all(self, file_paths: list[str]) -> dict[str, int]:
        """Index multiple PDF files. Returns {path: page_count}."""
        results = {}
        for fp in file_paths:
            if not fp.lower().endswith(".pdf"):
                logger.warning(f"Skipping non-PDF file: {fp}")
                continue
            try:
                index = await self.index_pdf(fp)
                results[fp] = index["total_pages"]
            except Exception as e:
                logger.error(f"Failed to index {fp}: {e}")
                results[fp] = -1
        return results
