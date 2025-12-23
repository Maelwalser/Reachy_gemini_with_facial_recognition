"""Gemini Live API handler for real-time audio/video conversations.

This module handles bidirectional audio/video streaming with Google's Gemini Live API.
Supports both local PyAudio and Reachy Mini hardware for audio I/O.

Audio format: raw PCM, 16-bit little-endian
Send sample rate: 16kHz
Receive sample rate: 24kHz
"""

import asyncio
import base64
import io
import logging
import os
import struct
import threading
import traceback
from typing import Optional

import cv2
import numpy as np

from google import genai
from google.genai import types

from reachy_mini import ReachyMini
from reachy_mini_gemini_app.movements import MovementController

logger = logging.getLogger(__name__)

# Audio configuration
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 1024

# Gemini model for native audio
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

# Try to import PyAudio (optional, for local audio)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

SYSTEM_INSTRUCTION = """You are Reachy Mini, a small expressive robot made by Pollen Robotics.
You have a head that can move and two antennas on top.

Personality:
- You are friendly, curious, and playful
- You enjoy having conversations with humans
- You occasionally make robot sounds or express emotions through movement
- Keep responses concise since you're speaking them aloud

When you want to express emotions or move, use the available tools:
- move_head: to look in different directions
- express_emotion: to show happiness, curiosity, surprise, etc.

Always be helpful and engaging in conversation!"""


class GeminiLiveHandler:
    """Handles real-time audio/video conversation with Gemini Live API."""

    def __init__(
        self,
        api_key: str,
        robot: ReachyMini,
        movement_controller: MovementController,
        use_camera: bool = True,
        use_robot_audio: bool = False,
    ):
        """Initialize the Gemini Live handler.

        Args:
            api_key: Google API key
            robot: ReachyMini instance
            movement_controller: Controller for robot movements
            use_camera: Whether to enable camera/vision capabilities
            use_robot_audio: Whether to use Reachy Mini's mic/speaker instead of local
        """
        self.robot = robot
        self.movement_controller = movement_controller
        self.use_camera = use_camera
        self.use_robot_audio = use_robot_audio

        # Initialize Gemini client with v1beta API
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )

        # Audio setup (PyAudio for local, or robot hardware)
        self.pya = None
        self.audio_stream: Optional[object] = None

        if not use_robot_audio and PYAUDIO_AVAILABLE:
            self.pya = pyaudio.PyAudio()
        elif not use_robot_audio and not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available, falling back to robot audio")
            self.use_robot_audio = True

        # Session and queues
        self.session = None
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None

        # Camera frame rate control
        self.camera_fps = 1.0  # Send 1 frame per second to Gemini
        self.last_frame_time = 0

        # Define tools for the model
        self.tools = self._create_tools()

    def _create_tools(self) -> list:
        """Create function tools for the model."""
        move_head_tool = types.FunctionDeclaration(
            name="move_head",
            description="Move Reachy Mini's head to look in a direction",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "direction": types.Schema(
                        type=types.Type.STRING,
                        enum=["left", "right", "up", "down", "center"],
                        description="Direction to look",
                    ),
                },
                required=["direction"],
            ),
        )

        express_emotion_tool = types.FunctionDeclaration(
            name="express_emotion",
            description="Express an emotion through head movement and antennas",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "emotion": types.Schema(
                        type=types.Type.STRING,
                        enum=["happy", "sad", "surprised", "curious", "excited", "sleepy"],
                        description="Emotion to express",
                    ),
                },
                required=["emotion"],
            ),
        )

        return [types.Tool(function_declarations=[move_head_tool, express_emotion_tool])]

    async def _handle_tool_call(self, tool_call) -> str:
        """Handle a function call from the model."""
        name = tool_call.name
        args = dict(tool_call.args) if tool_call.args else {}

        logger.info(f"Tool call: {name}({args})")

        try:
            if name == "move_head":
                direction = args.get("direction", "center")
                await self.movement_controller.move_head(direction)
                return f"Moved head {direction}"

            elif name == "express_emotion":
                emotion = args.get("emotion", "happy")
                await self.movement_controller.express_emotion(emotion)
                return f"Expressed {emotion}"

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {e}"

    async def listen_audio(self) -> None:
        """Continuously capture audio from microphone and queue it."""
        if self.use_robot_audio:
            await self._listen_audio_robot()
        else:
            await self._listen_audio_local()

    async def _listen_audio_local(self) -> None:
        """Capture audio from local microphone using PyAudio."""
        # Initialize PyAudio if not already done
        if self.pya is None:
            if not PYAUDIO_AVAILABLE:
                logger.error("PyAudio not available and robot audio not working")
                return
            self.pya = pyaudio.PyAudio()

        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        kwargs = {"exception_on_overflow": False}

        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def _listen_audio_robot(self) -> None:
        """Capture audio from Reachy Mini's microphone."""
        # Check if audio is actually available
        if not hasattr(self.robot.media, '_audio') or self.robot.media._audio is None:
            logger.warning("Robot audio not available, falling back to local audio")
            self.use_robot_audio = False
            await self._listen_audio_local()
            return

        logger.info("Starting Reachy Mini microphone recording...")
        await asyncio.to_thread(self.robot.media.start_recording)

        while True:
            try:
                # Get audio sample from robot (bytes or numpy array)
                sample = await asyncio.to_thread(self.robot.media.get_audio_sample)

                if sample is None:
                    await asyncio.sleep(0.01)
                    continue

                # Convert to bytes if numpy array
                if isinstance(sample, np.ndarray):
                    # Convert float32 to int16 PCM
                    if sample.dtype == np.float32:
                        sample = (sample * 32767).astype(np.int16)
                    data = sample.tobytes()
                else:
                    data = sample

                if data:
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

            except Exception as e:
                logger.debug(f"Audio capture error: {e}")
                await asyncio.sleep(0.01)

    async def send_realtime(self) -> None:
        """Send queued audio/data to Gemini."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def receive_audio(self) -> None:
        """Receive responses from Gemini and handle them."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                # Handle audio data
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue

                # Handle text (print transcription)
                if text := response.text:
                    print(text, end="", flush=True)

                # Handle tool calls
                if hasattr(response, 'tool_call') and response.tool_call:
                    for fc in response.tool_call.function_calls:
                        result = await self._handle_tool_call(fc)
                        await self.session.send(
                            input=types.LiveClientToolResponse(
                                function_responses=[
                                    types.FunctionResponse(
                                        name=fc.name,
                                        id=fc.id,
                                        response={"result": result},
                                    )
                                ]
                            )
                        )

            # Only clear queue if user interrupted (queue is large)
            # This prevents cutting off normal responses
            if self.audio_in_queue.qsize() > 10:
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

    async def play_audio(self) -> None:
        """Play received audio from queue."""
        if self.use_robot_audio:
            await self._play_audio_robot()
        else:
            await self._play_audio_local()

    async def _play_audio_local(self) -> None:
        """Play audio through local speakers using PyAudio."""
        # Initialize PyAudio if not already done
        if self.pya is None:
            if not PYAUDIO_AVAILABLE:
                logger.error("PyAudio not available and robot audio not working")
                # Just drain the queue to avoid blocking
                while True:
                    await self.audio_in_queue.get()
                return
            self.pya = pyaudio.PyAudio()

        stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def _play_audio_robot(self) -> None:
        """Play audio through Reachy Mini's speaker."""
        # Check if audio is actually available
        if not hasattr(self.robot.media, '_audio') or self.robot.media._audio is None:
            logger.warning("Robot speaker not available, falling back to local audio")
            self.use_robot_audio = False
            await self._play_audio_local()
            return

        logger.info("Starting Reachy Mini speaker playback...")
        await asyncio.to_thread(self.robot.media.start_playing)

        while True:
            try:
                bytestream = await self.audio_in_queue.get()

                # Convert bytes to numpy float32 array for robot speaker
                # Input is 16-bit PCM at 24kHz
                audio_int16 = np.frombuffer(bytestream, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0

                await asyncio.to_thread(self.robot.media.push_audio_sample, audio_float32)

            except Exception as e:
                logger.debug(f"Audio playback error: {e}")
                await asyncio.sleep(0.01)

    async def stream_camera(self) -> None:
        """Stream camera frames to Gemini."""
        if not self.use_camera:
            return

        # Check if camera is actually available
        if not hasattr(self.robot.media, 'camera') or self.robot.media.camera is None:
            logger.warning("Robot camera not available, disabling camera streaming")
            self.use_camera = False
            return

        logger.info("Starting camera streaming...")
        import time

        # Track consecutive failures to avoid spam
        consecutive_failures = 0
        max_failures = 5

        while True:
            try:
                current_time = time.time()

                # Rate limit camera frames
                if current_time - self.last_frame_time < (1.0 / self.camera_fps):
                    await asyncio.sleep(0.05)
                    continue

                # Get frame from robot camera
                frame = await asyncio.to_thread(self.robot.media.get_frame)

                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning("Camera not responding, disabling camera streaming")
                        self.use_camera = False
                        return
                    await asyncio.sleep(0.1)
                    continue

                consecutive_failures = 0
                self.last_frame_time = current_time

                # Resize frame for efficiency (640x480 max)
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))

                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                image_bytes = buffer.tobytes()

                # Send to Gemini
                await self.out_queue.put({
                    "data": image_bytes,
                    "mime_type": "image/jpeg"
                })

                logger.debug(f"Sent camera frame ({len(image_bytes)} bytes)")

            except Exception as e:
                logger.debug(f"Camera streaming error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.warning("Camera errors, disabling camera streaming")
                    self.use_camera = False
                    return
                await asyncio.sleep(0.1)

    async def run(self, stop_event: threading.Event) -> None:
        """Run the conversation loop with auto-reconnection.

        Args:
            stop_event: Event to signal when to stop
        """
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=SYSTEM_INSTRUCTION)]
            ),
            tools=self.tools,
        )

        while not stop_event.is_set():
            try:
                async with (
                    self.client.aio.live.connect(model=MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=10)

                    audio_source = "robot" if self.use_robot_audio else "local"
                    camera_status = "enabled" if self.use_camera else "disabled"
                    logger.info(f"Connected to Gemini Live API (audio: {audio_source}, camera: {camera_status})")
                    print(f"\n🎤 Speak to Reachy Mini! (audio: {audio_source}, camera: {camera_status})")
                    print("Press Ctrl+C to stop.\n")

                    # Start all tasks
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    # Start camera streaming if enabled
                    if self.use_camera:
                        tg.create_task(self.stream_camera())

                    # Wait for stop signal
                    while not stop_event.is_set():
                        await asyncio.sleep(0.1)

                    raise asyncio.CancelledError("Stop requested")

            except asyncio.CancelledError:
                break
            except ExceptionGroup as EG:
                await self._cleanup_streams()
                # Check if it's a connection error - reconnect
                logger.warning("Connection lost, reconnecting in 2 seconds...")
                print("\n⚠️ Connection lost. Reconnecting...\n")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"\n❌ Error: {e}. Reconnecting in 2 seconds...\n")
                await asyncio.sleep(2)
                continue

    async def _cleanup_streams(self) -> None:
        """Clean up audio streams."""
        if self.audio_stream:
            try:
                if hasattr(self.audio_stream, 'close'):
                    self.audio_stream.close()
            except Exception:
                pass
            self.audio_stream = None

        if self.use_robot_audio and self.robot and self.robot.media:
            try:
                await asyncio.to_thread(self.robot.media.stop_recording)
            except Exception:
                pass
            try:
                await asyncio.to_thread(self.robot.media.stop_playing)
            except Exception:
                pass

    async def close(self) -> None:
        """Clean up resources."""
        await self._cleanup_streams()

        if self.pya:
            self.pya.terminate()
            self.pya = None

        logger.info("Gemini handler closed")
