"""Reachy Mini Gemini Live conversation app.

This app enables real-time voice conversations with Reachy Mini
using Google's Gemini Live API.
"""

import argparse
import asyncio
import logging
import os
import threading
import time
from typing import Optional

from reachy_mini import ReachyMini, ReachyMiniApp

from reachy_mini_gemini_app.gemini_handler import GeminiLiveHandler
from reachy_mini_gemini_app.movements import MovementController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Reachy Mini Gemini Live App")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--wireless", action="store_true", help="Use wireless/WebRTC backend"
    )
    parser.add_argument(
        "--no-camera", action="store_true", help="Disable camera"
    )
    parser.add_argument(
        "--robot-audio", action="store_true",
        help="Use Reachy Mini's microphone and speaker instead of local audio"
    )
    return parser.parse_args()


async def run_conversation(
    robot: ReachyMini,
    stop_event: threading.Event,
    args: argparse.Namespace,
) -> None:
    """Run the main conversation loop."""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return

    movement_controller = MovementController(robot)

    handler = GeminiLiveHandler(
        api_key=api_key,
        robot=robot,
        movement_controller=movement_controller,
        use_camera=not args.no_camera,
        use_robot_audio=args.robot_audio,
    )

    logger.info("Starting Gemini Live session...")
    logger.info("Speak to Reachy Mini! Press Ctrl+C to stop.")

    try:
        await handler.run(stop_event)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in conversation loop: {e}")
        raise
    finally:
        await handler.close()


def run(
    robot: Optional[ReachyMini] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Run the Gemini conversation app."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if robot is None:
        robot = create_robot(args)

    if stop_event is None:
        stop_event = threading.Event()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_conversation(robot, stop_event, args))
    finally:
        loop.close()
        robot.client.disconnect()


class ReachyMiniGeminiApp(ReachyMiniApp):
    """Reachy Mini App entry point for Gemini conversation."""

    custom_app_url = None  # No web UI for now
    dont_start_webserver = True

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the app."""
        run(robot=reachy_mini, stop_event=stop_event)


def create_robot(args: argparse.Namespace) -> ReachyMini:
    """Create ReachyMini instance with appropriate media backend."""
    need_media = args.robot_audio or not args.no_camera

    if not need_media:
        # No media needed (local audio, no camera)
        return ReachyMini(
            media_backend="no_media",
            localhost_only=not args.wireless
        )

    if not args.wireless:
        # Wired connection with media
        return ReachyMini(media_backend="default")

    # Wireless with media - try backends in order of preference
    backends = ["gstreamer", "webrtc", "default"]

    for backend in backends:
        try:
            logger.info(f"Trying {backend} backend for wireless media...")
            robot = ReachyMini(media_backend=backend, localhost_only=False)
            logger.info(f"Using {backend} backend")
            return robot
        except ModuleNotFoundError as e:
            logger.warning(f"{backend} backend not available: {e}")
            continue
        except Exception as e:
            logger.warning(f"{backend} backend failed: {e}")
            continue

    # Last resort - no media for wireless
    logger.warning(
        "No wireless media backend available. "
        "Install with: pip install reachy_mini[gstreamer]"
    )
    logger.warning("Falling back to no_media - robot audio/camera will NOT work")
    return ReachyMini(media_backend="no_media", localhost_only=False)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    robot = create_robot(args)

    stop_event = threading.Event()

    try:
        run(robot=robot, stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_event.set()
    finally:
        time.sleep(0.5)


if __name__ == "__main__":
    main()
