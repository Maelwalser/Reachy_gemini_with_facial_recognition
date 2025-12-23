"""Movement controller for Reachy Mini expressions and head movements.

This module provides high-level movement commands that the Gemini model
can use as tools during conversation.
"""

import asyncio
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)


class MovementController:
    """Controls Reachy Mini head movements and expressions."""

    def __init__(self, robot: ReachyMini):
        """Initialize the movement controller.

        Args:
            robot: ReachyMini instance to control
        """
        self.robot = robot

    async def move_head(self, direction: str, duration: float = 0.5) -> None:
        """Move the head in a direction.

        Args:
            direction: One of 'left', 'right', 'up', 'down', 'center'
            duration: Movement duration in seconds
        """
        # Define head poses for each direction (roll, pitch, yaw in degrees)
        poses = {
            "left": (0, 0, 25),      # yaw left
            "right": (0, 0, -25),    # yaw right
            "up": (0, -20, 0),       # pitch up
            "down": (0, 20, 0),      # pitch down
            "center": (0, 0, 0),     # neutral
        }

        if direction not in poses:
            logger.warning(f"Unknown direction: {direction}, using center")
            direction = "center"

        roll, pitch, yaw = poses[direction]

        # Create head pose matrix
        pose = create_head_pose(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            degrees=True,
        )

        logger.info(f"Moving head {direction}")
        self.robot.goto_target(head=pose, duration=duration)

    async def express_emotion(self, emotion: str) -> None:
        """Express an emotion through movement.

        Args:
            emotion: One of 'happy', 'sad', 'surprised', 'curious', 'excited', 'sleepy'
        """
        logger.info(f"Expressing emotion: {emotion}")

        if emotion == "happy":
            await self._happy_expression()
        elif emotion == "sad":
            await self._sad_expression()
        elif emotion == "surprised":
            await self._surprised_expression()
        elif emotion == "curious":
            await self._curious_expression()
        elif emotion == "excited":
            await self._excited_expression()
        elif emotion == "sleepy":
            await self._sleepy_expression()
        else:
            logger.warning(f"Unknown emotion: {emotion}")

    async def _happy_expression(self) -> None:
        """Express happiness with a head wiggle and antenna bounce."""
        # Quick head tilt right
        pose = create_head_pose(roll=15, degrees=True)
        self.robot.goto_target(head=pose, antennas=[0.3, -0.3], duration=0.2)
        await asyncio.sleep(0.2)

        # Quick head tilt left
        pose = create_head_pose(roll=-15, degrees=True)
        self.robot.goto_target(head=pose, antennas=[-0.3, 0.3], duration=0.2)
        await asyncio.sleep(0.2)

        # Back to center
        pose = create_head_pose(degrees=True)
        self.robot.goto_target(head=pose, antennas=[0, 0], duration=0.2)

    async def _sad_expression(self) -> None:
        """Express sadness with droopy head and antennas."""
        # Look down with droopy antennas
        pose = create_head_pose(pitch=25, degrees=True)
        self.robot.goto_target(head=pose, antennas=[-1.5, 1.5], duration=0.8)
        await asyncio.sleep(1.0)

        # Slowly return to neutral
        pose = create_head_pose(degrees=True)
        self.robot.goto_target(head=pose, antennas=[0, 0], duration=0.5)

    async def _surprised_expression(self) -> None:
        """Express surprise with quick look up and antenna pop."""
        # Quick look up with antennas up
        pose = create_head_pose(pitch=-20, degrees=True)
        self.robot.goto_target(head=pose, antennas=[1.0, -1.0], duration=0.15)
        await asyncio.sleep(0.3)

        # Hold briefly
        await asyncio.sleep(0.3)

        # Return to neutral
        pose = create_head_pose(degrees=True)
        self.robot.goto_target(head=pose, antennas=[0, 0], duration=0.3)

    async def _curious_expression(self) -> None:
        """Express curiosity with head tilt."""
        # Tilt head to side
        pose = create_head_pose(roll=20, pitch=-10, degrees=True)
        self.robot.goto_target(head=pose, antennas=[0.5, -0.2], duration=0.4)
        await asyncio.sleep(0.6)

        # Return to neutral
        pose = create_head_pose(degrees=True)
        self.robot.goto_target(head=pose, antennas=[0, 0], duration=0.3)

    async def _excited_expression(self) -> None:
        """Express excitement with bouncy movements."""
        for _ in range(3):
            # Quick up
            pose = create_head_pose(pitch=-10, degrees=True)
            self.robot.goto_target(head=pose, antennas=[0.8, -0.8], duration=0.1)
            await asyncio.sleep(0.1)

            # Quick down
            pose = create_head_pose(pitch=5, degrees=True)
            self.robot.goto_target(head=pose, antennas=[-0.3, 0.3], duration=0.1)
            await asyncio.sleep(0.1)

        # Return to neutral
        pose = create_head_pose(degrees=True)
        self.robot.goto_target(head=pose, antennas=[0, 0], duration=0.2)

    async def _sleepy_expression(self) -> None:
        """Express sleepiness with slow droopy movement."""
        # Slowly droop head and antennas
        pose = create_head_pose(pitch=30, roll=10, degrees=True)
        self.robot.goto_target(head=pose, antennas=[-2.0, 2.0], duration=1.5)
        await asyncio.sleep(1.5)

        # Small "nod off" movement
        pose = create_head_pose(pitch=35, roll=10, degrees=True)
        self.robot.goto_target(head=pose, duration=0.3)
        await asyncio.sleep(0.4)

        # Wake up slightly
        pose = create_head_pose(pitch=20, degrees=True)
        self.robot.goto_target(head=pose, antennas=[-1.0, 1.0], duration=0.4)
