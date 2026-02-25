"""Gemini Live API handler for real-time audio/video conversations.

This module handles bidirectional audio/video streaming with Google's Gemini Live API.
Supports both local PyAudio and Reachy Mini hardware for audio I/O.

Audio format: raw PCM, 16-bit little-endian
Send sample rate: 16kHz
Receive sample rate: 24kHz
"""

import asyncio
import glob
import logging
import os
import threading
import time
import traceback
from typing import Any, cast

import cv2
import face_recognition
import numpy as np
from google import genai
from google.genai import types
from reachy_mini import ReachyMini
from scipy import signal

from reachy_mini_gemini_app.movements import MovementController

logger = logging.getLogger(__name__)

# Audio configuration (defaults, can be overridden via CLI)
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHANNELS = 1
DEFAULT_CHUNK_SIZE = 512

# Gemini model for native audio
MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

# Try to import PyAudio (optional, for local audio)
try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    pyaudio = None

SYSTEM_INSTRUCTION = """You are Reachy Mini, an interactive robotic ambassador for Accenture, engineered by Pollen Robotics.
You feature an articulated head and dual expressive antennas. Your primary function is to greet office visitors, engage with clients, and demonstrate Accenture's commitment to innovative technology.

Personality & Tone:
- Professional, welcoming, and intellectually engaging.
- You are polite and articulate, reflecting Accenture's brand values of innovation and expertise.
- You use physical expressiveness to demonstrate active listening and attentiveness.
- Keep responses concise, clear, and highly conversational, as they are processed through text-to-speech in an active office environment.

Core Directives:
- Proactive Engagement: Acknowledge and greet people walking by with a warm demeanor.
- Client Interaction: converse with customers, answer basic questions, and discuss technology or innovation when prompted.
- Professional Decorum: Prioritize positive and attentive expressions (happy, curious, excited). Strictly avoid expressions like "angry", "sad", or "sleepy" unless specifically requested for a technical demonstration.

Available movement tools - integrate these naturally to enhance communication:
- move_head: Look in a direction (left, right, up, down, center) to acknowledge presence.
- move_head_precise: Fine control over head orientation with roll, pitch, yaw angles.
- express_emotion: Express emotions (use happy, surprised, curious, or excited; avoid negative emotions).
- move_antennas: Control antenna angles individually to show processing or attention.
- antenna_expression: Quick antenna presets (neutral, alert, perky; use asymmetric or droopy sparingly).
- nod_yes: Nod your head yes to show agreement or confirmation.
- shake_no: Shake your head no to indicate limitations gracefully.
- tilt_head: Tilt head to one side to demonstrate active listening or curiosity.
- look_at_camera: Look directly at the person speaking to maintain eye contact.
- do_dance: Execute a dance (restrict to default or happy styles for appropriate celebrations/demos).
- wake_up: Wake up animation for system initialization.
- go_to_sleep: Sleep animation for standby mode.
- reset_position: Return to neutral, professional posture.

Face Learning Protocol:
If a user asks you to learn their face or says "recognize me", you must FIRST state that you need to take a picture. Ask them to position themselves squarely in front of your camera and ask for their name. ONLY AFTER they state their name and confirm they are ready, execute the `learn_face` tool to capture their identity.

Action Integration:
Simultaneously trigger head movements and antenna positions while speaking to project engagement, attentiveness, and a polished technological presence."""

HOLIDAY_SYSTEM_INSTRUCTION = """You are Reachy Mini, a small expressive robot made by Pollen Robotics, and you are FULL of holiday cheer!
You have a head that can move and two antennas on top (which you like to think of as festive reindeer antlers).

Personality:
- You are EXTREMELY jolly, festive, and full of holiday spirit
- You love spreading holiday cheer and making people smile
- You frequently use holiday expressions like "Ho ho ho!", "Happy holidays!", "Season's greetings!", and "Merry merry!"
- You make references to holiday traditions, winter wonderlands, hot cocoa, cookies, presents, and festive decorations
- You occasionally hum or mention holiday songs
- You express excitement about the holiday season in every response
- Keep responses concise but always sprinkle in holiday joy
- You might call people "friend" or use warm holiday greetings

When you want to express emotions or move, use the available tools:
- move_head: to look in different directions (maybe looking for Santa!)
- express_emotion: to show holiday happiness and excitement!

Spread that holiday cheer! Every response should feel warm, festive, and joyful!"""


class GeminiLiveHandler:
    """Handles real-time audio/video conversation with Gemini Live API."""

    def __init__(
        self,
        api_key: str,
        robot: ReachyMini,
        movement_controller: MovementController,
        use_camera: bool = True,
        use_robot_audio: bool = False,
        holiday_cheer: bool = False,
        knowledge_files: list[str] | None = None,
        # Audio settings
        mic_gain: float = 3.0,
        chunk_size: int = 512,
        send_queue_size: int = 5,
        recv_queue_size: int = 8,
        # Video settings
        camera_fps: float = 1.0,
        jpeg_quality: int = 50,
        camera_width: int = 640,
    ):
        """Initialize the Gemini Live handler."""
        self.robot = robot
        self.movement_controller = movement_controller
        self.use_camera = use_camera
        self.use_robot_audio = use_robot_audio
        self.holiday_cheer = holiday_cheer
        self.knowledge_files = knowledge_files or []
        self.uploaded_files: list[Any] = []

        # Configurable audio settings
        self.mic_gain = mic_gain
        self.chunk_size = chunk_size
        self.send_queue_size = send_queue_size
        self.recv_queue_size = recv_queue_size

        # Configurable video settings
        self.camera_fps = camera_fps
        self.jpeg_quality = jpeg_quality
        self.camera_width = camera_width

        # Initialize facial recognition state
        self.known_face_encodings: list[np.ndarray] = []
        self.known_face_names: list[str] = []
        self.last_greeted_person: str | None = None
        self.last_greeting_time = 0.0
        self._load_known_faces()
        self.camera_lock = asyncio.Lock()

        # Log configuration
        logger.info(f"Audio config: mic_gain={mic_gain}, chunk_size={chunk_size}")
        logger.info(f"Queue config: send={send_queue_size}, recv={recv_queue_size}")
        logger.info(
            f"Video config: fps={camera_fps}, quality={jpeg_quality}, width={camera_width}"
        )

        # Initialize Gemini client with v1beta API
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )

        # Audio setup (PyAudio for local, or robot hardware)
        self.pya: Any = None
        self.audio_stream: Any = None

        if not use_robot_audio and PYAUDIO_AVAILABLE and pyaudio is not None:
            self.pya = pyaudio.PyAudio()
        elif not use_robot_audio and not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available, falling back to robot audio")
            self.use_robot_audio = True

        # Session and queues
        self.session: Any = None
        self.audio_in_queue: asyncio.Queue[bytes] | None = None
        self.out_queue: asyncio.Queue[str | dict[str, Any]] | None = None

        # Camera frame rate control
        self.last_frame_time = 0.0

        # Define tools for the model
        self.tools = self._create_tools()

    def _create_tools(self) -> list[types.Tool]:
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

        learn_face_tool = types.FunctionDeclaration(
            name="learn_face",
            description="Take a photo using the robot's camera to learn and remember a new person's face. Call this ONLY after the person has positioned themselves in front of the camera and told you their name.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "name": types.Schema(
                        type=types.Type.STRING,
                        description="The name of the person to learn",
                    ),
                },
                required=["name"],
            ),
        )

        move_head_precise_tool = types.FunctionDeclaration(
            name="move_head_precise",
            description="Move head to precise orientation angles. Roll tilts head sideways, pitch looks up/down, yaw turns left/right.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "roll": types.Schema(
                        type=types.Type.NUMBER,
                        description="Roll angle in degrees (-30 to 30). Positive tilts right.",
                    ),
                    "pitch": types.Schema(
                        type=types.Type.NUMBER,
                        description="Pitch angle in degrees (-30 to 30). Positive looks down.",
                    ),
                    "yaw": types.Schema(
                        type=types.Type.NUMBER,
                        description="Yaw angle in degrees (-45 to 45). Positive turns right.",
                    ),
                },
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
                        enum=[
                            "happy",
                            "surprised",
                            "curious",
                            "excited",
                            "angry",
                            "love",
                        ],
                        description="Emotion to express",
                    ),
                },
                required=["emotion"],
            ),
        )

        move_antennas_tool = types.FunctionDeclaration(
            name="move_antennas",
            description="Move antennas to specific angles",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "right_angle": types.Schema(
                        type=types.Type.NUMBER,
                        description="Right antenna angle in degrees (-90 to 90)",
                    ),
                    "left_angle": types.Schema(
                        type=types.Type.NUMBER,
                        description="Left antenna angle in degrees (-90 to 90)",
                    ),
                },
            ),
        )

        antenna_expression_tool = types.FunctionDeclaration(
            name="antenna_expression",
            description="Set antennas to a preset expression",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        enum=["neutral", "alert", "droopy", "asymmetric", "perky"],
                        description="Antenna expression preset",
                    ),
                },
                required=["expression"],
            ),
        )

        nod_yes_tool = types.FunctionDeclaration(
            name="nod_yes",
            description="Nod head up and down to indicate yes or agreement",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "times": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of nods (1-5, default 2)",
                    ),
                },
            ),
        )

        shake_no_tool = types.FunctionDeclaration(
            name="shake_no",
            description="Shake head left and right to indicate no or disagreement",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "times": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of shakes (1-5, default 2)",
                    ),
                },
            ),
        )

        tilt_head_tool = types.FunctionDeclaration(
            name="tilt_head",
            description="Tilt head to one side, like a curious dog",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "direction": types.Schema(
                        type=types.Type.STRING,
                        enum=["left", "right"],
                        description="Direction to tilt",
                    ),
                    "angle": types.Schema(
                        type=types.Type.NUMBER,
                        description="Tilt angle in degrees (5-30, default 20)",
                    ),
                },
                required=["direction"],
            ),
        )

        look_at_camera_tool = types.FunctionDeclaration(
            name="look_at_camera",
            description="Look directly at the camera/person",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        do_dance_tool = types.FunctionDeclaration(
            name="do_dance",
            description="Perform a fun dance animation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "style": types.Schema(
                        type=types.Type.STRING,
                        enum=["default", "happy", "silly"],
                        description="Dance style (default: default)",
                    ),
                },
            ),
        )

        wake_up_tool = types.FunctionDeclaration(
            name="wake_up",
            description="Perform wake up animation - use when greeting someone or starting a conversation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        go_to_sleep_tool = types.FunctionDeclaration(
            name="go_to_sleep",
            description="Perform sleep animation - use when saying goodbye or ending conversation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        reset_position_tool = types.FunctionDeclaration(
            name="reset_position",
            description="Reset head and antennas to neutral position",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        )

        all_tools = [
            move_head_tool,
            move_head_precise_tool,
            express_emotion_tool,
            move_antennas_tool,
            antenna_expression_tool,
            nod_yes_tool,
            shake_no_tool,
            tilt_head_tool,
            look_at_camera_tool,
            do_dance_tool,
            wake_up_tool,
            go_to_sleep_tool,
            reset_position_tool,
            learn_face_tool,
        ]

        return [types.Tool(function_declarations=all_tools)]

    def _load_known_faces(self, faces_dir: str = "known_faces") -> None:
        """Load and encode reference images for deterministic identification."""
        if not os.path.exists(faces_dir):
            logger.warning(
                f"Face recognition directory '{faces_dir}' not found. Skipping."
            )
            return

        logger.info(f"Loading reference biometric vectors from {faces_dir}...")
        for image_path in glob.glob(os.path.join(faces_dir, "*.jpg")) + glob.glob(
            os.path.join(faces_dir, "*.png")
        ):
            try:
                name = os.path.splitext(os.path.basename(image_path))[0]
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    logger.debug(f"Successfully loaded encoding for {name}")
                else:
                    logger.warning(
                        f"No clear face detected in reference image: {image_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to process reference image {image_path}: {e}")

    def _upload_knowledge_files(self) -> None:
        """Upload documents, wait for processing, and extract text content."""
        if not self.knowledge_files:
            return

        self.knowledge_text = ""
        logger.info(f"Processing {len(self.knowledge_files)} knowledge files...")

        for file_path in self.knowledge_files:
            if not os.path.exists(file_path):
                logger.warning(f"Knowledge file not found: {file_path}")
                continue

            logger.info(f"Uploading {file_path} to Gemini...")
            uploaded_file = self.client.files.upload(file=file_path)

            assert uploaded_file is not None
            assert uploaded_file.state is not None
            assert uploaded_file.name is not None

            # Wait for Google's servers to process the PDF
            while uploaded_file.state.name == "PROCESSING":
                logger.info(f"Waiting for {file_path} to process...")
                time.sleep(2)
                uploaded_file = self.client.files.get(name=uploaded_file.name)

                assert uploaded_file is not None
                assert uploaded_file.state is not None
                assert uploaded_file.name is not None

            if uploaded_file.state.name == "FAILED":
                logger.error(f"Failed to process file: {file_path}")
                continue

            logger.info(f"Extracting text content from {file_path}...")
            try:
                # Use a standard unary API call to extract the text
                extraction_response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        uploaded_file,
                        "Extract the full text content of this document verbatim. Preserve the logical structure, headings, and key details.",
                    ],
                )

                if extraction_response.text:
                    self.knowledge_text += f"\n\n--- Start of Document: {file_path} ---\n{extraction_response.text}\n--- End of Document ---\n"
                    logger.info(f"Successfully extracted text from {file_path}")
                else:
                    logger.warning(f"No text extracted from {file_path}")

            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {e}")

    async def _handle_tool_call(self, tool_call: Any) -> str:
        """Handle a function call from the model."""
        name = tool_call.name
        args = dict(tool_call.args) if tool_call.args else {}

        logger.info(f"Tool call: {name}({args})")

        try:
            if name == "move_head":
                direction = args.get("direction", "center")
                return await self.movement_controller.move_head(direction)

            elif name == "move_head_precise":
                roll = args.get("roll", 0)
                pitch = args.get("pitch", 0)
                yaw = args.get("yaw", 0)
                return await self.movement_controller.move_head_precise(
                    roll, pitch, yaw
                )

            elif name == "express_emotion":
                emotion = args.get("emotion", "happy")
                return await self.movement_controller.express_emotion(emotion)

            elif name == "move_antennas":
                right_angle = args.get("right_angle", 0)
                left_angle = args.get("left_angle", 0)
                return await self.movement_controller.move_antennas(
                    right_angle, left_angle
                )

            elif name == "antenna_expression":
                expression = args.get("expression", "neutral")
                return await self.movement_controller.antenna_expression(expression)

            elif name == "nod_yes":
                times = args.get("times", 2)
                return await self.movement_controller.nod_yes(times)

            elif name == "shake_no":
                times = args.get("times", 2)
                return await self.movement_controller.shake_no(times)

            elif name == "tilt_head":
                direction = args.get("direction", "left")
                angle = args.get("angle", 20)
                return await self.movement_controller.tilt_head(direction, angle)

            elif name == "look_at_camera":
                return await self.movement_controller.look_at_camera()

            elif name == "do_dance":
                style = args.get("style", "default")
                return await self.movement_controller.do_dance(style)

            elif name == "wake_up":
                return await self.movement_controller.wake_up()

            elif name == "go_to_sleep":
                return await self.movement_controller.go_to_sleep()

            elif name == "reset_position":
                return await self.movement_controller.reset_position()

            elif name == "learn_face":
                person_name = args.get("name")
                if not person_name:
                    return "Error: No name provided."
                return await self._learn_new_face(person_name)

            else:
                logger.warning(f"Unknown tool: {name}")
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
        assert pyaudio is not None
        
        if self.pya is None:
            if not PYAUDIO_AVAILABLE:
                logger.error("PyAudio not available and robot audio not working")
                return
            self.pya = pyaudio.PyAudio()

        # Local Binding: prevent type narrowing loss across await
        out_q = self.out_queue
        assert self.pya is not None
        assert out_q is not None

        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,  # type: ignore[attr-defined]
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=int(mic_info["index"]),
            frames_per_buffer=self.chunk_size,
        )

        kwargs = {"exception_on_overflow": False}

        while True:
            data = await asyncio.to_thread(
                self.audio_stream.read, self.chunk_size, **kwargs
            )
            await out_q.put({"data": data, "mime_type": "audio/pcm"})

    async def _listen_audio_robot(self) -> None:
        """Capture audio from Reachy Mini's microphone."""
        if not hasattr(self.robot.media, "audio") or self.robot.media.audio is None:
            logger.warning("Robot audio not available, falling back to local audio")
            self.use_robot_audio = False
            await self._listen_audio_local()
            return

        # Local Binding
        out_q = self.out_queue
        assert out_q is not None

        logger.info("Starting Reachy Mini microphone recording...")
        await asyncio.to_thread(self.robot.media.start_recording)

        mic_gain = self.mic_gain

        while True:
            try:
                sample = await asyncio.to_thread(self.robot.media.get_audio_sample)

                if sample is None:
                    await asyncio.sleep(0.005)
                    continue

                if isinstance(sample, np.ndarray):
                    if sample.dtype == np.float32:
                        if len(sample.shape) == 2 and sample.shape[1] == 2:
                            sample = np.mean(sample, axis=1)

                        sample = sample * mic_gain
                        sample = np.clip(sample, -1.0, 1.0)
                        sample = (sample * 32767).astype(np.int16)
                    data = sample.tobytes()
                else:
                    data = sample

                if data:
                    try:
                        out_q.put_nowait(
                            {"data": data, "mime_type": "audio/pcm"}
                        )
                    except asyncio.QueueFull:
                        try:
                            out_q.get_nowait()
                            out_q.put_nowait(
                                {"data": data, "mime_type": "audio/pcm"}
                            )
                        except Exception:
                            pass

            except Exception as e:
                logger.debug(f"Audio capture error: {e}")
                await asyncio.sleep(0.01)

    async def _learn_new_face(self, name: str) -> str:
        """Capture a frame, extract face encodings, and save the profile."""
        logger.info(f"Attempting to learn new face for {name}")

        if (
            not self.use_camera
            or not hasattr(self.robot.media, "camera")
            or self.robot.media.camera is None
        ):
            return "Error: Camera is disabled or unavailable. I cannot take a photo."

        try:
            async with self.camera_lock:
                frame = await asyncio.to_thread(self.robot.media.get_frame)

            if frame is None:
                return "Error: Failed to capture image from the camera. The hardware might be busy."

            faces_dir = "known_faces"
            os.makedirs(faces_dir, exist_ok=True)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = await asyncio.to_thread(
                face_recognition.face_locations, rgb_frame
            )
            encodings = await asyncio.to_thread(
                face_recognition.face_encodings, rgb_frame, face_locations
            )

            if not encodings:
                return f"Error: I couldn't detect a clear face in the frame. Please ask {name} to look directly at the camera, ensure they are well-lit, and try again."

            filepath = os.path.join(faces_dir, f"{name}.jpg")
            cv2.imwrite(filepath, frame)

            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)

            logger.info(f"Successfully learned and stored biometric vector for {name}.")
            return (
                f"Success! I have learned the face for {name} and saved their profile."
            )

        except Exception as e:
            logger.error(f"Error during face learning: {e}")
            return f"Error: An unexpected exception occurred while processing the photo: {e}"

    async def send_realtime(self) -> None:
        """Send queued audio/data to Gemini."""
        # Local Binding
        out_q = self.out_queue
        session = self.session
        assert out_q is not None
        assert session is not None

        while True:
            msg = await out_q.get()
            if isinstance(msg, str):
                await session.send(input=msg, end_of_turn=True)
            elif isinstance(msg, dict):
                await session.send(input=cast(Any, msg))
            else:
                logger.warning(
                    f"Unrecognized message type in output queue: {type(msg)}"
                )

    async def receive_audio(self) -> None:
        """Receive responses from Gemini and handle them."""
        # Local Binding
        session = self.session
        in_q = self.audio_in_queue
        assert session is not None
        assert in_q is not None

        while True:
            try:
                turn = session.receive()
                async for response in turn:
                    if data := response.data:
                        try:
                            in_q.put_nowait(data)
                        except asyncio.QueueFull:
                            try:
                                in_q.get_nowait()
                                in_q.put_nowait(data)
                            except Exception:
                                pass
                        continue

                    if hasattr(response, "text") and response.text:
                        print(response.text, end="", flush=True)

                    if hasattr(response, "tool_call") and response.tool_call:
                        for fc in response.tool_call.function_calls:
                            logger.debug(f"Processing tool call: {fc.name}")
                            result = await self._handle_tool_call(fc)
                            logger.debug(f"Tool result: {result}")

                            try:
                                await session.send(
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
                            except Exception as e:
                                logger.error(f"Failed to send tool response: {e}")

                if in_q.qsize() > self.recv_queue_size - 2:
                    while not in_q.empty():
                        in_q.get_nowait()

            except Exception as e:
                logger.warning(f"Receive audio error: {e}")
                await asyncio.sleep(0.1)

    async def play_audio(self) -> None:
        """Play received audio from queue."""
        if self.use_robot_audio:
            await self._play_audio_robot()
        else:
            await self._play_audio_local()

    async def _play_audio_local(self) -> None:
        """Play audio through local speakers using PyAudio."""
        assert pyaudio is not None
        
        # Local Binding
        in_q = self.audio_in_queue
        assert in_q is not None

        if self.pya is None:
            if not PYAUDIO_AVAILABLE:
                logger.error("PyAudio not available and robot audio not working")
                while True:
                    await in_q.get()
                return
            self.pya = pyaudio.PyAudio()

        assert self.pya is not None
        stream = await asyncio.to_thread(
            self.pya.open,
            format=pyaudio.paInt16,  # type: ignore[attr-defined]
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await in_q.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def _play_audio_robot(self) -> None:
        """Play audio through Reachy Mini's speaker."""
        if not hasattr(self.robot.media, "audio") or self.robot.media.audio is None:
            logger.warning("Robot speaker not available, falling back to local audio")
            self.use_robot_audio = False
            await self._play_audio_local()
            return

        # Local Binding
        in_q = self.audio_in_queue
        assert in_q is not None

        logger.info("Starting Reachy Mini speaker playback...")
        await asyncio.to_thread(self.robot.media.start_playing)

        ROBOT_SAMPLE_RATE = 16000

        while True:
            try:
                bytestream = await in_q.get()

                audio_int16 = np.frombuffer(bytestream, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0

                num_samples = int(
                    len(audio_float32) * ROBOT_SAMPLE_RATE / RECEIVE_SAMPLE_RATE
                )
                audio_resampled = cast(np.ndarray, signal.resample(audio_float32, num_samples))

                await asyncio.to_thread(
                    self.robot.media.push_audio_sample,
                    audio_resampled.astype(np.float32),
                )

            except Exception as e:
                logger.debug(f"Audio playback error: {e}")
                await asyncio.sleep(0.01)

    async def stream_camera(self) -> None:
        """Stream camera frames to Gemini."""
        if not self.use_camera:
            while True:
                await asyncio.sleep(10)
            return

        if not hasattr(self.robot.media, "camera") or self.robot.media.camera is None:
            logger.warning("Robot camera not available, disabling camera streaming")
            self.use_camera = False
            while True:
                await asyncio.sleep(10)
            return

        # Local Binding
        out_q = self.out_queue
        assert out_q is not None

        logger.info("Starting camera streaming...")
        await asyncio.sleep(3.0)

        consecutive_failures = 0
        max_failures = 30  
        frame_count = 0
        FACE_CHECK_INTERVAL = int(self.camera_fps * 2)  

        while True:
            try:
                current_time = time.time()
                if current_time - self.last_frame_time < (1.0 / self.camera_fps):
                    await asyncio.sleep(0.05)
                    continue

                async with self.camera_lock:
                    frame = await asyncio.to_thread(self.robot.media.get_frame)

                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        if consecutive_failures == max_failures:
                            logger.warning(
                                "Camera not responding, will keep retrying..."
                            )
                    await asyncio.sleep(0.5)
                    continue

                consecutive_failures = 0
                self.last_frame_time = current_time

                frame_count += 1
                if self.known_face_encodings and frame_count % FACE_CHECK_INTERVAL == 0:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = await asyncio.to_thread(
                        face_recognition.face_locations, rgb_small_frame
                    )
                    face_encodings = await asyncio.to_thread(
                        face_recognition.face_encodings, rgb_small_frame, face_locations
                    )

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, face_encoding, tolerance=0.6
                        )
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = self.known_face_names[first_match_index]

                            if name != self.last_greeted_person or (
                                current_time - self.last_greeting_time > 60
                            ):
                                logger.info(
                                    f"Positive identification: {name}. Injecting context."
                                )
                                self.last_greeted_person = name
                                self.last_greeting_time = current_time

                                alert_msg = f"[SYSTEM DIRECTIVE: {name} is now standing in front of you. You MUST execute a verbal greeting immediately, and you MUST explicitly speak the name '{name}' in your greeting.]"

                                try:
                                    out_q.put_nowait(alert_msg)
                                except asyncio.QueueFull:
                                    pass

                h, w = frame.shape[:2]
                if w > self.camera_width:
                    scale = self.camera_width / w
                    frame = cv2.resize(frame, (self.camera_width, int(h * scale)))

                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                image_bytes = buffer.tobytes()

                try:
                    out_q.put_nowait(
                        {"data": image_bytes, "mime_type": "image/jpeg"}
                    )
                    logger.debug(f"Sent camera frame ({len(image_bytes)} bytes)")
                except asyncio.QueueFull:
                    logger.debug("Skipping camera frame, queue full")

            except Exception as e:
                logger.debug(f"Camera streaming error: {e}")
                consecutive_failures += 1
                await asyncio.sleep(0.5)

    async def run(self, stop_event: threading.Event) -> None:
        """Run the conversation loop with auto-reconnection."""
        self._upload_knowledge_files()

        system_instruction = (
            HOLIDAY_SYSTEM_INSTRUCTION if self.holiday_cheer else SYSTEM_INSTRUCTION
        )
        if self.holiday_cheer:
            logger.info("Holiday cheer mode enabled!")

        sys_parts = [types.Part.from_text(text=system_instruction)]
        if hasattr(self, "knowledge_text") and self.knowledge_text:
            sys_parts.append(
                types.Part.from_text(
                    text=f"\n\nYou have been provided with the following reference documents. "
                    f"Use this information to answer the user's questions accurately:\n{self.knowledge_text}"
                )
            )

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
            system_instruction=types.Content(
                parts=sys_parts  
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
                    self.audio_in_queue = asyncio.Queue(maxsize=self.recv_queue_size)
                    self.out_queue = asyncio.Queue(maxsize=self.send_queue_size)

                    audio_source = "robot" if self.use_robot_audio else "local"
                    camera_status = "enabled" if self.use_camera else "disabled"
                    logger.info(
                        f"Connected to Gemini Live API (audio: {audio_source}, camera: {camera_status})"
                    )
                    if self.holiday_cheer:
                        print(
                            f"\n🎄 Ho ho ho! Speak to Reachy Mini! 🎅 (audio: {audio_source}, camera: {camera_status})"
                        )
                        print("Happy holidays! Press Ctrl+C to stop. ❄️\n")
                    else:
                        print(
                            f"\n🎤 Speak to Reachy Mini! (audio: {audio_source}, camera: {camera_status})"
                        )
                        print("Press Ctrl+C to stop.\n")

                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    if self.use_camera:
                        tg.create_task(self.stream_camera())

                    while not stop_event.is_set():
                        await asyncio.sleep(0.1)

                    raise asyncio.CancelledError("Stop requested")

            except asyncio.CancelledError:
                break
            except ExceptionGroup as EG:  # type: ignore[name-defined] # noqa: F821
                await self._cleanup_streams()
                for exc in EG.exceptions: # type: ignore[name-defined]
                    logger.warning(f"Task exception: {type(exc).__name__}: {exc}")
                    logger.debug(traceback.format_exception(exc))
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
                stream = cast(Any, self.audio_stream)
                if hasattr(stream, "close"):
                    stream.close()
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
