# Reachy Mini Gemini Live App

Talk with Reachy Mini using Google's Gemini Live API for real-time voice conversations.

## Features

- Real-time voice conversation via Gemini Live API
- **Camera streaming**: Send robot's camera feed to Gemini for vision capabilities
- **Robot audio**: Use Reachy Mini's microphone and speaker (or local audio)
- Expressive head movements and antenna animations
- Function calling for robot control (move head, express emotions)
- Support for both wired and wireless Reachy Mini

## Requirements

- Python 3.10+
- Reachy Mini robot (or simulation)
- Google API key with Gemini access
- PortAudio (for PyAudio)

## Installation

```bash
# Install PortAudio (macOS)
brew install portaudio

# Install PortAudio (Ubuntu)
sudo apt-get install portaudio19-dev

# Install the app (basic - local audio only)
pip install -e .

# Install with wireless/GStreamer support (for robot audio/camera over network)
pip install -e ".[wireless]"
```

### Wireless Media Support

To use the robot's camera and microphone over wireless, you need GStreamer:

```bash
# macOS
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly

# Ubuntu
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

## Configuration

1. Copy `.env.example` to `.env`
2. Add your Google API key:
   ```
   GOOGLE_API_KEY=your-key-here
   ```

## Usage

First, make sure the Reachy Mini daemon is running:
```bash
# For simulation
reachy-mini-daemon --sim

# For real hardware (wired)
reachy-mini-daemon

# For wireless Reachy Mini (on the robot)
reachy-mini-daemon --wireless-version
```

Then run the app:
```bash
# Basic usage (local mic/speaker, robot camera enabled)
reachy-mini-gemini

# Wireless Reachy Mini
reachy-mini-gemini --wireless

# Use robot's microphone and speaker (instead of local audio)
reachy-mini-gemini --robot-audio

# Full robot mode (robot camera + robot audio)
reachy-mini-gemini --wireless --robot-audio

# Without camera
reachy-mini-gemini --no-camera

# Debug mode
reachy-mini-gemini --debug
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--wireless` | Connect to wireless Reachy Mini |
| `--robot-audio` | Use Reachy Mini's microphone and speaker |
| `--no-camera` | Disable camera streaming to Gemini |
| `--debug` | Enable debug logging |

## Available Tools

The Gemini model can use these tools during conversation:

| Tool | Description |
|------|-------------|
| `move_head` | Look left, right, up, down, or center |
| `express_emotion` | Express happy, sad, surprised, curious, excited, or sleepy |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Reachy Mini                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────────────┐   │
│  │ Camera  │  │   Mic   │  │ Speaker │  │ Head + Antennas│   │
│  └────┬────┘  └────┬────┘  └────▲────┘  └───────▲────────┘   │
└───────┼────────────┼────────────┼───────────────┼────────────┘
        │            │            │               │
        │ (JPEG)     │ (PCM 16k)  │ (PCM 24k)     │
        ▼            ▼            │               │
┌───────────────────────────────────────────────────────────┐
│                    Gemini Live API                        │
│  - Real-time audio conversation                           │
│  - Vision understanding (camera frames)                   │
│  - Function calling (robot control)                       │
└────────────────────────┬──────────────────────────────────┘
                         │
                ┌────────▼─────────┐
                │   Tool Calls     │
                │  move_head       │
                │  express_emotion │
                └──────────────────┘
```

### Audio Modes

- **Local audio (default)**: Uses your computer's microphone and speakers
- **Robot audio (`--robot-audio`)**: Uses Reachy Mini's built-in mic and speaker

## License

Apache 2.0
