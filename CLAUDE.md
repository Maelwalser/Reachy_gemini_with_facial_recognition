# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Reachy Mini Gemini App enables real-time voice/video conversations with Reachy Mini robot using Google's Gemini Live API.

## Common Commands

```bash
# Install (basic - local audio)
pip install -e .

# Install with wireless/GStreamer support
pip install -e ".[wireless]"

# Run with local audio + robot camera
reachy-mini-gemini

# Run with wireless connection
reachy-mini-gemini --wireless

# Run with robot's microphone and speaker
reachy-mini-gemini --robot-audio

# Full robot mode (wireless + robot audio)
reachy-mini-gemini --wireless --robot-audio

# Debug mode
reachy-mini-gemini --debug
```

## Architecture

```
User speaks -> Mic (local/robot) -> Gemini Live API -> Audio response -> Speaker (local/robot)
                                         ^
                                         |
Robot camera -> JPEG frames -------------+
                                         |
                                         v
                              Tool calls (move_head, express_emotion)
                                         |
                                         v
                              MovementController -> ReachyMini SDK
```

## Key Files

- `reachy_mini_gemini_app/main.py` - CLI entry point, ReachyMiniApp class
- `reachy_mini_gemini_app/gemini_handler.py` - Gemini Live API integration
- `reachy_mini_gemini_app/movements.py` - Robot movement controller

## Audio Configuration

- Send to Gemini: 16kHz, mono, 16-bit PCM
- Receive from Gemini: 24kHz, mono, 16-bit PCM

## Media Backends

- `default` - OpenCV camera, works for wired connections
- `gstreamer` - GStreamer WebRTC, for wireless media streaming
- `webrtc` - Alternative WebRTC backend
- `no_media` - No camera/robot audio support

## Dependencies

- `reachy_mini` - Reachy Mini SDK
- `google-genai` - Gemini API client
- `pyaudio` - Local audio I/O
- `opencv-python` - Camera frame processing
- `reachy_mini[gstreamer]` - For wireless media (optional)
