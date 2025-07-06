# Whisper Live & File Transcription

This project provides a Python script that uses [OpenAI Whisper](https://github.com/openai/whisper) to transcribe either a live audio stream from your microphone or a provided audio file. It supports multiple Whisper model sizes and languages.

## Features

- **Live Transcription:** Continuously transcribe audio from your microphone in real-time.
- **File Transcription:** Transcribe an existing audio file.
- **Multiple Models & Languages:** Select from various Whisper models (`tiny`, `base`, `small`, `medium`, `large`) and specify a language for transcription.

## Requirements

- **Operating System:** Linux, macOS, or Windows (with compatible audio drivers)
- **Python Version:** Python 3.8 or later
- **Hardware:** 
  - CPU-only runs are possible but slower. 
  - GPU (NVIDIA) recommended for faster inference (ensure CUDA and drivers are installed).

## Prerequisites

1. **Python & Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Linux/macOS
   venv\Scripts\activate         # On Windows PowerShell
