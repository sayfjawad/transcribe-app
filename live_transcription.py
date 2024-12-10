import sounddevice as sd
import numpy as np
import whisper
import torch
import tempfile
import os
from scipy.io.wavfile import write
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Live and file transcription using Whisper.")
parser.add_argument("--language", type=str, default="en", help="Language for transcription (e.g., 'en' for English, 'nl' for Dutch).")
parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model to use for transcription.")
parser.add_argument("--file", type=str, help="Path to an audio file for transcription.")
args = parser.parse_args()

# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model '{args.model}' on {device}...")
model = whisper.load_model(args.model, device=device)

# Temporary file for live audio chunks
temp_dir = tempfile.TemporaryDirectory()
temp_audio_file = os.path.join(temp_dir.name, "live_audio.wav")


def transcribe_file(file_path):
    """Transcribe an audio file."""
    print(f"Transcribing file: {file_path}")
    result = model.transcribe(file_path, language=args.language)
    print(f"Transcription: {result['text']}")


def callback(indata, frames, time, status):
    """Handle live audio input and transcription."""
    if status:
        print(f"Status: {status}")

    # Save audio data to a temporary file
    write(temp_audio_file, 16000, (indata * 32767).astype(np.int16))

    # Transcribe the audio chunk
    result = model.transcribe(temp_audio_file, language=args.language)
    print(f"Transcription: {result['text']}")


def live_transcription():
    """Perform live audio transcription."""
    print("Listening... Press Ctrl+C to stop.")
    with sd.InputStream(samplerate=16000, channels=1, callback=callback, ):
        while True:
            pass


# Main logic
try:
    if args.file:
        # Transcribe the specified file
        transcribe_file(args.file)
    else:
        # Perform live transcription
        live_transcription()
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    temp_dir.cleanup()
