import whisper
import sounddevice as sd
import numpy as np
import torch
import wave

# Load Whisper model (medium or large recommended)
model = whisper.load_model("large", device="cuda")

# Audio recording settings
SAMPLERATE = 16000  # Whisper requires 16kHz
CHANNELS = 1
DURATION = 3  # Record in 3-second chunks

print("🎤 Start speaking... (Press Ctrl+C to stop)")

try:
    while True:
        audio_data = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype="int16")
        sd.wait()

        # Save as temporary WAV file
        with wave.open("temp_audio.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLERATE)
            wf.writeframes(audio_data.tobytes())

        # Transcribe with Whisper
        result = model.transcribe("temp_audio.wav", language="nl")

        print("📝 You said:", result["text"])

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
