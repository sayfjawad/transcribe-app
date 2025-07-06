import whisper
import sounddevice as sd
import numpy as np
import torch
import wave

# Load Whisper model (use "medium" or "large" for better accuracy)
model = whisper.load_model("medium", device="cuda")

# Audio settings
SAMPLERATE = 16000  # Required sample rate
CHANNELS = 1
DURATION = 3  # Record in 3-second chunks

print("🎤 Start speaking in Dutch... (Press Ctrl+C to stop)")

try:
    while True:
        # Record audio from the microphone
        audio_data = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype="int16")
        sd.wait()

        # Save as temporary WAV file
        with wave.open("temp_audio.wav", "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLERATE)
            wf.writeframes(audio_data.tobytes())

        # Transcribe using Whisper
        result = model.transcribe("temp_audio.wav", language="nl")  # Force Dutch

        print("📝 Je zei:", result["text"])

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
