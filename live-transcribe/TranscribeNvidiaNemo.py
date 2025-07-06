import nemo.collections.asr as nemo_asr
import sounddevice as sd
import numpy as np
import torch

# 🔹 Load the Dutch ASR model (large model for best accuracy)
model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_nl_conformer_ctc_large")
model.to("cuda")  # Use NVIDIA GPU

# 🔹 Ensure model is set to Dutch (not needed explicitly, but avoids confusion)
print("✅ NeMo ASR Model Loaded: stt_nl_conformer_ctc_large (Dutch)")

# Audio recording settings
SAMPLERATE = 16000  # Required for NeMo
CHANNELS = 1
DURATION = 3  # Record in 3-second chunks

print("🎤 Start speaking in Dutch... (Press Ctrl+C to stop)")

try:
    while True:
        # Record audio from the microphone
        audio_data = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype="float32")
        sd.wait()

        # Convert audio to NumPy format for NeMo
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).cuda()

        # 🔹 Transcribe using the Dutch model
        transcription = model.transcribe([audio_tensor.cpu().numpy()])

        print("📝 Je zei:", transcription[0])  # Print the transcribed Dutch text

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
