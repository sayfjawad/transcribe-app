import whisper
import os
import json
import torchaudio

AUDIO_DIR = "../input_wavs"
OUTPUT_JSON = "../input_wavs/transcriptions/metadata.json"

model = whisper.load_model("large", device="cuda")
os.makedirs("transcripties", exist_ok=True)

metadata_list = []

for filename in os.listdir(AUDIO_DIR):
    if filename.lower().endswith(".wav"):
        audio_path = os.path.join(AUDIO_DIR, filename)
        print(f"🔍 Transcriberen: {filename}")

        # Bereken duur van audio (optioneel)
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.size(1) / sample_rate

        # Transcribe met segmenten
        result = model.transcribe(audio_path, language="nl", verbose=False, word_timestamps=False)

        # Metadata object maken
        entry = {
            "audio_filepath": audio_path,
            "text": result["text"].strip(),
            "language": "nl",
            "duration": round(duration, 2),
            "segments": [
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip()
                }
                for seg in result["segments"]
            ]
        }

        metadata_list.append(entry)

# Alles wegschrijven naar JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

print("✅ Alles getranscribeerd met metadata opgeslagen in metadata.json")
