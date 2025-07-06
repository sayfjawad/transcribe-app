import os
import whisper

input_dir = "input_wavs"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Laad Whisper-model ("base" is snel, "medium"/"large" zijn nauwkeuriger)
#model = whisper.load_model("base")
model = whisper.load_model("base", device="cpu")

metadata_entries = []

# Sorteer op nummer als dat kan, anders alfabetisch
def sort_key(filename):
    stem = os.path.splitext(filename)[0]
    return int(stem) if stem.isdigit() else stem

for filename in sorted(os.listdir(input_dir), key=sort_key):
    if filename.lower().endswith(".wav"):
        wav_path = os.path.join(input_dir, filename)
        print(f"Transcribing: {wav_path}")
        result = model.transcribe(wav_path)
        text = result["text"].strip()

        # Output naar txt-bestand
        txt_filename = f"{os.path.splitext(filename)[0]}.txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Transcriptie opgeslagen: {txt_path}")

        # Zet regel klaar voor metadata
        metadata_entries.append(f"{os.path.join(input_dir, filename)}|{text}")

# Schrijf metadata.csv
metadata_path = os.path.join(output_dir, "metadata.csv")
with open(metadata_path, "w", encoding="utf-8") as meta:
    for line in metadata_entries:
        meta.write(line + "\n")
print(f"Metadata bestand aangemaakt: {metadata_path}")

print("Klaar! Alle bestanden getranscribeerd en metadata gegenereerd.")

