import json
import os
import pandas as pd

# Pad naar je metadata-bestand
METADATA_PATH = "input_wavs/transcriptions/metadata.json"

# Pad naar output-bestanden
OUTPUT_CSV = "coqui_manifest.csv"
OUTPUT_JSONL = "coqui_manifest.jsonl"

def create_coqui_manifest(metadata_path, out_csv, out_jsonl):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = []
    for item in metadata:
        audio_path = item["audio_filepath"].replace("../", "")  # Maak pad relatief vanuit scriptlocatie
        text = item["text"].strip()
        duration = round(item.get("duration", 0), 2)

        # Sla leeg of verdacht korte zinnen over
        if len(text) < 3:
            continue

        records.append({
            "audio_filepath": audio_path,
            "text": text,
            "duration": duration
        })

    # Opslaan als CSV
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    # Opslaan als JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Manifest opgeslagen als:\n- {out_csv}\n- {out_jsonl}\n📝 Aantal regels: {len(records)}")

if __name__ == "__main__":
    create_coqui_manifest(METADATA_PATH, OUTPUT_CSV, OUTPUT_JSONL)
