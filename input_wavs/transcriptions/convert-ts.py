import json
import os

# Load your metadata
with open('metadata.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Sort files by filename (to keep things reproducible, or sort as needed)
data.sort(key=lambda x: os.path.basename(x['audio_filepath']))

# Write transcript and list wavs
with open('transcript.txt', 'w', encoding='utf-8') as txtf:
    for item in data:
        txtf.write(item['text'].strip() + '\n')

# (Optional) If you want to move or copy the wavs, do it here.
# For OpenVoice: Place the WAVs in the same directory, or update your paths accordingly.
