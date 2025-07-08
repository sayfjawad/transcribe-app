for f in *.mkv; do
  ffmpeg -i "$f" -vn -acodec pcm_s16le -ar 44100 -ac 1 "${f%.mkv}.wav"
done

