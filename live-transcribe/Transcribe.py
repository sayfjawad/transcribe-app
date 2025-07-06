import speech_recognition as sr

recognizer = sr.Recognizer()
audio_file = "/home/sayf/Videos/Groeien/audio6.wav"  # Vervang met jouw bestandspad

with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)
    transcription = recognizer.recognize_google(audio_data, language="nl-NL")

print(transcription)
