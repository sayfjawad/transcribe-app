from TTS.api import TTS
import soundfile as sf
import numpy as np

# Laad het XTTS-model (je hebt al tts_models/multilingual/multi-dataset/xtts_v2 gedownload)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")  # Zet het model expliciet op de GPU

# Jouw referentie (stemvoorbeeld)
speaker_wav = "/home/sayf/git/transcribe-app/input_wavs/8.wav"

# Maak grotere tekstblokken
text_blocks = [
    "Van Kudde naar Ziel – Waarom Zelf Denken Pijnlijk Maar Nodig Is",
    "de podcast waarin we samen de weg verkennen van onbewuste patronen naar bewuste keuzes.",
    "Mijn naam is Seef, en vandaag wil ik het met je hebben over iets dat misschien wat raar klinkt...",
    "maar wel iets losmaakt",
    "de Niet Speelbare karakter-mentaliteit in onze samenleving.",
    
    "In computerspellen zijn N S K ’s — Niet Speelbare karakter — de figuren die niet door spelers worden bestuurd.",
    "Ze lopen rond, zeggen steeds hetzelfde, volgen hun script.",
    "Ze reageren niet op jouw unieke keuzes — ze herhalen gewoon.",
    
    "En weet je?",
    "Soms lijkt het alsof een groot deel van de mensen om ons heen — of zelfs wijzelf op bepaalde momenten — zich precies zo gedragen.",
    "Alsof we op de automatische piloot leven.",
    "Alsof we niet meer zelf denken, maar alleen nog herhalen wat we hebben gehoord: in de media, van beroemdheden, van autoriteiten.",
    
    "Onderzoekers en denkers noemen dit wel de consensusstaat. Ongeveer 70% van de mensen leeft hierin.",
    "Niet per se omdat ze dom zijn, maar omdat het veilig voelt.",
    "We zoeken houvast in wat “de meeste mensen” vinden. Want afwijken? Dat is spannend. En soms zelfs pijnlijk.",
    
    "Wanneer mensen in deze staat geconfronteerd worden met andere ideeën, kunnen ze fel reageren.",
    "Niet omdat jouw mening hen echt raakt — maar omdat het hun gevoel van veiligheid bedreigt.",
    "Hun overtuigingen zijn namelijk niet zomaar meningen. Het zijn onbewuste ankers van zekerheid.",
    
    
    "Maar wat als jij voelt dat er meer is?",
    "Wat als jij merkt dat er iets in jou wringt, dat zich niet meer laat verklaren door 'de experts'?",
    "Dan begint iets wat ik een innerlijke roep noem.",
    "Het begin van je eigen pad — weg van de kudde. Richting jouw ware zelf.",
    
    "En laat me eerlijk zijn... dat pad is niet makkelijk.",
    "Het is verwarrend. Vervreemdend. Soms ronduit pijnlijk.",
    
    "Want ineens moet jij leren jezelf te vertrouwen. Niet meer leunen op de buitenwereld, maar op je eigen gevoel.",
    "En dat betekent dat je dingen moet loslaten. Overtuigingen. Zekerheden. Mensen, soms.",
    
    
    "Maar het is ook het begin van vrijheid.",
    "Want daar, in dat niemandsland tussen wie je dacht te zijn en wie je werkelijk bent, ",
    "vind je iets dat niemand je ooit kan geven: je ziel.",
    
    
    "Je kunt anderen niet dwingen om wakker te worden.",
    "Wakker worden is geen project. Het is een innerlijke beslissing.",
    "En velen zullen die beslissing jaren — misschien wel levenslang — uitstellen. En dat is oké.",
    "Want ook dat is een fase in iemands zielsontwikkeling.",
    
    
    "Wat kun jij wél doen?",
    "Jezelf ontwikkelen.",
    "Je eigen pad bewandelen.",
    "Iets unieks in jezelf ontdekken — iets wat alleen jij kunt brengen.",
    "En dát teruggeven aan de wereld.",
    
    "Want uiteindelijk zijn het de meest rijpe, bewuste zielen…",
    "die het nieuwe collectieve bewustzijn vormen.",
    "Niet door te forceren, maar door te stralen.",
    "Niet door te schreeuwen, maar door te zijn.",

    "Dank je wel voor het luisteren naar deze aflevering van Van Kudde naar Ziel.",
    "Voel je vrij om deze aflevering te delen met iemand die dieper durft te voelen.",
    "En onthoud: jouw bewustzijn is het begin van verandering."
]

# Genereer audiofragmenten voor elk blok
audio_clips = []
sample_rate = None

for i, block in enumerate(text_blocks):
    wav = tts.tts(
        text=block,
        speaker_wav=speaker_wav,
        language="nl"
    )
    audio_clips.append(wav)
    if sample_rate is None:
        sample_rate = tts.synthesizer.output_sample_rate

# Plak de fragmenten aan elkaar
final_audio = np.concatenate(audio_clips)

# Opslaan als .wav
sf.write("nsk_podcast_xtts.wav", final_audio, sample_rate)
print("Bestand opgeslagen als nsk_podcast_xtts.wav")

