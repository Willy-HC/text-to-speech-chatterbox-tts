import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo....")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device = "cpu"
)
texto = "Bienvenidos a inteligencia artificial generativa vamos a crear imagene, audios , texto y si el pc aguanta video"
print("Generando audio....")

audio = modelo.generate(
    texto,
    language_id = "es",
    exaggeration = 0.7,#Expresión o emoción
    cfg_weight = 0.4,#Nos ajusta el ritmo, si la voz va rápida
    temperature = 0.8,#Controla la variabilidad, más alta= más variabilidad, menos
    repetition_penalty = 1.2,#Penaliza repeticiones extrañas
    min_p = 0.05, #Filtra opciones improbables durante la generación
    top_p = 1.0 #Controla el muestreo probabilistico
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio guardado")
