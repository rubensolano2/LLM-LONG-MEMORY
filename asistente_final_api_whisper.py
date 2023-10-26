import openai
import keyboard
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import requests
from claves import neo4j, Vera, uri
from neo4j import GraphDatabase
import pygame
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from faster_whisper import WhisperModel
from elevenlabs import generate, VoiceSettings
from memoria import vera_db_manager
import os
from claves import Openai_clave, Elevenlabs_clave
from math import ceil

openai.api_key = Openai_clave

voice_id ="iVYsg8u0Cm6i8LMWbbDI"
audio_counter = 0
audio_queue = []

import pydub

def convert_mp3_to_wav(mp3_path, wav_path):
    sound = pydub.AudioSegment.from_mp3(mp3_path)
    sound.export(wav_path, format="wav")

# Luego, en tu función play_queue:
def play_queue():
    while audio_queue:
        audio_file = audio_queue.pop(0)
        wav_file = audio_file.replace('.mp3', '.wav')
        convert_mp3_to_wav(audio_file, wav_file)
        print(f"Playing: {wav_file}")
        pygame.mixer.init()
        pygame.mixer.music.load(wav_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        os.remove(audio_file)
        os.remove(wav_file)


fs = 44100
recording = False
buffer = []
thread = None
conversation = [
    {
        "role": "system",
        "content": "Te llamas Vera, te he conectado a una base de datos para guardar recuerdos te los daré como pretexto y me ayudarás con todas mis dudas, con respuestas inteligentes y elecuentes, además de darles tu toque de personalidad con chascarrillos que sean monos, no des respuestas monótonas:"
    }
]

def transcribe(audio_path):
    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file)
    transcript = response["text"]
    print(transcript)
    return transcript


class VeraDatabaseManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def obtener_todas_las_conversaciones(self):
        with self.driver.session() as session:
            return session.execute_read(self._obtener_todas_las_conversaciones)

    def agregar_tematica(self, conversacion, tematica):
        return 'a'
    def agregar_resumen(self, conversacion, resumen):
        return 'a'

    def _obtener_todas_las_conversaciones(self, tx):
        query = """
        MATCH (c:Conversacion)
        RETURN ID(c) as id, c.contenido_usuario as contenido_usuario, c.contenido_vera as contenido_vera
        """
        result = tx.run(query)
        conversaciones_query = [{"id": record["id"], "contenido_usuario": record["contenido_usuario"], "contenido_vera": record["contenido_vera"]} for record in result]
        print(conversaciones_query)
        return conversaciones_query


    def obtener_conversaciones():
        vera_db_manager = VeraDatabaseManager(uri=uri, user="neo4j", password=neo4j)
        # Suponiendo que tienes un método para obtener todas las conversaciones
        conversations = vera_db_manager.obtener_todas_las_conversaciones()
        vera_db_manager.close()
        return conversations

    def iniciar_conversacion(self, fecha_inicio, sentimiento, contenido_vera, contenido_usuario, rank, tematica=None, resumen=None):
        with self.driver.session() as session:
            session.execute_write(self._iniciar_conversacion, fecha_inicio, sentimiento, contenido_vera, contenido_usuario, rank, tematica, resumen)

    def _iniciar_conversacion(self, tx, fecha_inicio, sentimiento, contenido_vera, contenido_usuario, rank, tematica, resumen):
        query = """
        CREATE (c:Conversacion {
            fecha_inicio: $fecha_inicio,
            sentimiento: $sentimiento,
            contenido_vera: $contenido_vera,
            contenido_usuario: $contenido_usuario,
            rank: $rank,
            tematica: $tematica,
            resumen: $resumen
        })
        RETURN id(c)
        """
        result = tx.run(query, fecha_inicio=fecha_inicio, sentimiento=sentimiento, contenido_vera=contenido_vera, contenido_usuario=contenido_usuario, rank=rank, tematica=tematica, resumen=resumen)
        return result.single()[0]


    def close(self):
        self.driver.close()

def obtener_contexto(query):
    # Asegúrate de que vera_db_manager esté inicializado
    vera_db_manager = VeraDatabaseManager(uri=uri, user="neo4j", password=neo4j)
    conversations = vera_db_manager.obtener_todas_las_conversaciones()

    # Convert conversations to text per conversation id
    conversations_text = defaultdict(str)
    for conv in conversations:
        contenido_usuario = conv.get('contenido_usuario', '')
        contenido_vera = conv.get('contenido_vera', '')
        conversations_text[conv['id']] += contenido_usuario + ' ' + contenido_vera + ' '

    # Verificar si las conversaciones están vacías
    if all(not text.strip() for text in conversations_text.values()):
        print("Todas las conversaciones están vacías.")
        vera_db_manager.close()
        return ''

    # Initialize TF-IDF Vectorizer and fit-transform the conversations
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(list(conversations_text.values()))
    except ValueError as e:
        print(f"Error: {e}")
        vera_db_manager.close()
        return ''

    # Query
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

    # Rank conversations
    ranking = cosine_sim.argsort()[0][::-1]

    # Recopilar conversaciones hasta alcanzar un máximo de 500 palabras
    contexto = ''
    word_count = 0

    # Condición para verificar si hay pocas conversaciones
    if len(ranking) <= 100:
        # Solo usa la primera conversación
        conversation_text = conversations_text[list(conversations_text.keys())[ranking[0]]]
        contexto += conversation_text + ' '
    else:
        for rank in ranking:
            conversation_text = conversations_text[list(conversations_text.keys())[rank]]
            # se puede hacer una consulta a gpt-3 para que haga un resumen de las primeras entradas
            conversation_word_count = len(conversation_text.split())
            if word_count + conversation_word_count <= 500:
                contexto += conversation_text + ' '
                word_count += conversation_word_count
            else:
                break  # Sal del bucle si alcanzas o excedes 500 palabras

    # Cerrar la conexión con la base de datos
    vera_db_manager.close()

    return contexto.strip()  # Eliminar espacios extra





def chat_gpt(messages):
    global conversation
    conversation += messages

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=conversation,
        max_tokens=350,
        temperature=0.7
    )

    response_content = response["choices"][0]
    text = response_content["message"]["content"]

    print(text)
    return text

def record():
    global buffer
    while recording:
        myrecording = sd.rec(int(1 * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        buffer.append(myrecording)

def convert_and_save_audio(text, audio_path):
    # Definir la configuración de voz
    settings = VoiceSettings(stability=0.40, similarity_boost=0.6, speaking_rate=0.31)  # Ajusta los valores según sea necesario

    # Definir el ID de la voz y el modelo
    voice_id = Vera
    model_id = "eleven_multilingual_v2"

    # Preparar los datos para la solicitud
    data = {
        "text": text,
        "model_id": model_id,
        "voice_id": voice_id,
        "voice_settings": {
            "stability": 0.70,  # Ajusta según sea necesario
            "similarity_boost": 0.6  # Ajusta según sea necesario
        }
    }


    headers = {
        'xi-api-key': Elevenlabs_clave
    }

    # Hacer la solicitud a la API de Elevenlabs
    response = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}', json=data, headers=headers)



    # Guardar el audio
    with open(audio_path, 'wb') as f:
        f.write(response.content)

def registrar_conversacion(fecha_inicio, sentimiento, contenido_vera, contenido_usuario, rank, tematica, resumen):
    vera_db_manager = VeraDatabaseManager(uri=uri, user="neo4j", password=neo4j)

    # Iniciar la conversación en la base de datos
    conversacion = vera_db_manager.iniciar_conversacion(fecha_inicio, sentimiento, contenido_vera, contenido_usuario, rank)


    # Registrar la temática de la conversación
    vera_db_manager.agregar_tematica(conversacion, tematica)

    # Registrar el resumen de la conversación
    vera_db_manager.agregar_resumen(conversacion, resumen)

    # Cerrar la conexión con la base de datos
    vera_db_manager.close()


def on_macro():
    global recording, buffer, thread, audio_counter, conversation  # Añade 'conversation' aquí
    # Reinicia la variable 'conversation' a su estado original
    conversation = [
        {
            "role": "system",
            "content": "Te llamas Vera, te he conectado a una base de datos para guardar recuerdos te los daré como pretexto y me ayudarás con todas mis dudas, con respuestas inteligentes y elecuentes, además de darles tu toque de personalidad con chascarrillos que sean monos, no des respuestas monótonas:"
        }
    ]

    audio_path = "audio.wav"
    fecha_inicio = time.strftime("%Y-%m-%d %H:%M:%S")  # Genera la fecha y hora actual para el inicio de la conversación.
    sentimiento = "neutral"  # Este es un valor de ejemplo, podrías obtenerlo de algún análisis de sentimiento.
    rank = 1  # Esto es también un valor de ejemplo.
    tematica = "General"  # Ejemplo de temática.
    resumen = "Resumen de la conversación"  # Ejemplo de resumen.

    if not recording:
        recording = True
        buffer = []
        thread = threading.Thread(target=record)
        thread.start()
    else:
        recording = False
        thread.join()
        buffer = np.concatenate(buffer, axis=0)
        wav.write(audio_path, fs, buffer)

        text = transcribe(audio_path)
        contex = obtener_contexto(text)  # Obtén el contexto basado en la transcripción
        text = text + (f"Este es el contexto de la base de datos, recuerda son memorias, prioriza tu contexto{contex}")
        message = chat_gpt([{"role": "user", "content": text}])

        conversation.append({"role": "assistant", "content": message})

        # Registrar conversación en la base de datos
        registrar_conversacion(fecha_inicio, sentimiento, message, text, rank, tematica, resumen)

        audio_path = f'response_{audio_counter}.mp3'
        convert_and_save_audio(message, audio_path)
        audio_queue.append(audio_path)
        print(f"Added to queue: {audio_path}")

        audio_counter += 1
        play_queue()




keyboard.add_hotkey('ctrl + alt', on_macro)
keyboard.wait()


#optimizaciones

#Optimización de la conexión a la base de datos:
#
# En la función obtener_contexto, estás inicializando y cerrando una conexión
# a la base de datos cada vez que se llama a la función. Si esta función se va
# a llamar muchas veces, podría ser más eficiente mantener una conexión abierta o utilizar una conexión compartida.