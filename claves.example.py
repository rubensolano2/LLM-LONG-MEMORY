"""Copy to claves.py. Supply credentials through environment variables."""
import os

uri = os.environ['NEO4J_URI']
neo4j = os.environ['NEO4J_PASSWORD']
Openai_clave = os.environ['OPENAI_API_KEY']
Elevenlabs_clave = os.environ['ELEVENLABS_API_KEY']
Vera = os.environ['ELEVENLABS_VOICE_ID']
