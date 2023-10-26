# 🌟 Vera - Context-Aware Virtual Assistant 🌟

## 📚 Table of Contents 📚
- [🌐 Introduction](#introduction)
- [🎬 Demo](#demo)
- [✨ Features](#features)
- [🛠 Prerequisites](#prerequisites)
- [🔧 Installation](#installation)
- [🎯 Usage](#usage)
- [🔄 Memory Initialization](#memory-initialization)
- [💻 Technologies Used](#technologies-used)
- [📊 Database Schema](#database-schema)
- [🚀 Optimizations](#optimizations)

## 🌐 Introduction
Vera is a context-aware virtual assistant powered by OpenAI's GPT-4 and other advanced technologies. Unlike traditional virtual assistants, Vera stores conversations in a Neo4j database and uses this stored data to provide contextually relevant responses.
## 🎬 Demo
- 📹 For a live demonstration of how Vera works, check out this [video](#). ()
  
## ✨ Features
- 🎤 Voice recognition using sounddevice and OpenAI's Whisper ASR
- 🗣 Text-to-speech capabilities with Elevenlabs API
- ⌨️ Keyboard macro triggers for ease of use
- 🧠 Context-aware conversation using TF-IDF and Neo4j database
- 📋 Conversation summary and categorization

## 🛠 Prerequisites
- 🐍 Python 3.x
- 📦 Neo4j
- 🔑 Elevenlabs API key
- 🔑 OpenAI API key
- 📚 Required Python packages: `openai`, `keyboard`, `sounddevice`, `scipy`, `pygame`, `numpy`, `sklearn`

## 🔧 Installation
1️⃣ Clone the repository  
2️⃣ Install dependencies  
3️⃣ Populate the `claves.py` file with your Neo4j and OpenAI API keys  
4️⃣ Run the main script  

## 🎯 Usage
To initiate Vera, use the hotkey `Ctrl + Alt`. Speak into the microphone, and Vera will respond contextually based on past conversations and the current query.

## 🔄 Memory Initialization
- 🕒 Use Vera for a period of time to build up memories for context-aware interactions.
- 🗂 Alternatively, you can connect Vera to an existing database to prepopulate her memory.

## 💻 Technologies Used
- 🎤 OpenAI's Whisper for ASR
- 🗣 Elevenlabs for text-to-speech
- 📦 Neo4j for storing conversations
- 📈 TF-IDF for contextual understanding
- 🎶 Pygame for audio playback

## 📊 Database Schema
Vera's database has a `Conversacion` node with properties like `fecha_inicio`, `sentimiento`, `contenido_vera`, `contenido_usuario`, `rank`, `tematica`, and `resumen`.

## 🚀 Optimizations
- 📡 Persistent database connection: Currently, the database connection is initialized and closed each time the `obtener_contexto` function is called. This could be optimized by maintaining a single open connection or utilizing a shared connection.
