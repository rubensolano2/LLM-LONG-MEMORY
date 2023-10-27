## 📚 Table of Contents 📚
- [🌐 Introduction](#-introduction)
- [🐞 Problems to Solve](#-problems-to-solve)
- [🎬 Demo](#-demo)
- [🚀 Anticipated Updates for the Project](#-anticipated-updates-for-the-project)
- [✨ Features](#-features)
- [🛠 Prerequisites](#-prerequisites)
- [🔧 Installation](#-installation)
- [🎯 Usage](#-usage)
- [🔄 Memory Initialization](#-memory-initialization)
- [💻 Technologies Used](#-technologies-used)
- [📊 Database Schema](#-database-schema)
- [🚀 Optimizations](#-optimizations)


## 🌐 Introduction- project under construction
Vera is a context-aware virtual assistant powered by OpenAI's GPT-4 and other advanced technologies. Unlike traditional virtual assistants, Vera stores conversations in a Neo4j database and uses this stored data to provide contextually relevant responses.

## 🐞  Beta Version of Conversational Assistant Software

1. Voice Interaction:

The software is capable of recording audio inputs, transcribing them to text using OpenAI, and generating voice responses through Elevenlabs API.
Utilizes an ultra-realistic synthetic voice for responses, enhancing the user interaction experience.
A queue system is employed for managing audio playback, ensuring a smooth user interaction.

2. Database Management:

Vera utilizes Neo4j as a database to chronicle and retrieve conversation data.
Supports the initiation, storage, and retrieval of conversations, which is pivotal for maintaining a coherent and context-aware interaction over time.

3. Context Awareness:

Aims for context-awareness by preserving past conversations and utilizing this data to shape future responses.
The current beta adopts a basic approach to context retrieval; further development is targeted to augment this facet for more precise context-aware responses.

4. Hotkey Activation:

Incorporates a hotkey activation mechanism for triggering the conversation, allowing user engagement at any given moment.

5. Advanced Language Models:

Utilizes state-of-the-art language models, leveraging powerful LLM (Language Model) technologies available in the market to ensure high-quality text generation and understanding.




## 🎬 Demo
- 📹 For a live demonstration of how Vera works, check out.

- DEMO 1: In this demo, the model was informed about the improvements that were going to be made in the future, and this was its response:
[![Reproducir Audio](https://img.shields.io/badge/Reproducir-Audio-blue)](https://drive.google.com/file/d/1iYRN9jrnDqcnpMa_fw-wv3eIrUS_jwHK/view?usp=sharing)

- Demo 2: In this demo, the model was disconnected, restarted, and then asked about the improvements from a previous conversation:
[![Reproducir Audio](https://img.shields.io/badge/Reproducir-Audio-blue)](https://drive.google.com/file/d/1iZzSjNRlOgSlf5WqzVPeFRJczSt3-Amx/view?usp=sharing)


---
# 🚀 Anticipated Updates for the Project

Greetings, it's a pleasure to share the innovative improvements on the horizon for my project. These updates are meticulously designed to optimize the code and strengthen interaction. They are detailed as follows:

## 1. 🕰️ Incorporation of Temporal Awareness in Conversations
There are plans to integrate a temporal awareness functionality, allowing for more precise contextualization in responses. For instance, when inquiring "What time is it?" at 2 p.m., the answer will be "It's 2 p.m." instead of "I don't know." This improvement is a significant step towards more intuitive and coherent interactions.

## 2. 📄 Ability to Create Summaries
The ability to introduce sibling nodes in the database will be included to add a summary that can optimize data search.

## 3. 💡 Generation of New Ideas from Summaries
With this feature, new ideas can be conceived from the summaries of our conversations. This function acts as a breeding ground for inspiration and the exploration of new possibilities based on previous discussions, introducing relevant ideas as new information in the database. This process will be carried out by GPT-4 in parallel to filter out irrelevant ideas.

## 4. 🗺️ Thematization of Conversations
The functionality of thematizing conversations will be introduced, allowing to maintain a clear focus and follow a coherent thematic line throughout the interactions. This improvement will help keep our discussions aligned and create large areas of knowledge within the database itself. The idea is that nodes belonging to the same theme can merge and link together, being prioritized in the information search.

These refined improvements represent exciting steps towards expanding the project's capabilities. Thank you for contributing to the evolution of this project! 😄

---

  
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
1. Clone the repository  
2. Install dependencies  
3. Create in your directory the `claves.py` file with your Neo4j, ElevenLabs and OpenAI API keys 
4. Run the main script  

## 🎯 Usage
To initiate Vera run the code wait like 6 seconds and use the hotkey `Ctrl + Alt`. Speak into the microphone, and Vera will respond contextually based on past conversations and the current query.
You will be able to see in the terminal some prints with the processing of your audio, nodes, and responses.

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
