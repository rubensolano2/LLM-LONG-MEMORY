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


## 🌐 Introduction
Vera is a context-aware virtual assistant powered by OpenAI's GPT-4 and other advanced technologies. Unlike traditional virtual assistants, Vera stores conversations in a Neo4j database and uses this stored data to provide contextually relevant responses.

## 🐞 Problems to Solve

### Excessive Information Consumption

Currently, the code faces a challenge concerning information consumption. An excessive amount of information is being gathered during execution, which may be affecting the efficiency and performance of the program. The primary task is to review and optimize the use of variables and context handling throughout the code to identify and address context accumulation.

#### Steps for resolution:

1. **Variable Review**: Conduct a thorough review of the variables used in the code to understand their role and how they are contributing to information consumption.
   
2. **Context Handling Analysis**: Analyze how context is being handled throughout the code, and how this may be contributing to excessive information consumption.

3. **Optimization**: Implement necessary optimizations to reduce information consumption and improve code efficiency.

4. **Testing and Validation**: Perform tests to validate that the implemented optimizations have resolved the issue without introducing new problems.

Any contribution to help resolve this issue is very welcome. If you have any suggestions or solutions, feel free to open a [new Issue](https://github.com/rubensolano2/LLM-LONG-MEMORY/issues) or submit a [Pull Request](https://github.com/rubensolano2/LLM-LONG-MEMORY/pulls).



## 🎬 Demo
- 📹 For a live demonstration of how Vera works, check out.






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
1️⃣ Clone the repository  
2️⃣ Install dependencies  
3️⃣ Populate the `claves.py` file with your Neo4j, ElevenLabs and OpenAI API keys  
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
