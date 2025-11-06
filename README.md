# CivicConnect: Accessible Civic Assistant with ASL & Voice

## Overview
**CivicConnect** is an accessibility-first civic assistant that makes government services inclusive for everyone, especially those who are deaf or hard of hearing. The platform bridges the communication gap by providing multi-modal responses in **text, speech, and American Sign Language (ASL)**, ensuring equal access to civic information regardless of communication mode or ability.

### What CivicConnect Does
- **Fetches & Summarizes** government documents (forms, policies, FAQs) using RAG and Gemini AI
- **Converts responses** into spoken explanations using ElevenLabs voice synthesis
- **Generates ASL animations** from gloss text using our custom 2D motion renderer
- **Displays all outputs** (text, speech, sign) side-by-side for inclusive accessibility

**Example Query:**  
*"How do I apply for unemployment benefits?"*  
→ CivicConnect finds the official document, summarizes it, explains it out loud, and signs it in ASL.

### Core Capabilities
- Extract **pose and hand keypoints** from ASL videos using MediaPipe
- Save extracted motions as **JSON files** for animation
- Animate ASL sequences into video for visualization
- Interact with a **LangChain conversational assistant** with RAG (retrieval-augmented generation) capabilities  

---

## Demo

<p align="center">
  <img src="./assets/demo.gif" width="700" alt="CivicConnect Demo">
  <br>
  <em>Simulation of user interaction with CivicConnect showing synchronized text, speech, and ASL output</em>
</p>

### 📺 Watch Full Demo
[![Watch Demo on YouTube](https://www.youtube.com/watch?v=d3TZcW6SjN0)](https://www.youtube.com/watch?v=d3TZcW6SjN0)

### 🏆 Devpost
[![View on Devpost](https://devpost.com/software/civicconnect-5gqacv)](https://devpost.com/software/civicconnect-5gqacv)

---

## Inspiration
Millions of citizens, especially those who are deaf or hard of hearing, struggle to access government services and documents because most digital interfaces rely only on text or speech. We created CivicConnect to be an inclusive AI assistant that makes every civic document speak, sign, and explain itself—giving equal access to all, regardless of communication mode or ability.

---

## Features
- **Multi-Modal Accessibility**: Provides synchronized text, speech, and ASL sign outputs for every response
- **Government Document RAG**: Retrieves and summarizes civic documents using ChromaDB vector search and Gemini AI
- **ASL Animation Engine**: Converts gloss sequences into animated 2D videos using pose and hand keypoints
- **Voice Synthesis**: Generates lifelike speech narration using ElevenLabs API
- **ASL Keypoint Extraction**: Extracts pose and hand keypoints from videos and saves them in JSON format
- **Conversational AI**: LangChain-based assistant with:
  - Context retrieval from ChromaDB (government documents)
  - Optional real-time web search via Tavily
  - Session-based chat history
- **FastAPI Backend**: Exposes endpoints for AI questions and RAG testing
- **Accessible UI**: Streamlit-based frontend designed for inclusive civic engagement  

---

## Project Structure

```
CivicConnect/
│
├── frontend/                    # Streamlit UI
│   └── app.py
│
├── api/                         # FastAPI backend
│   └── main.py
│
├── src/                         # Core LangChain modules
│   ├── LangChainWithChromaAndWeb.py
│   └── chroma_db/
│
├── motions/                     # Extracted ASL motion JSON files
│   └── *.json
│
├── asl/                      # WLASL dataset JSON
│   └── WLASL_v0.3.json
│
├── scripts/                     # Utility scripts
│   ├── asljson.py               # Extract keypoints from videos
│   └── animate_from_json.py     # Animate ASL sequences
│
├── models/                      # Optional ML or MediaPipe models
│   └── *.task
│
├── requirements.txt             # Python dependencies
├── README.md
└── .env                         # API keys and environment variables
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/CivicConnect.git
cd CivicConnect
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_gemini_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key  # optional
ELEVENLABS_API_KEY=your_elevenlabs_key
```

---

## Usage

### 1. Run the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

* Opens an accessible web interface for asking civic questions and receiving multi-modal responses (text, speech, ASL).

### 2. Extract ASL Keypoints

```bash
python scripts/asljson.py
```

* Downloads videos (non-YouTube URLs), extracts pose and hand keypoints, and saves them to `motions/`.

### 3. Animate ASL Sequence

```bash
python scripts/animate_from_json.py
```

* Creates a video of ASL glosses defined in the `SEQUENCE` list inside the script.

### 4. Run FastAPI LangChain Server

```bash
uvicorn api.main:app --reload
```

**Endpoints**:
- `POST /ask` → send question + session_id, get AI response.
- `GET /test-rag` → test RAG context retrieval.
- `GET /health` → check server and RAG status.

---

## Configuration

* **Sequence of glosses** for animation: Modify `SEQUENCE` in `animate_from_json.py`.
* **Motion storage**: `motions/` directory.
* **ChromaDB**: `src/chroma_db/` (automatically detected if exists).
* **LLM Model**: Google Gemini (via `ChatGoogleGenerativeAI`).

---

## Dependencies

* Python ≥ 3.10
* [Streamlit](https://streamlit.io/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [LangChain](https://www.langchain.com/)
* [Gemini API](https://ai.google.dev/)
* [ElevenLabs](https://elevenlabs.io/)
* [MediaPipe](https://mediapipe.dev/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)
* [HuggingFace Embeddings](https://huggingface.co/)
* [Chroma](https://www.trychroma.com/)
* [Tavily](https://www.tavily.com/) (optional)

---

## Challenges We Faced

* Finding reliable ASL gloss-to-motion data for generating signs dynamically
* Managing multi-modal output synchronization between text, voice, and animation
* Integrating Chroma vector search for government data RAG efficiently
* Designing an inclusive yet modern UI in a short time frame

---

## Accomplishments

* Built a multi-modal AI accessibility tool within 25 hours
* Seamlessly integrated LLM (Gemini) + speech & sign generation
* Designed a visually beautiful Streamlit prototype for inclusive design
* Created a working ASL animation engine that visualizes gloss sequences

---

## What We Learned

* How to leverage ChromaDB for RAG pipelines efficiently
* Designing for accessibility first changes the way you think about UX
* Converting text to ASL glosses and motion JSONs requires linguistic insight
* Multi-modal LLM applications can be powerful bridges for inclusion

---

## What's Next for CivicConnect

* Integrate real ASL 3D motion capture data for lifelike signing
* Add speech-to-sign translation for real-time interactions
* Expand coverage to more government agencies and languages
* Implement form auto-complete suggestions
* Deploy on Render for broader accessibility

---

## Built With

* **elevenlabs** - Voice synthesis for speech narration
* **fastapi** - Backend API framework
* **gemini** - Multi-modal AI and text generation
* **langchain** - Conversational AI and RAG pipeline
* **streamlit** - Accessible frontend interface
* **mediapipe** - ASL keypoint extraction
* **matplotlib** - ASL animation rendering
* **chroma** - Vector database for document retrieval

---

## License

MIT License

---

## Contact

* Authors: **Harish Singaravelan, Kirubhakaran Meenakshi Sundaram**
* Email: [hs7569@g.rit.edu](mailto:hs7569@g.rit.edu), [km1079@g.rit.edu](mailto:km1079@g.rit.edu)
* GitHub: [github.com/harishsingaravelan/](https://github.com/harishsingaravelan/), [github.com/kirubhakaranm/](https://github.com/kirubhakaranm/)