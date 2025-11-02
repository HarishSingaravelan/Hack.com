# CivicConnect - AI Government Assistant with Voice I/O + ASL
# Using Gemini API for speech-to-text and ElevenLabs for text-to-speech
# Built for MLH Best Use of ElevenLabs & Best Use of Gemini API

import streamlit as st
from datetime import datetime
import requests
import json
import os
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import base64
import tempfile
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="CivicConnect - AI Government Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Translations dictionary (keeping existing translations)
TRANSLATIONS = {
    "English": {
        "app_title": "CivicConnect",
        "app_subtitle": "AI-Powered Government Services Assistant",
        "app_powered": "🎤 Gemini Voice Input + 🔊 ElevenLabs Voice Output",
        "accessibility_settings": "🎛️ Accessibility Settings",
        "voice_input": "🎤 Voice Input (Gemini)",
        "enable_voice_input": "Enable Voice Input",
        "voice_input_help": "Record questions using microphone + Gemini API",
        "gemini_key_needed": "⚠️ Gemini API key needed",
        "gemini_ready": "✅ Gemini ready for voice input",
        "voice_output": "🔊 Voice Output (ElevenLabs)",
        "enable_voice_output": "Enable Voice Output",
        "voice_output_help": "Hear responses with ElevenLabs TTS",
        "select_voice": "Select Voice",
        "elevenlabs_ready": "✅ ElevenLabs ready",
        "elevenlabs_key_needed": "⚠️ ElevenLabs API key needed",
        "asl_avatar": "👤 ASL Video Output",
        "asl_help": "American Sign Language video interpretation",
        "language": "🌍 Language",
        "select_language": "Select Language",
        "session_info": "ℹ️ Session Info",
        "session_id": "ID",
        "active_area": "Area",
        "messages": "Messages",
        "new_conversation": "🔄 New Conversation",
        "back_to_menu": "⬅️ Back to Main Menu",
        "how_can_help": "📋 How can I help you today?",
        "select_focus": "Select a focus area to begin:",
        "taxes": "💰 **Taxes**",
        "taxes_desc": "*IRS forms, payments, deductions*",
        "housing": "🏠 **Housing Aid**",
        "housing_desc": "*Section 8, PHA, rental assistance*",
        "dmv": "🚗 **DMV/License**",
        "dmv_desc": "*REAL ID, vehicle titling, licenses*",
        "grounded_message": "🛡️ Grounded in official documents • 🎤 Gemini-powered voice input • 🔊 ElevenLabs natural voice output",
        "welcome_taxes": "Welcome to **Taxes** assistance! How can I help?",
        "welcome_housing": "Welcome to **Housing Aid** assistance! How can I help?",
        "welcome_dmv": "Welcome to **DMV/License** assistance! How can I help?",
        "voice_input_section": "🎤 Voice Input (Gemini-powered)",
        "click_to_record": "🎙️ Click the microphone to record your question",
        "clear": "🗑️ Clear",
        "transcribing": "🎧 Transcribing with Gemini...",
        "transcribed": "📝 Transcribed",
        "transcription": "Transcription",
        "use_transcription": "✅ Use this transcription",
        "generating_response": "🤔 Generating response...",
        "voice_ready": "🔊 ElevenLabs voice ready!",
        "text_input": "💬 Text Input",
        "type_question": "Type your question:",
        "ask_about": "Ask about",
        "send": "📤 Send",
        "clear_chat": "🗑️ Clear",
        "chat_cleared": "Chat cleared. How can I help with",
        "verified_sources": "🛡️ Verified by",
        "sources": "Sources",
        "view_sources": "📚 View Sources",
        "footer": "🏛️ CivicConnect | 🎤 Powered by Gemini API | 🔊 Powered by ElevenLabs | Built for accessible government services",
        "generating_asl": "🤟 Generating ASL video...",
        "asl_video_ready": "✅ ASL video ready",
    },
    "Spanish": {
        # ... (keeping all Spanish translations)
        "asl_avatar": "👤 Salida de Video ASL",
        "asl_help": "Interpretación en video de Lengua de Señas Americana",
        "generating_asl": "🤟 Generando video ASL...",
        "asl_video_ready": "✅ Video ASL listo",
    },
    "French": {
        # ... (keeping all French translations)
        "asl_avatar": "👤 Sortie Vidéo ASL",
        "asl_help": "Interprétation vidéo en Langue des Signes Américaine",
        "generating_asl": "🤟 Génération de vidéo ASL...",
        "asl_video_ready": "✅ Vidéo ASL prête",
    },
}

# API Configuration
API_URL = os.getenv("RAG_API_ENDPOINT", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
GEMINI_API_KEY_FRONTEND = os.getenv("GEMINI_API_KEY_FRONTEND", "")
# MOTION_DIR = "motions"  # Directory containing ASL motion JSON files
MOTION_DIR = os.path.join("Hack.com", "motions")
os.makedirs(MOTION_DIR, exist_ok=True)

# Initialize ElevenLabs client
elevenlabs_client = None
if ELEVENLABS_API_KEY:
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        st.sidebar.warning(f"ElevenLabs initialization failed: {e}")

# Initialize Gemini API
if GEMINI_API_KEY_FRONTEND:
    try:
        genai.configure(api_key=GEMINI_API_KEY_FRONTEND)
    except Exception as e:
        st.sidebar.warning(f"Gemini initialization failed: {e}")


# Custom CSS (add to existing st.markdown CSS section)
# st.markdown(
#     """
# <style>
#     .main-header {
#         text-align: center;
#         padding: 2rem 0;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border-radius: 10px;
#         margin-bottom: 2rem;
#     }
#     .voice-input-section {
#         background: #f0f9ff;
#         padding: 1rem;
#         border-radius: 8px;
#         border: 2px solid #3b82f6;
#         margin-bottom: 1rem;
#     }
#     /* NEW: ASL Video Container */
#     .asl-video-container {
#         display: flex;
#         justify-content: center;
#         margin: 15px 0;
#     }
#     .asl-video-container video {
#         width: 140px !important;
#         height: 140px !important;
#         max-width: 140px !important;
#         max-height: 140px !important;
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
#         border: 2px solid #667eea;
#         object-fit: contain;
#         background: #f0f9ff;
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )

st.markdown(
    """
<style>
    .asl-video-container {
        display: flex;
        justify-content: center;
        margin: 15px 0;
    }
    .asl-video-container video {
        width: 200px !important;
        height: 200px !important;
        max-width: 200px !important;
        max-height: 200px !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        border: 2px solid #667eea;
        object-fit: contain;
        background: #f0f9ff;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
if "current_goal" not in st.session_state:
    st.session_state.current_goal = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S')}"
if "asl_active" not in st.session_state:
    st.session_state.asl_active = False
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False
if "voice_input_active" not in st.session_state:
    st.session_state.voice_input_active = False
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "Rachel"
if "last_transcription" not in st.session_state:
    st.session_state.last_transcription = ""
if "language" not in st.session_state:
    st.session_state.language = "English"
if "current_asl_video" not in st.session_state:
    st.session_state.current_asl_video = None


def t(key):
    """Get translation for current language"""
    return TRANSLATIONS.get(st.session_state.language, TRANSLATIONS["English"]).get(
        key, key
    )


# ElevenLabs voices
AVAILABLE_VOICES = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Drew": "29vD33N1CtxCmqQRPOHJ",
    "Clyde": "2EiwWnXFnvU5JabPnv8n",
    "Paul": "5Q0t7uMcjvnagumLfvZi",
    "Domi": "AZnzlk1XvdvUeBnXmlld",
    "Dave": "CYw3kZ02Hs0563khs1Fj",
    "Fin": "D38z5RcWu1voky8WS1ja",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
}


# ========== ASL VIDEO GENERATION ==========

# Load available ASL words at startup
AVAILABLE_ASL_WORDS = set()
try:
    with open("gloss_list.json", "r") as f:
        word_dict = json.load(f)
    # Extract keys from the dictionary
    AVAILABLE_ASL_WORDS = set(word.lower() for word in word_dict.keys())
    # AVAILABLE_ASL_WORDS = set(word.lower() for word in json.load(f))
    # AVAILABLE_ASL_WORDS = {"more", "money", "more", "rich"}
    print(f"✅ Loaded {len(AVAILABLE_ASL_WORDS)} available ASL signs")
except Exception as e:
    print(f"⚠️ Could not load available_words.json: {e}")


def find_semantic_match(word, available_words):
    """
    Use Gemini to find semantic match for a word from available ASL signs
    Returns the best matching word or None
    """
    try:
        if not GEMINI_API_KEY_FRONTEND or not available_words:
            return None

        # Direct match first (case-insensitive)
        word_lower = word.lower()
        if word_lower in available_words:
            return word_lower

        model = genai.GenerativeModel("gemini-2.5-flash")

        # Create a sample of available words (max 100 for faster processing)
        sample_words = (
            list(available_words)[:100]
            if len(available_words) > 100
            else list(available_words)
        )

        prompt = f"""Find the best semantic match for the word "{word}" from this list of available ASL signs.

Available signs: {', '.join(sample_words)}

Rules:
1. Return ONLY the matching word from the list, nothing else
2. Choose the closest meaning/synonym
3. If no good match exists, return "NONE"
4. Return lowercase

Word to match: {word}
Best match:"""

        response = model.generate_content(prompt)
        match = response.text.strip().lower()

        # Validate the match is in our available words
        if match != "none" and match in available_words:
            print(f"✅ Semantic match: {word} → {match}")
            return match
        else:
            print(f"⚠️ No semantic match found for: {word}")
            return None

    except Exception as e:
        print(f"Semantic matching error for '{word}': {e}")
        return None


def summarize_to_asl_gloss(text):
    """
    Convert text response to ASL gloss (simplified sign sequence)
    Maximum 10-12 words for ~10 second video
    """
    try:
        if not GEMINI_API_KEY_FRONTEND:
            return []

        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""Convert the following text to ASL gloss format (capitalized keywords only).
        
Rules:
1. Use ONLY common nouns, verbs, and key concepts
2. Maximum 10-12 words
3. Remove articles (a, an, the), prepositions, and connecting words
4. Use present tense
5. Use simple, common words that likely have ASL signs
6. Return ONLY the gloss words separated by spaces, nothing else

Text: {text}

ASL Gloss:"""

        response = model.generate_content(prompt)
        gloss_text = response.text.strip().upper()

        # Extract only alphanumeric words
        words = re.findall(r"\b[A-Z]+\b", gloss_text)

        # Limit to 12 words max for ~10 second video
        return words[:12]

    except Exception as e:
        print(f"ASL gloss generation error: {e}")
        # Fallback: simple keyword extraction
        words = text.upper().split()
        keywords = [w.strip(".,!?;:") for w in words if len(w) > 3]
        return keywords[:10]


def load_motion(gloss):
    """Load motion data for a specific gloss with semantic matching"""
    gloss_lower = gloss.lower()

    # Try direct lookup first
    direct_path = os.path.join(MOTION_DIR, f"{gloss_lower}.json")
    if os.path.exists(direct_path):
        try:
            with open(direct_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {gloss}: {e}")
            return []

    # Try semantic matching if direct lookup fails
    print(f"🔍 Trying semantic match for: {gloss}")
    semantic_match = find_semantic_match(gloss, AVAILABLE_ASL_WORDS)

    if semantic_match:
        semantic_path = os.path.join(MOTION_DIR, f"{semantic_match}.json")
        if os.path.exists(semantic_path):
            try:
                with open(semantic_path, "r") as f:
                    print(f"✅ Using semantic match: {gloss} → {semantic_match}")
                    return json.load(f)
            except Exception as e:
                print(f"Error loading semantic match {semantic_match}: {e}")
                return []

    print(f"⚠️ No motion found for: {gloss} (skipping)")
    return []


def draw_frame(ax, keypoints):
    """Draw a single frame of ASL animation"""
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")
    ax.set_facecolor("#f0f9ff")

    # Draw pose (body)
    if "pose" in keypoints:
        pose_points = np.array(keypoints["pose"])
        ax.scatter(
            pose_points[:, 0], pose_points[:, 1], color="#667eea", s=20, alpha=0.8
        )

    # Draw hands
    if "hands" in keypoints:
        for i, hand in enumerate(keypoints["hands"]):
            hand_points = np.array(hand)
            color = "#10b981" if i == 0 else "#f59e0b"
            ax.scatter(
                hand_points[:, 0], hand_points[:, 1], color=color, s=15, alpha=0.9
            )
            ax.plot(
                hand_points[:, 0],
                hand_points[:, 1],
                color=color,
                linewidth=1.5,
                alpha=0.7,
            )


def generate_asl_video(gloss_sequence, caption=None, is_fallback=False):
    """Generate ASL video from gloss sequence with optional caption"""
    if not gloss_sequence:
        return None

    # Load all available motions
    all_frames = []
    used_glosses = []

    for gloss in gloss_sequence:
        frames = load_motion(gloss)
        if frames:
            all_frames.extend(frames)
            used_glosses.append(gloss)

    # Check if we have enough signs (minimum 3 for meaningful content)
    if not is_fallback and len(used_glosses) < 3:
        print("⚠️ Not enough valid motions, using fallback message")
        # Use fallback: "THANK YOU UNDERSTAND"
        return generate_asl_video(
            ["THANK", "YOU", "UNDERSTAND"],
            caption="ASL video not available. Coming soon!",
            is_fallback=True,
        )

    if not all_frames:
        print("⚠️ No valid motions found")
        return None

    print(f"🤟 Generating ASL video with: {' → '.join(used_glosses)}")

    # ADJUST SIZE HERE: Reduced from (4, 4) to (3, 3) for smaller video
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    fig.patch.set_facecolor("#f0f9ff")

    # Add space for caption at bottom if provided
    if caption:
        plt.subplots_adjust(bottom=0.15)  # Make room for caption

    def update(frame_idx):
        if frame_idx < len(all_frames):
            draw_frame(ax, all_frames[frame_idx])

            # Add caption text to each frame
            if caption:
                fig.text(
                    0.5,
                    0.05,  # Position: centered horizontally, near bottom
                    caption,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#1f2937",
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        edgecolor="#667eea",
                        linewidth=2,
                    ),
                )

    ani = animation.FuncAnimation(
        fig, update, frames=len(all_frames), interval=40, repeat=False
    )

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_path = temp_file.name
    temp_file.close()

    try:
        # ADJUST BITRATE/QUALITY HERE if needed
        writer = animation.FFMpegWriter(
            fps=25,
            bitrate=1200,  # Reduced from 1800 for smaller file size
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],  # Better compatibility
        )
        ani.save(video_path, writer=writer)
        plt.close(fig)

        # Read video file
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        # Clean up
        os.unlink(video_path)

        return video_bytes

    except Exception as e:
        print(f"Error saving ASL video: {e}")
        plt.close(fig)
        if os.path.exists(video_path):
            os.unlink(video_path)
        return None


# ========== EXISTING FUNCTIONS ==========
def transcribe_audio_with_gemini(audio_bytes):
    """Transcribe audio using Gemini API with speech recognition"""
    try:
        if not GEMINI_API_KEY_FRONTEND:
            st.error("⚠️ Gemini API key not configured")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        audio_file = genai.upload_file(path=temp_audio_path)
        model = genai.GenerativeModel("gemini-2.5-flash")
        language_name = st.session_state.language
        prompt = f"Transcribe this audio exactly as spoken in {language_name}. Only return the transcribed text, nothing else."
        response = model.generate_content([prompt, audio_file])

        os.unlink(temp_audio_path)
        genai.delete_file(audio_file.name)

        if response.text:
            return response.text.strip()
        else:
            st.error("No transcription received from Gemini")
            return None

    except Exception as e:
        st.error(f"Gemini transcription error: {e}")
        if "temp_audio_path" in locals():
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        return None


def generate_speech(text, voice_id):
    """Generate speech using ElevenLabs API"""
    if not elevenlabs_client:
        return None

    try:
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        audio_bytes = b"".join(audio_generator)
        return audio_bytes

    except Exception as e:
        st.error(f"ElevenLabs speech generation failed: {e}")
        return None


def autoplay_audio(audio_bytes):
    """Create an HTML audio player that autoplays"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay style="width: 100%;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)


def call_civic_connect_api(question, session_id, language):
    """Call the CivicConnect API with the user's question"""
    try:
        payload = {
            "session_id": session_id,
            "question": question,
            "language": language,
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return {"success": True, "data": response.json()}

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to the server."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"Server error: {e.response.status_code}"}
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred: {str(e)}"}


def generate_ai_response(user_query, goal):
    """Generate AI response using the backend API"""
    language = st.session_state.language
    api_response = call_civic_connect_api(
        user_query, st.session_state.session_id, language
    )

    if not api_response["success"]:
        return {
            "text": f"⚠️ **Error:** {api_response['error']}\n\nPlease try again.",
            "sources": [],
            "asl_gloss": [],
            "asl_message": None,
            "error": True,
            "audio": None,
            "asl_video": None,
        }

    data = api_response["data"]
    answer = data.get("answer", data.get("response", "No response received"))
    sources = data.get("sources", data.get("references", []))

    formatted_sources = []
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict):
                formatted_sources.append(
                    {
                        "title": source.get(
                            "title", source.get("name", "Untitled Source")
                        ),
                        "uri": source.get(
                            "uri", source.get("url", source.get("link", "#"))
                        ),
                    }
                )
            elif isinstance(source, str):
                formatted_sources.append({"title": source, "uri": "#"})

    # Generate ASL gloss and video if enabled
    asl_gloss = []
    asl_video = None
    asl_message = None

    if st.session_state.asl_active:
        asl_gloss = summarize_to_asl_gloss(answer)

        if asl_gloss and os.path.exists(MOTION_DIR) and AVAILABLE_ASL_WORDS:
            # Count how many glosses have available motions
            matched_glosses = sum(
                1
                for g in asl_gloss
                if g.lower() in AVAILABLE_ASL_WORDS
                or find_semantic_match(g, AVAILABLE_ASL_WORDS)
            )

            if matched_glosses < 3:
                # Not enough matches - use fallback with caption
                asl_message = (
                    "⚠️ ASL video not available for this response. Coming soon!"
                )
                asl_video = generate_asl_video(
                    ["THANK", "YOU", "UNDERSTAND"],
                    caption="Thank you for understanding",
                    is_fallback=True,
                )
            else:
                # Generate normal video without caption (or with custom caption)
                asl_video = generate_asl_video(asl_gloss, caption=None)

    # Generate speech with ElevenLabs
    audio_bytes = None
    if st.session_state.voice_active and elevenlabs_client:
        voice_id = AVAILABLE_VOICES.get(
            st.session_state.selected_voice, AVAILABLE_VOICES["Rachel"]
        )
        audio_bytes = generate_speech(answer, voice_id)

    return {
        "text": answer,
        "sources": formatted_sources,
        "asl_gloss": asl_gloss,
        "asl_message": asl_message,
        "error": False,
        "audio": audio_bytes,
        "asl_video": asl_video,
    }


# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown(f"### {t('accessibility_settings')}")

    st.markdown(f"#### {t('voice_input')}")
    st.session_state.voice_input_active = st.checkbox(
        t("enable_voice_input"),
        value=st.session_state.voice_input_active,
        help=t("voice_input_help"),
    )

    if st.session_state.voice_input_active:
        if not GEMINI_API_KEY_FRONTEND:
            st.warning(t("gemini_key_needed"))
        else:
            st.success(t("gemini_ready"))

    st.markdown("---")

    st.markdown(f"#### {t('voice_output')}")
    st.session_state.voice_active = st.checkbox(
        t("enable_voice_output"),
        value=st.session_state.voice_active,
        help=t("voice_output_help"),
    )

    if st.session_state.voice_active:
        if elevenlabs_client:
            st.session_state.selected_voice = st.selectbox(
                t("select_voice"),
                options=list(AVAILABLE_VOICES.keys()),
                index=list(AVAILABLE_VOICES.keys()).index(
                    st.session_state.selected_voice
                ),
            )
            st.success(t("elevenlabs_ready"))
        else:
            st.warning(t("elevenlabs_key_needed"))

    st.markdown("---")

    st.session_state.asl_active = st.checkbox(
        t("asl_avatar"),
        value=st.session_state.asl_active,
        help=t("asl_help"),
    )

    if st.session_state.asl_active and not os.path.exists(MOTION_DIR):
        st.warning(f"⚠️ ASL motion folder '{MOTION_DIR}' not found")

    st.markdown("---")
    st.markdown(f"### {t('language')}")

    new_language = st.selectbox(
        t("select_language"),
        ["English", "Spanish", "French"],
        index=["English", "Spanish", "French"].index(st.session_state.language),
    )

    if new_language != st.session_state.language:
        st.session_state.language = new_language
        st.rerun()

    st.markdown("---")
    st.markdown(f"### {t('session_info')}")
    st.info(f"**{t('session_id')}:** {st.session_state.session_id[:20]}...")
    if st.session_state.current_goal:
        st.success(f"**{t('active_area')}:** {st.session_state.current_goal}")
    st.markdown(f"**{t('messages')}:** {len(st.session_state.chat_history)}")

    st.markdown("---")

    if st.session_state.current_goal:
        if st.button(t("back_to_menu"), use_container_width=True, type="primary"):
            st.session_state.current_goal = None
            st.session_state.chat_history = []
            st.session_state.last_transcription = ""
            st.session_state.current_asl_video = None
            st.rerun()

    if st.button(t("new_conversation"), use_container_width=True):
        st.session_state.current_goal = None
        st.session_state.chat_history = []
        st.session_state.session_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.last_transcription = ""
        st.session_state.current_asl_video = None
        st.rerun()


# ========== MAIN APPLICATION ==========
if st.session_state.current_goal is None:
    st.markdown(
        f"""
    <div class="main-header">
        <h1>🏛️ {t('app_title')}</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">{t('app_subtitle')}</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">{t('app_powered')}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(f"### {t('how_can_help')}")
    st.markdown(t("select_focus"))

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(
            f"{t('taxes')}\n\n{t('taxes_desc')}", key="taxes", use_container_width=True
        ):
            st.session_state.current_goal = "Taxes"
            st.session_state.chat_history = [
                {
                    "role": "ai",
                    "text": t("welcome_taxes"),
                    "sources": [],
                    "error": False,
                    "audio": None,
                    "asl_video": None,
                    "asl_gloss": [],
                }
            ]
            st.rerun()

    with col2:
        if st.button(
            f"{t('housing')}\n\n{t('housing_desc')}",
            key="housing",
            use_container_width=True,
        ):
            st.session_state.current_goal = "Housing Aid"
            st.session_state.chat_history = [
                {
                    "role": "ai",
                    "text": t("welcome_housing"),
                    "sources": [],
                    "error": False,
                    "audio": None,
                    "asl_video": None,
                    "asl_gloss": [],
                }
            ]
            st.rerun()

    with col3:
        if st.button(
            f"{t('dmv')}\n\n{t('dmv_desc')}", key="dmv", use_container_width=True
        ):
            st.session_state.current_goal = "DMV/License"
            st.session_state.chat_history = [
                {
                    "role": "ai",
                    "text": t("welcome_dmv"),
                    "sources": [],
                    "error": False,
                    "audio": None,
                    "asl_video": None,
                    "asl_gloss": [],
                }
            ]
            st.rerun()

    st.markdown("---")
    st.markdown(
        f"""
    <div style="background: #ecfdf5; padding: 1.5rem; border-radius: 10px; border: 2px solid #10b981; text-align: center;">
        <span style="color: #059669; font-weight: 600;">{t('grounded_message')}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    # Chat Interface
    st.markdown(
        f"""
    <div class="main-header">
        <h2>🏛️ {t('app_title')} - {st.session_state.current_goal}</h2>
        <p>{t('app_powered')}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(message["text"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    if message.get("error", False):
                        st.error(message["text"])
                    else:
                        st.markdown(message["text"])

                    if message.get("audio") and st.session_state.voice_active:
                        autoplay_audio(message["audio"])

                    if message.get("sources") and len(message["sources"]) > 0:
                        st.success(
                            f"{t('verified_sources')} {len(message['sources'])} {t('sources')}"
                        )
                        with st.expander(t("view_sources")):
                            for i, source in enumerate(message["sources"], 1):
                                st.markdown(
                                    f"**{i}. [{source['title']}]({source['uri']})**"
                                )

                    # Display ASL video if available
                    # if message.get("asl_video") and st.session_state.asl_active:
                    #     st.success(f"🤟 {t('asl_video_ready')}")
                    #     st.video(message["asl_video"])
                    #     if message.get("asl_gloss"):
                    #         with st.expander("📝 ASL Gloss"):
                    #             st.code(" → ".join(message["asl_gloss"]))
                    if message.get("asl_video") and st.session_state.asl_active:
                        st.success(f"🤟 {t('asl_video_ready')}")

                        # Autoplay, loop, muted video without controls
                        video_bytes = message["asl_video"]
                        b64_video = base64.b64encode(video_bytes).decode()
                        video_html = f"""
                            <div class="asl-video-container">
                                <video autoplay loop muted playsinline style="width: 300px; height: 300px;">
                                    <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                                </video>
                            </div>
                        """
                        st.markdown(video_html, unsafe_allow_html=True)

                        if message.get("asl_gloss"):
                            with st.expander("📝 ASL Gloss"):
                                st.code(" → ".join(message["asl_gloss"]))

    # Voice Input Section
    st.markdown("---")

    if st.session_state.voice_input_active:
        st.markdown('<div class="voice-input-section">', unsafe_allow_html=True)
        st.markdown(f"#### {t('voice_input_section')}")

        col_v1, col_v2 = st.columns([3, 1])
        with col_v1:
            st.info(t("click_to_record"))
        with col_v2:
            if st.session_state.last_transcription:
                if st.button(t("clear"), key="clear_voice"):
                    st.session_state.last_transcription = ""
                    st.rerun()

        audio_bytes = audio_recorder(
            text="",
            recording_color="#ef4444",
            neutral_color="#3b82f6",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
        )

        if audio_bytes and audio_bytes != st.session_state.get("last_audio_bytes"):
            st.session_state.last_audio_bytes = audio_bytes

            with st.spinner(t("transcribing")):
                transcribed_text = transcribe_audio_with_gemini(audio_bytes)

                if transcribed_text:
                    st.session_state.last_transcription = transcribed_text
                    st.success(f"{t('transcribed')}: {transcribed_text}")
                    st.rerun()

        if st.session_state.last_transcription:
            st.markdown(
                f"**{t('transcription')}:** {st.session_state.last_transcription}"
            )
            if st.button(t("use_transcription"), key="use_transcription"):
                user_input = st.session_state.last_transcription
                st.session_state.last_transcription = ""

                st.session_state.chat_history.append(
                    {"role": "user", "text": user_input, "sources": []}
                )

                with st.spinner(t("generating_response")):
                    if st.session_state.asl_active:
                        st.info(t("generating_asl"))

                    response = generate_ai_response(
                        user_input, st.session_state.current_goal
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "ai",
                            "text": response["text"],
                            "sources": response["sources"],
                            "asl_gloss": response.get("asl_gloss", []),
                            "error": response.get("error", False),
                            "audio": response.get("audio"),
                            "asl_video": response.get("asl_video"),
                        }
                    )

                    if st.session_state.voice_active and response.get("audio"):
                        st.toast(t("voice_ready"))

                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    # Text Input
    st.markdown(f"#### {t('text_input')}")
    user_input = st.text_input(
        t("type_question"),
        placeholder=f"{t('ask_about')} {st.session_state.current_goal}...",
        key="user_input",
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col2:
        send_button = st.button(t("send"), type="primary", use_container_width=True)
    with col3:
        clear_button = st.button(t("clear_chat"), use_container_width=True)

    if send_button and user_input:
        st.session_state.chat_history.append(
            {"role": "user", "text": user_input, "sources": []}
        )

        with st.spinner(t("generating_response")):
            if st.session_state.asl_active:
                st.info(t("generating_asl"))

            response = generate_ai_response(user_input, st.session_state.current_goal)
            st.session_state.chat_history.append(
                {
                    "role": "ai",
                    "text": response["text"],
                    "sources": response["sources"],
                    "asl_gloss": response.get("asl_gloss", []),
                    "error": response.get("error", False),
                    "audio": response.get("audio"),
                    "asl_video": response.get("asl_video"),
                }
            )

            if st.session_state.voice_active and response.get("audio"):
                st.toast(t("voice_ready"))

        st.rerun()

    if clear_button:
        st.session_state.chat_history = [
            {
                "role": "ai",
                "text": f"{t('chat_cleared')} {st.session_state.current_goal}?",
                "sources": [],
                "error": False,
                "audio": None,
                "asl_video": None,
                "asl_gloss": [],
            }
        ]
        st.session_state.last_transcription = ""
        st.session_state.current_asl_video = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <small>{t('footer')}</small>
</div>
""",
    unsafe_allow_html=True,
)
