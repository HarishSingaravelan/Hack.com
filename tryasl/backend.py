import os
import shutil
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from google import genai
from google.genai.errors import APIError
import traceback

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found.")

# --- Initialization ---
app = FastAPI(title="ASL Video Analyzer")

# Initialize Gemini Client with API key
client = genai.Client(api_key=API_KEY)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---

def save_upload_file_sync(upload_file: UploadFile) -> str:
    """Saves the uploaded file to a temporary location synchronously."""
    temp_filepath = f"temp_{upload_file.filename}"
    try:
        with open(temp_filepath, "wb") as buffer:
            buffer.write(upload_file.file.read())
    finally:
        upload_file.file.close()
    return temp_filepath

def cleanup_sync(temp_path: str, uploaded_file_name: str):
    """Deletes the local temp file and the file from Gemini's server synchronously."""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"✅ Deleted local temp file: {temp_path}")
        except Exception as e:
            print(f"⚠️ Error deleting temp file {temp_path}: {e}")
    
    if uploaded_file_name:
        try:
            client.files.delete(name=uploaded_file_name)
            print(f"✅ Cleaned up Gemini file: {uploaded_file_name}")
        except Exception as e:
            print(f"⚠️ Error during Gemini file cleanup for {uploaded_file_name}: {e}")

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ASL Video Analyzer is running", "gemini_configured": bool(API_KEY)}

@app.post("/analyze-asl/")
async def analyze_asl_video(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Receives a video file, uploads it to the Gemini Files API, 
    and requests an ASL-to-Text transcription.
    """
    if not video_file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {video_file.content_type}. Only video files are supported."
        )

    temp_path = ""
    uploaded_file = None
    
    try:
        # 1. Save video file locally
        print(f"📁 Saving uploaded file: {video_file.filename}")
        temp_path = await run_in_threadpool(save_upload_file_sync, video_file)
        print(f"✅ File saved to: {temp_path}")

        # 2. Upload to Gemini Files API (FIXED: use 'file=' not 'path=')
        print(f"☁️ Uploading to Gemini Files API...")
        uploaded_file = await run_in_threadpool(
            client.files.upload,
            file=temp_path  # Changed from path= to file=
        )
        print(f"✅ File uploaded. Gemini file name: {uploaded_file.name}")
        print(f"   URI: {uploaded_file.uri}")
        print(f"   State: {uploaded_file.state}")

        # 3. Wait for file to be processed (if needed)
        if hasattr(uploaded_file, 'state') and uploaded_file.state.name == 'PROCESSING':
            print("⏳ File is processing, waiting...")
            import time
            max_wait = 30  # seconds
            waited = 0
            while uploaded_file.state.name == 'PROCESSING' and waited < max_wait:
                await run_in_threadpool(time.sleep, 2)
                waited += 2
                uploaded_file = await run_in_threadpool(
                    client.files.get,
                    name=uploaded_file.name
                )
                print(f"   State: {uploaded_file.state.name}")
            
            if uploaded_file.state.name != 'ACTIVE':
                raise HTTPException(
                    status_code=500,
                    detail=f"File processing failed. State: {uploaded_file.state.name}"
                )

        # 4. Define the multimodal prompt
        asl_prompt = (
            "Analyze the signs, gestures, and facial expressions visible in this video. "
            "Based on American Sign Language (ASL), transcribe the signed content into "
            "a complete English sentence. you should always translate asl into english'. "
            "Only provide the transcribed sentence."
        )

        # 5. Call Gemini API for content generation
        print("🤖 Calling Gemini API for ASL analysis...")
        
        response = await run_in_threadpool(
            client.models.generate_content,
            model='gemini-2.0-flash-exp',
            contents=[
                uploaded_file,  # Simplified - just pass the file object
                asl_prompt
            ]
        )
        
        print(f"✅ Gemini API response received")
        
        # 6. Extract response text
        if not response or not response.text:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response"
            )

        transcription = response.text.strip()
        print(f"📝 Transcription: {transcription}")

        # 7. Check if interpretation was unclear
        if transcription.lower().startswith("asl interpretation unclear"):
            return {
                "filename": video_file.filename,
                "status": "Inconclusive",
                "gemini_response": transcription,
                "message": "ASL interpretation was unclear or could not be determined"
            }

        # 8. Return successful result
        return {
            "filename": video_file.filename,
            "status": "Success",
            "gemini_response": transcription,
            "file_uri": uploaded_file.uri
        }

    except APIError as e:
        print(f"❌ Gemini API Error: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Gemini API Error: {str(e)}"
        )
    
    except HTTPException:
        raise
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Server error: {str(e)}"
        )
        
    finally:
        # 9. Clean up files
        if temp_path or uploaded_file:
            uploaded_file_name = uploaded_file.name if uploaded_file else ""
            await run_in_threadpool(cleanup_sync, temp_path, uploaded_file_name)

# Test endpoint to verify Gemini connection
@app.get("/test-gemini")
async def test_gemini():
    """Test endpoint to verify Gemini API is working"""
    try:
        response = await run_in_threadpool(
            client.models.generate_content,
            model='gemini-2.0-flash-exp',
            contents="Say 'Hello, Gemini is working!'"
        )
        return {
            "status": "success",
            "response": response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Serve the HTML frontend
@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the HTML frontend"""
    with open("frontend.html", "r") as f:
        return f.read()

# Run: uvicorn tryasl.backend:app --reload