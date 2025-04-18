from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import whisper
import difflib
import os
import logging
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Use the smallest possible model to reduce resource usage
MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large
logger.info(f"Loading Whisper model: {MODEL_SIZE}")

# Load model at startup
try:
    model = whisper.load_model(MODEL_SIZE)
    logger.info(f"Whisper {MODEL_SIZE} model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model = None

# Cache for storing results (simple in-memory cache)
results_cache = {}

# Background processing function
def process_audio(file_path: str, ayah: str, task_id: str):
    try:
        logger.info(f"Starting transcription for task {task_id}")
        start_time = time.time()
        
        # Transcribe with timeout safety
        result = model.transcribe(file_path, language="ar")
        
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
        
        user_text = result["text"]
        feedback = "\n".join(difflib.ndiff(ayah, user_text))
        
        # Store result in cache
        results_cache[task_id] = {
            "status": "completed",
            "original_ayah": ayah,
            "user_text": user_text,
            "feedback": feedback,
            "processing_time": time.time() - start_time
        }
        logger.info(f"Task {task_id} completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        results_cache[task_id] = {
            "status": "error",
            "error": str(e)
        }

@app.post("/quran-correct")
async def quran_correct(
    background_tasks: BackgroundTasks,
    file: UploadFile, 
    ayah: str = Form(...),
    async_processing: bool = Form(False)
):
    """
    Process Quranic recitation and compare with the correct ayah.
    Set async_processing=True to process in the background.
    """
    try:
        # Generate a unique task ID
        task_id = f"task_{int(time.time())}"
        logger.info(f"Received file: {file.filename} for task {task_id}")
        logger.info(f"Ayah: {ayah}")
        
        # Save the uploaded file
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to: {audio_path}")
        
        if async_processing:
            # Start background processing
            results_cache[task_id] = {"status": "processing"}
            background_tasks.add_task(process_audio, audio_path, ayah, task_id)
            
            return JSONResponse(content={
                "task_id": task_id,
                "status": "processing",
                "message": "Audio processing started in background"
            })
        else:
            # Synchronous processing
            try:
                logger.info("Starting synchronous transcription")
                start_time = time.time()
                
                result = model.transcribe(audio_path)
                user_text = result["text"]
                feedback = "\n".join(difflib.ndiff(ayah, user_text))
                
                # Clean up
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                
                logger.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")
                
                return JSONResponse(content={
                    "original_ayah": ayah,
                    "user_text": user_text,
                    "feedback": feedback
                })
            
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                # Clean up on error
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                raise
                
    except Exception as e:
        logger.error(f"Error in quran-correct endpoint: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of an async processing task."""
    if task_id not in results_cache:
        return JSONResponse(
            status_code=404,
            content={"error": "Task not found"}
        )
    
    return JSONResponse(content=results_cache[task_id])

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "model_loaded": model is not None}