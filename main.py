from fastapi import FastAPI, UploadFile, Form, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import whisper
import difflib
import os
import logging
import time
import sqlite3
import json
from typing import Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Database setup
DB_PATH = "quran_tasks.db"

def init_db():
    """Initialize the SQLite database with a tasks table."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create tasks table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            ayah TEXT,
            user_text TEXT,
            feedback TEXT,
            error TEXT,
            created_at INTEGER,
            completed_at INTEGER
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

# Create DB at startup
init_db()

def get_db():
    """Database connection dependency."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()

# Use the smallest possible model to reduce resource usage
MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large
logger.info(f"Loading Whisper model: {MODEL_SIZE}")

# Configure Whisper to use minimal resources
os.environ["WHISPER_USE_CPU"] = "1"  # Force CPU usage
os.environ["WHISPER_CPU_THREADS"] = "1"  # Limit threads

# Load model at startup with minimal settings
try:
    model = whisper.load_model(MODEL_SIZE)
    logger.info(f"Whisper {MODEL_SIZE} model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    model = None

# Background processing function
def process_audio(file_path: str, ayah: str, task_id: str):
    """Process audio in background and update database with results."""
    conn = None
    try:
        logger.info(f"Starting transcription for task {task_id}")
        start_time = time.time()
        
        # Update task status to processing
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tasks SET status = ? WHERE task_id = ?", 
            ("processing", task_id)
        )
        conn.commit()
        conn.close()
        conn = None
        
        # Transcribe with minimal settings
        result = model.transcribe(
            file_path, 
            language="ar",
            fp16=False,  # Use FP32 on CPU
            verbose=False  # Reduce log output
        )
        
        # Clean up the file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
        
        user_text = result["text"]
        feedback = "\n".join(difflib.ndiff(ayah, user_text))
        processing_time = time.time() - start_time
        
        # Update database with results
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE tasks 
            SET status = ?, user_text = ?, feedback = ?, completed_at = ?
            WHERE task_id = ?
            """, 
            ("completed", user_text, feedback, int(time.time()), task_id)
        )
        conn.commit()
        logger.info(f"Task {task_id} completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        # Update database with error
        try:
            if conn is None:
                conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tasks SET status = ?, error = ? WHERE task_id = ?", 
                ("error", str(e), task_id)
            )
            conn.commit()
        except Exception as db_error:
            logger.error(f"Failed to update error status: {str(db_error)}")
    finally:
        if conn:
            conn.close()
        # Always try to clean up files
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file {file_path} after error")
            except:
                pass

@app.post("/quran-correct")
async def quran_correct(
    background_tasks: BackgroundTasks,
    file: UploadFile, 
    ayah: str = Form(...),
    db: sqlite3.Connection = Depends(get_db)
):
    """Process Quranic recitation and compare with the correct ayah."""
    try:
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Received file: {file.filename} for task {task_id}")
        logger.info(f"Ayah: {ayah}")
        
        # Create task record in database
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO tasks (task_id, status, ayah, created_at) VALUES (?, ?, ?, ?)",
            (task_id, "pending", ayah, int(time.time()))
        )
        db.commit()
        
        # Save the uploaded file with a unique name to prevent collisions
        audio_path = f"temp_{task_id}_{file.filename}"
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to: {audio_path}")
        
        # Start background processing
        background_tasks.add_task(process_audio, audio_path, ayah, task_id)
        
        return JSONResponse(content={
            "task_id": task_id,
            "status": "pending",
            "message": "Audio processing started in background"
        })
                
    except Exception as e:
        logger.error(f"Error in quran-correct endpoint: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.get("/task/{task_id}")
async def get_task_status(task_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Check the status of an async processing task."""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        task = cursor.fetchone()
        
        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": "Task not found"}
            )
        
        # Convert to dict
        task_dict = dict(task)
        
        return JSONResponse(content=task_dict)
    
    except Exception as e:
        logger.error(f"Error retrieving task {task_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve task: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "database": os.path.exists(DB_PATH)
    }

@app.get("/tasks")
async def list_tasks(limit: int = 10, db: sqlite3.Connection = Depends(get_db)):
    """List recent tasks (admin endpoint)."""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT task_id, status, created_at FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))
        tasks = [dict(task) for task in cursor.fetchall()]
        
        return JSONResponse(content={"tasks": tasks})
    
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list tasks: {str(e)}"}
        )