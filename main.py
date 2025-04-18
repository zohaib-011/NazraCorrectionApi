from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import whisper
import difflib
import os

app = FastAPI()
# model = whisper.load_model("base")  # Use "tiny" if low memory
model = whisper.load_model("tiny")  # Change from "base" to "tiny"

# Dummy ayah (Surah Baqarah 2:285) — for test purposes
ayah_285 = "آمَنَ الرَّسُولُ بِمَا أُنزِلَ إِلَيْهِ مِن رَّبِّهِ وَالْمُؤْمِنُونَ..."

@app.post("/quran-correct")
async def quran_correct(file: UploadFile, ayah: str = Form(...)):
    try:
        print(f"Received file: {file.filename}")
        print(f"Ayah: {ayah}")

        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as f:
            f.write(await file.read())

        print(f"File saved to: {audio_path}")

        result = model.transcribe(audio_path, language="ar")
        os.remove(audio_path)

        user_text = result["text"]
        feedback = "\n".join(difflib.ndiff(ayah, user_text))

        return JSONResponse(content={
            "original_ayah": ayah,
            "user_text": user_text,
            "feedback": feedback
        })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
