import io
import asyncio
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.requests import Request
from app.analysis import run_analysis
from app.utils import read_text_file
from flask import Flask
import os

app = FastAPI(title="TDS Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/", summary="Run data analysis task")
async def analyze(request: Request, files: List[UploadFile] = File(...)):
    """
    Accepts one or more files. questions.txt is mandatory.
    Returns JSON (object or array) as required by the evaluator.
    """

    loaded = {}
    for f in files:
        content = await f.read()
        loaded[f.filename] = content

    if "questions.txt" not in loaded:
        raise HTTPException(status_code=400, detail="questions.txt is required")

    try:
        questions_text = read_text_file(loaded["questions.txt"])
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(loop.run_in_executor(None, run_analysis, questions_text, loaded), timeout=165)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timed out")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
