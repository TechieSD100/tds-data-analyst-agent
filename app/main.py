import io
import asyncio
import traceback
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.requests import Request
from app.analysis import run_analysis
from app.utils import read_text_file

app = FastAPI(title="TDS Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# POST /  - accepts multipart/form-data
@app.post("/", summary="Run data analysis task")
async def analyze(request: Request, files: List[UploadFile] = File(...)):
    """
    Accepts one or more files. questions.txt is mandatory.
    Returns JSON (object or array) as required by the evaluator.
    """

    # load files into memory (small files expected)
    loaded = {}
    for f in files:
        content = await f.read()
        loaded[f.filename] = content

    if "questions.txt" not in loaded:
        raise HTTPException(status_code=400, detail="questions.txt is required")

    try:
        questions_text = read_text_file(loaded["questions.txt"])
        # run the analysis pipeline with a timeout to guarantee response within 3 minutes
        # We'll give a safe operational limit of 165 seconds (2m45s).
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(loop.run_in_executor(None, run_analysis, questions_text, loaded), timeout=165)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timed out")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
