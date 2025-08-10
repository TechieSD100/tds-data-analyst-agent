import requests
import os
import json
import base64

# Basic integration-style test: requires local server running on :8080
def test_wikipedia_sample():
    url = os.environ.get("AGENT_URL", "http://localhost:8080/")
    files = {
        'files': ('questions.txt', open('sample_inputs/question_wikipedia.txt','rb'), 'text/plain')
    }
    # send as multipart/form-data with field name `files` multiple times
    r = requests.post(url, files=[('files', ('questions.txt', open('sample_inputs/question_wikipedia.txt','rb')))])
    assert r.status_code == 200
    out = r.json()
    # basic structural checks
    assert isinstance(out, list) and len(out) == 4
    # 1st item must be an int
    assert isinstance(out[0], int)
    # 2nd must be string
    assert isinstance(out[1], str)
    # 3rd must be numeric
    assert isinstance(out[2], float) or isinstance(out[2], int)
    # 4th must be data URI string
    assert isinstance(out[3], str) and out[3].startswith("data:image/")
