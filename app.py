import os
from flask import Flask, request, jsonify

port = int(os.environ.get("PORT", 8080))  # Railway sets PORT automatically
app.run(host="0.0.0.0", port=port)
