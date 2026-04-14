import os
import uuid
from backend.config import UPLOAD_DIR

def save_uploaded_file(file):
    # Store uploaded file with a random name to avoid collisions.
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(file.file.read())

    return path
