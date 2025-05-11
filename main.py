from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from database import SessionLocal
from models import KnownFace
import face_recognition
import numpy as np
import cv2
import pickle
import tempfile
from sqlalchemy.orm import Session
from PIL import Image
import io
from datetime import datetime, timedelta
import subprocess
import json

app = FastAPI()

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load known faces from database (we're assuming `KnownFace` model has `name` and `encoding` fields)
def load_known_faces(db: Session):
    known_face_encodings = []
    known_face_names = []
    known_faces = db.query(KnownFace).all()

    for known_face in known_faces:
        encoding = pickle.loads(known_face.encoding)
        known_face_encodings.append(encoding)
        known_face_names.append(known_face.name)

    return known_face_encodings, known_face_names

@app.get("/")
def home():
    return {"message": "Face Recognition Attendance API"}

# @app.post("/recognize/")
# async def recognize_face(video: UploadFile = File(...), db: Session = Depends(get_db)):
#     # Load known faces from the database
#     known_face_encodings, known_face_names = load_known_faces(db)

#     # Write the uploaded video file to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(await video.read())
#         tmp_path = tmp.name

#     # Open the video file
#     video_capture = cv2.VideoCapture(tmp_path)
#     if not video_capture.isOpened():
#         raise HTTPException(status_code=400, detail="Unable to open video file.")

#     # Set of seen faces to avoid printing the same face multiple times
#     seen_faces = set()

#     # Process each frame
#     recognized_faces = []

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break  # End of video

#         # Rotate the frame (as you did in the notebook to fix orientation)
#         frame = cv2.rotate(frame, cv2.ROTATE_180)

#         # Resize frame for faster processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

#         # Find face locations and encodings
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             if matches:
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = known_face_names[best_match_index]

#             # Scale back to original frame size
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             # Add the name to the set if it's not already in
#             if name != "Unknown" and name not in seen_faces:
#                 seen_faces.add(name)
#                 recognized_faces.append(name)

#     # Close the video file
#     video_capture.release()

#     # Return the recognized faces
#     return {"recognized_faces": recognized_faces}

@app.post("/recognize/")
async def recognize_face(
    video: UploadFile = File(...),
    base_time: str = Form(None),  # Optional fallback if metadata fails
    db: Session = Depends(get_db)
):
    # Step 1: Load known faces from DB
    known_faces = db.query(KnownFace).all()
    known_face_encodings = [pickle.loads(f.encoding) for f in known_faces]
    known_face_names = [f.name for f in known_faces]
    known_face_map = {f.name: f.employee_id for f in known_faces}

    # Step 2: Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    # Step 3: Extract video creation time or use base_time
    def get_video_start_time(path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
                capture_output=True,
                text=True,
                check=True
            )
            metadata = json.loads(result.stdout)
            creation_time_str = metadata['format']['tags'].get('creation_time')
            if creation_time_str:
                return datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))
        except Exception:
            pass
        return None

    video_start_time = get_video_start_time(tmp_path)
    if not video_start_time:
        if not base_time:
            raise HTTPException(status_code=400, detail="No video timestamp found and base_time not provided.")
        try:
            video_start_time = datetime.fromisoformat(base_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid base_time format. Use ISO format (e.g., '2025-05-04T10:00:00').")

    # Step 4: Read video
    video_capture = cv2.VideoCapture(tmp_path)
    if not video_capture.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video.")

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    seen_faces = set()
    recognized_faces = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # ✅ Frame timestamp
        frame_timestamp = video_start_time + timedelta(seconds=frame_number / fps)

        # Rotate if needed
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            if name != "Unknown" and name not in seen_faces:
                seen_faces.add(name)
                recognized_faces.append({
                    "name": name,
                    "employee_id": known_face_map.get(name),
                    "detected_at": frame_timestamp.isoformat()
                })

        frame_number += 1

    video_capture.release()

    return {"recognized_faces": recognized_faces}

# @app.post("/add-face")
# async def add_face(
#     name: str = Form(...),
#     employee_id: str = Form(...),
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db)
# ):
#     # Read image
#     image_bytes = await file.read()
#     image = np.array(Image.open(io.BytesIO(image_bytes)))

#     # Extract face encoding
#     encodings = face_recognition.face_encodings(image)
#     if not encodings:
#         raise HTTPException(status_code=400, detail="No face found in the image.")

#     encoding = encodings[0]
#     encoding_blob = pickle.dumps(encoding)

#     # Save to DB
#     new_face = KnownFace(
#         name=name,
#         employee_id=employee_id,
#         encoding=encoding_blob
#     )
#     db.add(new_face)
#     db.commit()
#     db.refresh(new_face)

#     return {"message": "Face added successfully", "id": new_face.id}

@app.post("/add-face")
async def add_face(
    name: str = Form(...),
    employee_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Ensure the image is in RGB format (important for face_recognition)
    image = image.convert("RGB")
    image_np = np.array(image)

    # Extract face encoding
    encodings = face_recognition.face_encodings(image_np)
    if not encodings:
        raise HTTPException(status_code=400, detail="No face found in the image.")

    encoding = encodings[0]
    encoding_blob = pickle.dumps(encoding)

    # Save to DB
    new_face = KnownFace(
        name=name,
        employee_id=employee_id,
        encoding=encoding_blob
    )
    db.add(new_face)
    db.commit()
    db.refresh(new_face)

    return {"message": "Face added successfully", "id": new_face.id}
