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

@app.post("/recognize/")
async def recognize_face(video: UploadFile = File(...), db: Session = Depends(get_db)):
    # Load known faces from the database
    known_face_encodings, known_face_names = load_known_faces(db)

    # Write the uploaded video file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    # Open the video file
    video_capture = cv2.VideoCapture(tmp_path)
    if not video_capture.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video file.")

    # Set of seen faces to avoid printing the same face multiple times
    seen_faces = set()

    # Process each frame
    recognized_faces = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Rotate the frame (as you did in the notebook to fix orientation)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Scale back to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Add the name to the set if it's not already in
            if name != "Unknown" and name not in seen_faces:
                seen_faces.add(name)
                recognized_faces.append(name)

    # Close the video file
    video_capture.release()

    # Return the recognized faces
    return {"recognized_faces": recognized_faces}

@app.post("/add-face")
async def add_face(
    name: str = Form(...),
    employee_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Read image
    image_bytes = await file.read()
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    # Extract face encoding
    encodings = face_recognition.face_encodings(image)
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
