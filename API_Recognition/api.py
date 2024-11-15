from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config_qdrant import QdrantConfig
import uvicorn
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid
import base64

app = FastAPI(title="Face Recognition API")

# Initialize FaceAnalysis
face_analyzer = FaceAnalysis(name="buffalo_s")
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Initialize Qdrant
qdrant = QdrantConfig(collection_name='face_embeddings_v2')


class FaceAddRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    name: str = Field(..., description="Unique identifier for this specific face")
    common_name: str = Field(..., description="Common name or group for this face")


class FaceUpdateRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    new_name: Optional[str] = None
    new_common_name: Optional[str] = None


class FaceRecognizeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    limit: int = Field(1, ge=1)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to image."""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]

        # Decode base64 string
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


async def get_face_embedding(img: np.ndarray):
    """Extract face embedding from image."""
    faces = face_analyzer.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return face.embedding, face.bbox


@app.post("/faces/add")
async def add_face(request: FaceAddRequest):
    """Add a new face to the database."""
    try:
        # Decode base64 image
        img = decode_base64_image(request.image_base64)
        embedding, bbox = await get_face_embedding(img)

        # Check if face already exists
        existing_faces = qdrant.search_embedding(embedding, limit=1)
        if existing_faces and existing_faces[0].score > 0.9:
            raise HTTPException(
                status_code=400,
                detail=f"Face already exists as {existing_faces[0].payload['name']}"
            )

        # Generate unique ID for the face
        face_id = str(uuid.uuid4())

        # Insert into Qdrant
        qdrant.insert_embedding(
            embedding=embedding,
            name=request.name,
            common_name=request.common_name,
            face_id=face_id
        )

        return JSONResponse(
            content={
                "status": "success",
                "message": "Face added successfully",
                "face_id": face_id,
                "details": {
                    "name": request.name,
                    "common_name": request.common_name,
                    "bbox": bbox.tolist()
                }
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/faces/{face_id}")
async def update_face(face_id: str, request: FaceUpdateRequest):
    """Update an existing face record."""
    try:
        # Get existing face data
        existing_face = qdrant.get_face_by_id(face_id)
        if not existing_face:
            raise HTTPException(status_code=404, detail="Face not found")

        # Update embedding if new image provided
        new_embedding = None
        if request.image_base64:
            img = decode_base64_image(request.image_base64)
            new_embedding, _ = await get_face_embedding(img)

        # Update face record
        qdrant.update_face(
            face_id=face_id,
            new_embedding=new_embedding,
            new_name=request.new_name,
            new_common_name=request.new_common_name
        )

        return JSONResponse(
            content={
                "status": "success",
                "message": "Face updated successfully",
                "face_id": face_id
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/faces/{face_id}")
async def delete_face(face_id: str):
    """Delete a face from the database."""
    try:
        success = qdrant.delete_face(face_id)
        if not success:
            raise HTTPException(status_code=404, detail="Face not found")

        return JSONResponse(
            content={
                "status": "success",
                "message": "Face deleted successfully",
                "face_id": face_id
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces")
async def list_faces(
        common_name: Optional[str] = None,
        name: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
):
    """List all faces with optional filtering."""
    try:
        faces = qdrant.list_faces(
            common_name=common_name,
            name=name,
            skip=skip,
            limit=limit
        )
        return JSONResponse(
            content={
                "status": "success",
                "total": len(faces),
                "faces": faces
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces/{face_id}")
async def get_face(face_id: str):
    """Get details of a specific face."""
    try:
        face = qdrant.get_face_by_id(face_id)
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")

        return JSONResponse(
            content={
                "status": "success",
                "face": face
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/faces/recognize")
async def recognize_face(request: FaceRecognizeRequest):
    """Recognize faces in an uploaded image."""
    try:
        img = decode_base64_image(request.image_base64)
        faces = face_analyzer.get(img)

        if not faces:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "No faces detected in the image"
                }
            )

        results = []
        for face in faces:
            embedding = face.embedding
            matches = qdrant.search_embedding(embedding, limit=request.limit)

            face_matches = []
            for match in matches:
                if match.score >= request.threshold:
                    face_matches.append({
                        "face_id": match.id,
                        "name": match.payload.get("name"),
                        "common_name": match.payload.get("name_common"),
                        "confidence": float(match.score)
                    })

            if face_matches:
                results.append({
                    "bbox": face.bbox.tolist(),
                    "matches": face_matches
                })

        return JSONResponse(
            content={
                "status": "success",
                "faces_found": len(faces),
                "faces_recognized": len(results),
                "results": results
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
