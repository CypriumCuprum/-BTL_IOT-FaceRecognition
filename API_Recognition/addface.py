import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config_qdrant import QdrantConfig

# Initialize the FaceAnalysis app
app = FaceAnalysis(name="buffalo_s")
app.prepare(ctx_id=0, det_size=(640, 640))
qdrant = QdrantConfig(collection_name='face_embeddings_v2')


def add_face_from_image(image_path, name, common_name):
    """
    Add a face from an image to the Qdrant database.

    Args:
        image_path (str): The path to the image file.
        name (str): The name associated with the face.
    """
    img = cv2.imread(image_path)
    faces = app.get(img)

    # just get face biggest in faces
    if len(faces) < 0:
        return False
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    embedding = face.embedding
    qdrant.insert_embedding(embedding, name, common_name)

    return True


if __name__ == '__main__':
    # file_image = 'D:\CUPRUM\PTIT\Term 7\IoT\Recognition\FaceRecognition_DL\dataset\Dong\WIN_20241022_01_36_41_Pro.jpg'
    file_image1 = "C:\\Users\\admin\\OneDrive\\Pictures\\Camera Roll\\WIN_20241022_01_36_41_Pro.jpg"
    file_image2 = "C:\\Users\\admin\\OneDrive\\Pictures\\Camera Roll\\WIN_20241030_22_49_16_Pro.jpg"
    if add_face_from_image(file_image1, 'Cuprum1', "cuprum") and add_face_from_image(file_image2, 'Cuprum2', "cuprum"):
        print('Done')
