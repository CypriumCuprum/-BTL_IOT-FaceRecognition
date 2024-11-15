# from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams
# import uuid  # Import UUID for unique ID generation
#
#
# class QdrantConfig:
#     def __init__(self, host='localhost', port=6333, collection_name='face_embeddings_v2'):
#         self.client = QdrantClient(host=host, port=port)
#         self.collection_name = collection_name
#
#     def create_collection(self):
#         """Create a collection for storing face embeddings."""
#         if self.client.collection_exists(self.collection_name):
#             print(f"Collection '{self.collection_name}' already exists.")
#             return
#         self.client.create_collection(
#             collection_name=self.collection_name,
#             vectors_config=VectorParams(size=512, distance='Cosine')  # Adjust the size based on your model
#         )
#         print(f"Collection '{self.collection_name}' created.")
#
#     def insert_embedding(self, embedding, name, name_common):
#         """Insert a new face embedding into the collection."""
#         payload = {'name': name, "name_common": name_common}
#         # check if embedding is already exist
#         search_result = self.search_embedding(embedding, limit=1)
#         if len(search_result) > 0 and search_result[0].score > 0.9:
#             print(f"Embedding for {name} already exists.")
#             return
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=[{
#                 'id': str(uuid.uuid4()),  # Generate a unique ID using UUID
#                 'vector': embedding,
#                 'payload': payload
#             }]
#         )
#         print(f"Inserted embedding for {name}.")
#
#     def search_embedding(self, embedding, limit=1):
#         """Search for a similar face embedding."""
#         search_result = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=embedding,
#             limit=limit
#         )
#         return search_result
#
#     def delete_collection(self):
#         """Delete the collection."""
#         self.client.delete_collection(self.collection_name)
#         print(f"Collection '{self.collection_name}' deleted.")
#
#
# # Example usage:
# if __name__ == "__main__":
#     qdrant = QdrantConfig(collection_name='face_embeddings_v2')
#     qdrant.create_collection()
#     qdrant.delete_collection()
#     # Insert example embedding (replace with actual data)
#     # embedding_example = [0.1, 0.2, ..., 0.5]  # Your actual embedding
#     # qdrant.insert_embedding(embedding_example, "John Doe")

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Filter, FieldCondition, MatchValue
from qdrant_client.http.models import UpdateStatus
import uuid
from typing import Optional, List, Dict, Any
import os


class QdrantConfig:
    def __init__(self, collection_name='face_embeddings_v2'):
        host = os.getenv('QDRANT_HOST', 'localhost')
        port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = 512
        self.create_collection()

    def create_collection(self):
        """Create a collection for storing face embeddings."""
        if self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists.")
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance='Cosine')
        )
        print(f"Collection '{self.collection_name}' created.")

    def insert_embedding(self, embedding, name: str, common_name: str, face_id: str = None):
        """Insert a new face embedding into the collection."""

        # check if embedding is already exist or almost similar
        search_result = self.search_embedding(embedding, limit=1)
        if len(search_result) > 0 and search_result[0].score > 0.95:
            print(f"Embedding for {name} already exists.")
            return

        if face_id is None:
            face_id = str(uuid.uuid4())

        payload = {
            'name': name,
            'name_common': common_name,
            'face_id': face_id
        }

        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                'id': face_id,
                'vector': embedding.tolist(),
                'payload': payload
            }]
        )
        return face_id

    def update_face(self, face_id: str, new_embedding=None, new_name=None, new_common_name=None):
        """Update an existing face record."""
        updates = {}
        if new_name is not None:
            updates['name'] = new_name
        if new_common_name is not None:
            updates['name_common'] = new_common_name

        if updates:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=updates,
                points=[face_id]
            )

        if new_embedding is not None:
            self.client.update_vectors(
                collection_name=self.collection_name,
                vectors={face_id: new_embedding.tolist()}
            )

        return True

    def delete_face(self, face_id: str) -> bool:
        """Delete a face from the collection."""
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=[face_id]
        )
        return result.status == UpdateStatus.COMPLETED

    def get_face_by_id(self, face_id: str) -> Optional[Dict[str, Any]]:
        """Get face details by ID."""
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[face_id]
        )
        if not points:
            return None

        point = points[0]
        return {
            'face_id': point.id,
            'name': point.payload.get('name'),
            'common_name': point.payload.get('name_common')
        }

    def list_faces(
            self,
            common_name: Optional[str] = None,
            name: Optional[str] = None,
            skip: int = 0,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List faces with optional filtering."""
        conditions = []
        if common_name:
            conditions.append(
                FieldCondition(key='name_common', match=MatchValue(value=common_name))
            )
        if name:
            conditions.append(
                FieldCondition(key='name', match=MatchValue(value=name))
            )

        filter_param = Filter(must=conditions) if conditions else None

        points = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_param,
            limit=limit,
            offset=skip
        )[0]

        return [
            {
                'face_id': point.id,
                'name': point.payload.get('name'),
                'common_name': point.payload.get('name_common')
            }
            for point in points
        ]

    def search_embedding(self, embedding, limit=1):
        """Search for similar face embeddings."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=limit
        )

    def delete_collection(self):
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
