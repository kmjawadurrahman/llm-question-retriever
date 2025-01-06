from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.config import Config

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(path=Config.QDRANT_PATH)
    
    def create_collection(self, collection_name: str, vector_size: int):
        """Create or recreate collection"""
        # Delete if exists
        if self._collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)
        
        # Create new collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        collections = self.client.get_collections().collections
        return any(collection.name == collection_name for collection in collections)
    
    def upload_points(self, collection_name: str, points: List[PointStruct]):
        """Upload points to collection"""
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
    
    def prepare_point(self, idx: int, vector: List[float], 
                     ques_no: int, question: str, 
                     ques_template: str) -> PointStruct:
        """Prepare point for upload"""
        return PointStruct(
            id=idx,
            vector=vector,
            payload={
                'ques_no': ques_no,
                'question': question,
                'ques_template': ques_template
            }
        )
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10) -> List:
        """Search for similar vectors"""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )