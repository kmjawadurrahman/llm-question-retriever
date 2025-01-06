from typing import List, Union, Dict
import ast
from openai import OpenAI
from config.config import Config

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
    def get_embedding(self, text: Union[str, Dict]) -> List[float]:
        """Get embedding for a text or dictionary"""
        embedding_res = self.client.embeddings.create(
            input=str(text),
            model=Config.OPENAI_EMBEDDING_MODEL
        )
        return embedding_res.data[0].embedding
    
    @staticmethod
    def convert_embedding_str(embedding_str: str) -> List[float]:
        """Convert embedding string to list of floats"""
        return ast.literal_eval(str(embedding_str))