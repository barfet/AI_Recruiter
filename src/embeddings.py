from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List, Union

from src.data_models import JobPosting, CandidateProfile

class VectorStoreManager:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.embedder = OpenAIEmbeddings(model=model_name)
        self.stores = {
            'jobs': None,
            'candidates': None
        }
    
    def create_embeddings(self, 
                         documents: List[Union[JobPosting, CandidateProfile]], 
                         store_type: str):
        """Convert documents to text and embed"""
        texts = []
        for doc in documents:
            if isinstance(doc, JobPosting):
                texts.append(f"{doc.title}\n{doc.description}\n{doc.requirements}")
            else:
                texts.append(f"{' '.join(doc.experience)}\n{' '.join(doc.skills)}")
        
        self.stores[store_type] = FAISS.from_texts(texts, self.embedder)
    
    def similarity_search(self, query: str, store_type: str, k=3):
        """Retrieve top k matches"""
        return self.stores[store_type].similarity_search(query, k=k) 