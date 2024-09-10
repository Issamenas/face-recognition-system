import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.models.embedding_model import FaceEmbedder
from src.utils.vector_store import VectorStore

def compute_and_store_embeddings(data_dir, output_file):
    face_embedder = FaceEmbedder()
    vector_store = VectorStore(128)  #face_recognition produces 128-dimensional embeddings
    
    for person_name in tqdm(os.listdir(data_dir), desc="Processing people"):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                embedding = face_embedder.get_embedding_from_file(image_path)
                if embedding is not None:
                    vector_store.add_embedding(embedding, person_name)
    
    vector_store.save(output_file)
    print(f"Embeddings computed and stored in {output_file}")

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'embeddings', 'vector_store.pkl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    compute_and_store_embeddings(data_dir, output_file)