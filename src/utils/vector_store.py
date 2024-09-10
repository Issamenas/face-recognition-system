import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.labels = []

    def add_embedding(self, embedding, label):
        if embedding.shape != (self.dimension,):
            raise ValueError(f"Embedding must be a {self.dimension}-dimensional vector")
        self.index.add(np.array([embedding]))
        self.labels.append(label)

    def search(self, query_embedding, k=1):
        if query_embedding.shape != (self.dimension,):
            raise ValueError(f"Query embedding must be a {self.dimension}-dimensional vector")
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [(self.labels[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'index': faiss.serialize_index(self.index), 'labels': self.labels}, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        vector_store = cls(0)  # Dimension will be set when loading the index
        vector_store.index = faiss.deserialize_index(data['index'])
        vector_store.dimension = vector_store.index.d
        vector_store.labels = data['labels']
        return vector_store
