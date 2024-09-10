import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from src.models.embedding_model import FaceEmbedder
from src.utils.vector_store import VectorStore

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize face embedder and vector store
face_embedder = FaceEmbedder()
vector_store = VectorStore.load('data\\embeddings\\vector_store.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Read and preprocess the image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get embedding
        embedding = face_embedder.get_embedding(image)
        
        if embedding is None:
            return jsonify({'error': 'No face detected in the image'})
        
        # Search for similar faces
        results = vector_store.search(embedding)
        
        if results:
            closest_match, distance = results[0]
            return jsonify({'match': closest_match, 'distance': float(distance)})
        else:
            return jsonify({'error': 'No match found'})

if __name__ == '__main__':
    app.run(debug=True)
