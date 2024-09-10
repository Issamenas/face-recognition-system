# Face Recognition System

This project implements a simple face recognition system using machine learning techniques and open-source tools.

## Setup

1. Ensure you have Python 3.7+ installed.
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Place your dataset in the `data/raw` directory. Each person should have their own subfolder containing their images.

## Usage

1. Compute embeddings for your dataset:
   ```
   python scripts/compute_embeddings.py
   ```
2. Run the Flask application:
   ```
   python app/app.py
   ```
3. Open a web browser and go to `http://localhost:5000` to use the face recognition system.


## Project Structure

- `data/`: Contains the raw dataset and computed embeddings.
- `models/`: Contains the face embedding model.
- `utils/`: Includes preprocessing and vector store operations.
- `app/`: Flask application for the user interface.
- `scripts/`: Scripts for embedding computation.

## Notes

- ### Using a Mini Dataset:
For testing and experimentation, you can use a mini dataset of face images located in the data/mini_dataset directory (just change raw by mini_dataset in the compute_embeddings.py file "line 26").

- This system uses the face_recognition library for face detection and embedding computation.
- FAISS is used for efficient similarity search of face embeddings.
- The web interface allows users to upload an image and find the closest match in the dataset.

"# face-recognition-system" 
"# face-recognition-system" 
