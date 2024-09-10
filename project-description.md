# Face Recognition System: Project Overview

## 1. Introduction
This project implements a web-based face recognition system using machine learning techniques and open-source tools. It allows users to upload an image of a face and find the closest match within a pre-existing dataset of known individuals.

## 2. Project Components

### 2.1 Data Management (`data/` directory)
- **Raw Data**: 
  - Located in `data/raw/`
  - Contains the dataset of known faces, organized into subdirectories for each individual
  - Each subdirectory contains multiple images of the same person

- **Embeddings**: 
  - Stored in `data/embeddings/`
  - Contains pre-computed face embeddings for efficient searching

### 2.2 Core Functionality (`src/` directory)

#### 2.2.1 Face Embedding Model (`src/models/embedding_model.py`)
- Utilizes the `face_recognition` library
- Responsible for:
  - Detecting faces in images
  - Computing 128-dimensional face embeddings
- Key features:
  - Can process both in-memory images and image files
  - Handles cases where no face is detected

#### 2.2.2 Vector Store (`src/utils/vector_store.py`)
- Implements efficient similarity search using FAISS (Facebook AI Similarity Search)
- Key functionalities:
  - Storing face embeddings and associated labels
  - Performing fast nearest-neighbor searches
  - Saving and loading the index for persistence

#### 2.2.3 Preprocessing (`src/utils/preprocess.py`)
- Handles image preprocessing tasks
- Currently implements resizing functionality

#### 2.2.4 Web Application (`src/app.py`)
- Flask-based web server
- Provides RESTful API for face recognition
- Handles:
  - Serving the web interface
  - Processing image uploads
  - Performing face recognition
  - Returning results to the user

### 2.3 User Interface
- **HTML Template** (`templates/index.html`):
  - Provides a simple, user-friendly interface for image upload
  - Displays recognition results

- **CSS Styling** (`static/styles.css`):
  - Defines the visual style of the web interface

### 2.4 Data Processing (`scripts/compute_embeddings.py`)
- Script for batch processing the raw dataset
- Computes embeddings for all faces in the dataset
- Stores the results in the vector store for quick retrieval

## 3. Technology Stack
- **Programming Language**: Python
- **Web Framework**: Flask
- **Face Recognition**: face_recognition library (based on dlib)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Image Processing**: OpenCV
- **Front-end**: HTML, CSS, JavaScript

## 4. Workflow
1. **Data Preparation**: Images of known individuals are stored in the raw data directory.
2. **Embedding Computation**: The `compute_embeddings.py` script processes all images, computing face embeddings.
3. **User Interaction**: A user uploads an image through the web interface.
4. **Face Recognition**:
   - The system detects the face in the uploaded image.
   - It computes the face embedding for the detected face.
   - The embedding is compared against the pre-computed embeddings in the vector store.
5. **Result Presentation**: The system returns the identity of the closest match, along with a confidence score.

## 5. Key Features
- Fast and efficient face recognition
- Web-based interface for easy access
- Scalable to large datasets of known faces
- Utilizes state-of-the-art face recognition and similarity search technologies



