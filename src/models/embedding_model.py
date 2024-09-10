import face_recognition
import numpy as np

class FaceEmbedder:
    def __init__(self):
        pass

    def get_embedding(self, image):
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = image[:, :, ::-1]
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if not face_locations:
            return None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return None
        
        # Return the first face encoding (assuming one face per image)
        return np.array(face_encodings[0])

    def get_embedding_from_file(self, image_path):
        # Load image file
        image = face_recognition.load_image_file(image_path)
        return self.get_embedding(image)