# utils.py
import cv2
import face_recognition

def get_face_encoding(image_path):
    """
    Simple face detection and encoding using face_recognition library
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect and encode faces
    encodings = face_recognition.face_encodings(rgb)
    
    if len(encodings) == 0:
        return None
    
    return encodings[0]