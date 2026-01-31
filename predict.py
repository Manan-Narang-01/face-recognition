# predict.py

import os
import pickle
import shutil
import numpy as np
from utils import get_face_encoding
from sklearn.metrics.pairwise import cosine_similarity

# Load trained encodings
with open("trained_model/encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

test_dir = "test"
output_dir = "result"
unknown_folder = os.path.join(output_dir, "other_person")
skipped_folder = os.path.join(output_dir, "skipped_photos")
os.makedirs(unknown_folder, exist_ok=True)
os.makedirs(skipped_folder, exist_ok=True)

# Higher similarity = better match. Typical threshold: 0.55
# You can tune this: 0.5 (strict) to 0.7 (lenient)
SIMILARITY_THRESHOLD = 0.7

def get_unique_filename(dest_path):
    """Generate unique filename if file already exists"""
    base, ext = os.path.splitext(dest_path)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = f"{base} ({counter}){ext}"
        counter += 1
    return dest_path

print(f"[INFO] Loaded {len(known_encodings)} known face encodings")
print(f"[INFO] Processing test images from '{test_dir}'...")

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    
    # Skip if not a file
    if not os.path.isfile(img_path):
        continue
    
    # Get face encoding
    encoding = get_face_encoding(img_path)

    # No face detected
    if encoding is None:
        dest_path = get_unique_filename(os.path.join(skipped_folder, img_name))
        shutil.copy(img_path, dest_path)  
        print(f"[!] Skipped {img_name} (no face found) → copied to 'skipped_photos'")
        continue

    # Compare with known faces using cosine similarity
    similarities = cosine_similarity([encoding], known_encodings)[0]
    best_match_index = np.argmax(similarities)
    max_similarity = similarities[best_match_index]

    # Convert similarity to percentage
    confidence = max_similarity * 100

    # Determine if it's a match
    if max_similarity < SIMILARITY_THRESHOLD:
        dest_folder = unknown_folder
        label = "other_person"
    else:
        label = known_names[best_match_index]
        dest_folder = os.path.join(output_dir, label)

    # Create folder and copy image
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = get_unique_filename(os.path.join(dest_folder, img_name))
    shutil.copy(img_path, dest_path)
    
    print(f"{img_name}: {label} (similarity: {max_similarity:.3f}, confidence: {confidence:.2f}%)")

print(f"\n[INFO] Processing complete! Results saved in '{output_dir}'")