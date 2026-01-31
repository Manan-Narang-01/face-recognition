# train.py
import os
import pickle
from utils import get_face_encoding
from tqdm import tqdm

dataset_dir = "dataset"
encodings = []
names = []

print("[INFO] Generating face encodings...")

# Main loop: iterate over people
for person in tqdm(os.listdir(dataset_dir), desc="People"):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    person_encoded = False
    for image_file in tqdm(os.listdir(person_dir), desc=f"Processing {person}", leave=False):
        image_path = os.path.join(person_dir, image_file)
        try:
            encoding = get_face_encoding(image_path)
        except Exception as e:
            print(f"[WARN] Failed to process {image_file}: {e}")
            continue

        if encoding is not None:
            encodings.append(encoding)
            names.append(person)
            person_encoded = True
            break  # Only one image per person

    if not person_encoded:
        print(f"[WARN] No valid face found for person: {person}")

# Save encodings
os.makedirs("trained_model", exist_ok=True)
with open("trained_model/encodings.pkl", "wb") as f:
    pickle.dump((encodings, names), f)

print(f"[INFO] Saved {len(encodings)} face encodings to trained_model/encodings.pkl")
