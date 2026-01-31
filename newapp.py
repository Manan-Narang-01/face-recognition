# app.py
import streamlit as st
import os
import pickle
import numpy as np
from utils import get_face_encoding
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Constants
SIMILARITY_THRESHOLD = 0.55  # Using cosine similarity (higher is better)
ENCODINGS_PATH = "trained_model/encodings.pkl"
DATASET_DIR = "dataset"

# Ensure directories exist
os.makedirs("trained_model", exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Cache model load to avoid reloading on every interaction
@st.cache_data(show_spinner=False)
def load_model():
    """Load trained face encodings"""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            known_encodings, known_names = pickle.load(f)
        return known_encodings, known_names
    return [], []

def save_model(encodings, names):
    """Save encodings to pickle file"""
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump((encodings, names), f)
    # Clear cache to reload updated model
    st.cache_data.clear()

def add_person_to_model(name, image_path):
    """Add a new person to the trained model"""
    # Get face encoding
    encoding = get_face_encoding(image_path)
    
    if encoding is None:
        return False, "No face detected in the image. Please upload a clear face photo."
    
    # Load current model
    known_encodings, known_names = load_model()
    
    # Check if person already exists
    if name in known_names:
        return False, f"Person '{name}' already exists in the database."
    
    # Add new encoding
    known_encodings.append(encoding)
    known_names.append(name)
    
    # Save updated model
    save_model(known_encodings, known_names)
    
    return True, f"Successfully added '{name}' to the database!"

def recognize_face(image_path):
    """Recognize face in the uploaded image using cosine similarity"""
    known_encodings, known_names = load_model()
    
    if len(known_encodings) == 0:
        return None, None, "No trained faces in the database. Please add people first."
    
    # Get face encoding
    encoding = get_face_encoding(image_path)
    
    if encoding is None:
        return None, None, "No face detected in the image."
    
    # Use cosine similarity (better for face recognition)
    similarities = cosine_similarity([encoding], known_encodings)[0]
    best_match_index = np.argmax(similarities)
    max_similarity = similarities[best_match_index]
    
    # Convert similarity to percentage (0-100%)
    percentage_match = max_similarity * 100
    
    # Determine if it's a match
    if max_similarity < SIMILARITY_THRESHOLD:
        label = "Unknown Person"
    else:
        label = known_names[best_match_index]
    
    return label, percentage_match, None

# Streamlit UI
st.set_page_config(page_title="Face Recognition System", page_icon="👤", layout="wide")
st.title("👤 Face Recognition System")

# Load current model stats
known_encodings, known_names = load_model()
st.sidebar.header("📊 System Info")
st.sidebar.metric("Total People in Database", len(known_names))

# Add adjustable threshold in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
SIMILARITY_THRESHOLD = st.sidebar.slider(
    "Recognition Threshold",
    min_value=0.3,
    max_value=0.8,
    value=0.55,
    step=0.05,
    help="Higher = stricter matching (fewer false positives)"
)

if len(known_names) > 0:
    st.sidebar.write("**Registered People:**")
    for name in sorted(set(known_names)):
        st.sidebar.write(f"• {name}")

# Create tabs
tab1, tab2 = st.tabs(["🧪 Test/Recognize", "➕ Add New Person"])

# ============= TAB 1: RECOGNITION =============
with tab1:
    st.header("🔍 Face Recognition")
    st.write("Upload an image to identify the person")
    
    test_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="test")
    
    col1, col2 = st.columns(2)
    
    if test_file is not None:
        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        test_path = os.path.join(temp_dir, test_file.name)
        
        with open(test_path, "wb") as f:
            f.write(test_file.getbuffer())
        
        # Display uploaded image
        with col1:
            st.subheader("Uploaded Image")
            st.image(test_path, use_container_width=True)
        
        # Recognize face
        with col2:
            st.subheader("Recognition Result")
            with st.spinner("Analyzing face..."):
                label, percentage, error_msg = recognize_face(test_path)
            
            if error_msg:
                st.error(error_msg)
            elif label == "Unknown Person":
                st.warning(f"**{label}**")
                st.metric("Match Percentage", f"{percentage:.2f}%")
                st.info("💡 This person is not in the database. You can add them using the 'Add New Person' tab.")
            else:
                st.success(f"**Recognized: {label}**")
                st.metric("Match Percentage", f"{percentage:.2f}%")
                
                # Visual progress bar for match percentage
                st.progress(min(percentage / 100, 1.0))
        
        # Clean up
        try:
            os.remove(test_path)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            st.error(f"Error cleaning temp files: {e}")

# ============= TAB 2: ADD NEW PERSON =============
with tab2:
    st.header("➕ Add New Person to Database")
    st.write("Upload a clear photo of a person's face to add them to the recognition system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        person_name = st.text_input("Enter person's name:", placeholder="e.g., John Doe")
        train_file = st.file_uploader("Upload photo:", type=["jpg", "jpeg", "png"], key="train")
        
        if train_file is not None:
            st.image(train_file, caption="Preview", use_container_width=True)
    
    with col2:
        st.info("""
        **📝 Tips for best results:**
        - Use a clear, well-lit photo
        - Face should be clearly visible
        - Frontal view works best
        - Only one face in the image
        - Avoid sunglasses or face coverings
        """)
    
    if st.button("Add to Database", type="primary", use_container_width=True):
        if not person_name:
            st.error("Please enter a name!")
        elif train_file is None:
            st.error("Please upload an image!")
        else:
            # Save uploaded file temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            train_path = os.path.join(temp_dir, train_file.name)
            
            with open(train_path, "wb") as f:
                f.write(train_file.getbuffer())
            
            # Add person to database
            with st.spinner(f"Adding {person_name} to database..."):
                success, message = add_person_to_model(person_name.strip(), train_path)
            
            if success:
                # Save to dataset folder for backup
                person_dir = os.path.join(DATASET_DIR, person_name.strip())
                os.makedirs(person_dir, exist_ok=True)
                
                # Save original image to dataset
                dataset_img_path = os.path.join(person_dir, train_file.name)
                with open(dataset_img_path, "wb") as f:
                    f.write(train_file.getbuffer())
                
                st.success(message)
                st.info(f"📁 Image saved to: {dataset_img_path}")
                
                # Force UI refresh
                st.rerun()
            else:
                st.error(message)
            
            # Clean up
            try:
                os.remove(train_path)
                if not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                st.error(f"Error cleaning temp files: {e}")

# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Face Recognition System | Powered by face_recognition & Streamlit</small>
</div>
""", unsafe_allow_html=True)