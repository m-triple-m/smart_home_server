import face_recognition
import os
import numpy as np
import shutil

# Default directory where known face images are stored (Name.jpg)
DEFAULT_FACES_DIR = os.path.join('static', 'faces')

def match_face(target_image_path, faces_dir=DEFAULT_FACES_DIR):
    """
    Matches a given face image against all faces stored in faces_dir.
    
    Returns:
        dict: Match results including name, confidence, and distance.
    """
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
        return {"status": "error", "message": f"Storage directory '{faces_dir}' created. Please add images."}

    # 1. Load and encode the target image
    try:
        target_image = face_recognition.load_image_file(target_image_path)
        target_encodings = face_recognition.face_encodings(target_image)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load image: {str(e)}"}
    
    if not target_encodings:
        return {"status": "error", "message": "No face detected in the image."}
    
    target_encoding = target_encodings[0]
    
    known_face_encodings = []
    known_face_names = []

    # 2. Scan the faces directory for known people
    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(faces_dir, filename)
            try:
                # We can speed this up later by saving/loading .npy files of encodings
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_face_encodings.append(encodings[0])
                    # Use filename (without extension) as the person's name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
            except Exception as e:
                print(f"Skipping {filename}: {e}")

    if not known_face_encodings:
        return {"status": "error", "message": "No faces found in storage."}

    # 3. Compare and find the best match
    face_distances = face_recognition.face_distance(known_face_encodings, target_encoding)
    best_match_index = np.argmin(face_distances)
    
    # 0.6 is the standard threshold. Lower = stricter.
    tolerance = 0.6 
    if face_distances[best_match_index] <= tolerance:
        match_name = known_face_names[best_match_index]
        return {
            "status": "success",
            "match_found": True,
            "name": match_name,
            "confidence": round((1 - face_distances[best_match_index]) * 100, 2),
            "distance": float(face_distances[best_match_index])
        }
    else:
        return {
            "status": "success",
            "match_found": False,
            "message": "Face detected but no match found."
        }

def add_new_face(image_path, name, faces_dir=DEFAULT_FACES_DIR):
    """
    Validates that a face exists in the image, then saves it to the faces directory.
    """
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    try:
        # 1. Verify there is a face in the image before saving
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if not encodings:
            return {"status": "error", "message": "No face detected. Cannot add this image."}
        
        # 2. Save the file with the given name
        # Sanitize name for filename
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '_')]).strip()
        if not safe_name:
            return {"status": "error", "message": "Invalid name provided."}
            
        extension = os.path.splitext(image_path)[1]
        if not extension:
            extension = ".jpg"
            
        new_filename = f"{safe_name}{extension}"
        dest_path = os.path.join(faces_dir, new_filename)
        
        # Copy the image to the faces directory
        shutil.copy(image_path, dest_path)
        
        return {"status": "success", "message": f"Added face for {safe_name}.", "name": safe_name}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_all_faces(faces_dir=DEFAULT_FACES_DIR):
    """
    Returns a list of names of all registered faces.
    """
    if not os.path.exists(faces_dir):
        return []
    
    names = []
    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            names.append(name)
    return sorted(list(set(names)))

# Example Usage:
if __name__ == "__main__":
    # Test matching
    test_image = os.path.join('static', 'sample6.jpg')
    if os.path.exists(test_image):
        print(f"Testing with: {test_image}")
        result = match_face(test_image)
        print(f"Result: {result}")
    else:
        print(f"Test file {test_image} not found.")
