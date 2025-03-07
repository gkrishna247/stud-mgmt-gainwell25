# face_recognition_system.py
import os
import cv2
import face_recognition
import argparse
import pickle

def train_model(training_dir="known_people", model_save_path="trained_model.pkl"):
    """
    Trains a face recognition model using images from a directory
    Format: training_dir/PersonName/image1.jpg
    """
    known_encodings = []
    known_names = []

    print(f"⚙️ Training model using images from '{training_dir}'...")
    
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"✅ Learned {person_name} from {image_file}")
                else:
                    print(f"⚠️ No face found in {image_path}")

    # Save trained model
    with open(model_save_path, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    
    print(f"🎉 Training complete! Saved model to {model_save_path}")
    return known_encodings, known_names

def realtime_recognition(model_path="trained_model.pkl"):
    """Performs real-time face recognition using webcam"""
    # Load trained model
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    except FileNotFoundError:
        print("❌ No trained model found. Train first with --mode train")
        return

    video_capture = cv2.VideoCapture(0)
    process_frame = True  # Process every other frame to speed up

    print("🚀 Starting real-time recognition. Press Q to quit...")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize and convert to RGB
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_frame:
            # Detect faces and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for encoding in face_encodings:
                # Compare with known faces
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                
                if True in matches:
                    # Find best match
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_index = face_distances.argmin()
                    if face_distances[best_match_index] < 0.6:
                        name = known_names[best_match_index]

                face_names.append(name)

        process_frame = not process_frame  # Toggle frame processing

        # Display results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--mode", choices=["train", "recognize"], required=True,
                       help="Mode: train or recognize")
    parser.add_argument("--training-data", default="known_people",
                       help="Path to training data directory")
    parser.add_argument("--model", default="trained_model.pkl",
                       help="Path to save/load trained model")
    
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.training_data, args.model)
    elif args.mode == "recognize":
        realtime_recognition(args.model)
