# -*- coding: utf-8 -*-
import face_recognition  # Core face recognition library
import cv2  # OpenCV for image and video processing
import numpy as np  # For numerical operations, especially with arrays
import os  # For interacting with the operating system (e.g., listing files)
import pickle  # For saving and loading data (e.g., face encodings)
from flask import Flask, render_template, request, Response, redirect, url_for, send_from_directory, jsonify # For creating the web application and image loading

app = Flask(__name__)


# --- Configuration and Setup ---

UPLOAD_FOLDER = 'uploads'
KNOWN_FACES_DIR = 'known_faces'  # Directory to store images of known people
TOLERANCE = 0.55  # Lower values are stricter.  0.6 is a common default, adjust based on your needs.
MODEL = 'hog'  # 'hog' (faster, less accurate) or 'cnn' (slower, more accurate)
ENCODINGS_FILE = 'face_encodings.pkl'  # File to store the face encodings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_known_faces():
    """Loads known faces and their encodings from the directory and pickle file."""
    known_face_encodings = []
    known_face_names = []

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
    else:
        # First-time setup, or encodings file missing
        print("Generating face encodings for the first time...")

    
    if os.path.exists(KNOWN_FACES_DIR):

      for name in os.listdir(KNOWN_FACES_DIR):  # Loop through each person's directory

          person_dir = os.path.join(KNOWN_FACES_DIR, name)

          if not os.path.isdir(person_dir):
              continue # Skip if it is not a directory

          for filename in os.listdir(person_dir):  # Loop through images for each person

            # Build path for the known face images

            # Added check for allowed extensions
            if allowed_file(filename):

              try:

                print(f"filename is {filename}")

                image_path = os.path.join(person_dir, filename)

                print(f"image_path is {image_path}")

                # Loading images directly as a byte array as expected by face_recognition
                with open(image_path, "rb") as f:
                  face_image = face_recognition.load_image_file(f)


                  #  Find all face encodings within the image using face_recognition's
                  # batch_face_locations: find all the locations of faces and use all these positions
                  # to get face_encodings

                  batch_of_face_locations = face_recognition.batch_face_locations([face_image], number_of_times_to_upsample=0, batch_size=128)

                  # Find the face encodings in the current image.  It is very important to keep the [0].
                  face_encodings = [ face_encoding
                                    for a in face_recognition.face_encodings(
                                        face_image, known_face_locations = batch_of_face_locations,
                                        model=MODEL,
                                        num_jitters = 1  )  for face_encoding in a]  # More jitters => more accurate (but slower)



                  if len(face_encodings) > 0:

                      print(f"length of encodings {len(face_encodings)}")
                      known_face_encodings.append(face_encodings[0])
                      known_face_names.append(name)
                  else:

                      # Catch error, such as "no face found in this picture"

                      print(f"No faces detected in {filename} or problem with the file.")


              # Generic catch all of errors to capture face_recognition problems and other general problems
              except Exception as e:

                #Print to console
                print(f"Error processing image {filename}: {e}")

                # Save errors into errors.txt, so can handle the image errors after program completion
                with open('errors.txt', 'a') as error_log:
                        error_log.write(f'Image_path = "{image_path}":{str(e)}\n')
                
    # Save new data or update file
    # Persists this so does not have to re-run everything again if a person is already known
    save_encodings(known_face_encodings, known_face_names)

    return known_face_encodings, known_face_names


def save_encodings(encodings, names):
    """Saves face encodings and names to a pickle file."""
    with open(ENCODINGS_FILE, 'wb') as f:
        data = {'encodings': encodings, 'names': names}
        pickle.dump(data, f)



def recognize_faces(frame, known_face_encodings, known_face_names):
    """Recognizes faces in a single frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    #  Find all face encodings within the image using face_recognition's
    # batch_face_locations: find all the locations of faces and use all these positions
    # to get face_encodings
    face_locations = face_recognition.batch_face_locations([rgb_frame], number_of_times_to_upsample=0, batch_size=128)

    # Find the face encodings in the current image.

    face_encodings = [ face_encoding
                      for a in face_recognition.face_encodings(
                        rgb_frame,
                        known_face_locations=face_locations,
                        model=MODEL,
                        num_jitters=1
    )
                    for face_encoding in a]  # More jitters => more accurate (but slower)


    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        name = "Unknown"  # Default if no match found

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if face_distances.size > 0:  # Make sure the face_distances has value for np.argmin
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)


    # Draw rectangles and labels on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)


    return frame



# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():

    """Handles the main page, including form submission and image display."""
    known_face_encodings, known_face_names = load_known_faces()  # Load (or create) encodings at start.


    if request.method == 'POST':
        # check if the post request has the file part

        if 'file' not in request.files:

            return redirect(request.url)
        
        file = request.files['file']


        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':

            return redirect(request.url)

        if file and allowed_file(file.filename):

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath) # Save image
           

            frame = cv2.imread(filepath)  # Correct usage: Read image using cv2.imread
            if frame is None:
                 return "Error: Could not open or read the image file."

            processed_frame = recognize_faces(frame, known_face_encodings, known_face_names)
            
             # Save the result image with recognition to directory, uploads, temporarily
             # so it is saved, rendered, and then shown in img HTML attribute on index.html
             # then save in memory until app restarted
             # NOTE: does not actually save an image with face_recognition data.   This will be empty data, except in rare edge cases

            # Need a unique file path here.
            cv2.imwrite(filepath, processed_frame)


            return render_template('index.html', image_filename=file.filename)
    return render_template('index.html', image_filename=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Displays image that contains all known face encodings"""
    # The images (file) is stored inside the app folder itself, and this can be sent to client to get.

    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/manage_known_faces', methods=['GET', 'POST'])
def manage_known_faces():
    """Allows uploading images of known individuals and clearing the database."""
    if request.method == 'POST':
        if 'add_person' in request.form:
            name = request.form['person_name']
            person_dir = os.path.join(KNOWN_FACES_DIR, name)

             # Check for empty name before creating directory, return and prompt user
            if not name.strip():  # Checks for empty or whitespace-only names

                # return redirect(url_for('manage_known_faces'))
                # OR (better), set an error message and pass to template
                return render_template('manage.html', error='Person name cannot be empty!')
            
            if not os.path.exists(person_dir):

               # Create person's dir in known_faces and load and update known_face_encodings list and ENCODING_FILES file
                os.makedirs(person_dir)  # Changed to use known_faces as root for new image location

            if 'file' not in request.files:
                return 'No file part'
            files = request.files.getlist('file')  # For multiple file uploads.

            # Multiple file checking loop through array and test file extension validity and if it exist (isn't an empty entry, see manage.html for details).

            # The if empty extension and allowed file extension
            for file in files:
                if file.filename != '' and allowed_file(file.filename):
                  filepath = os.path.join(person_dir, file.filename)

                  file.save(filepath)  # Save to known_faces

                  print(f"Uploaded File {filepath} success.")
                  
                elif not allowed_file(file.filename):

                    return render_template('manage.html', error=f"Warning: Not a supported file type: {file.filename}. Only .jpg .png .jpeg or .gif image files allowed ")

            # reload known faces
            load_known_faces()
        elif 'clear_database' in request.form:  # Check if the "Clear Database" button was clicked.
            # clear encodings

            for name in os.listdir(KNOWN_FACES_DIR):
                person_dir = os.path.join(KNOWN_FACES_DIR, name)

                 #  Remove people folder by walking from child folders backwards with all directory children.
                 # for directory_name in os.walk, loop each path within folder.
                for directory_name, _, filenames in os.walk(person_dir, topdown=False): # Start bottom, recursively deleting parent directory

                    for filename in filenames:  # Loop each file and remove each in loop

                       os.remove(os.path.join(directory_name, filename))
                    
                    os.rmdir(directory_name)
                print(f"Deleting all contents in known_faces and folder known_faces at {person_dir}")

            os.remove(ENCODINGS_FILE)
           
            known_face_encodings = []  # empty
            pass  # No need to reinitialize known_face_encodings and known_face_names here

    known_people = os.listdir(KNOWN_FACES_DIR) if os.path.exists(KNOWN_FACES_DIR) else []  # Get list for display

    #Pass "Error"
    error = request.args.get('error', '')

    return render_template('manage.html', known_people=known_people, error=error)



@app.route('/webcam_feed')
def webcam_feed():

    """Provides a continuous stream of processed webcam frames for the browser."""

    # loading face images
    known_face_encodings, known_face_names = load_known_faces() # Load the face images


    return Response(gen_frames(known_face_encodings, known_face_names),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames(known_face_encodings, known_face_names):

   """Gets webcam input and handles encoding stream, using recognize_faces helper function and image encoding.

    """
   # Initialize some variables
   camera = cv2.VideoCapture(0)  # Use the default camera (webcam index 0)


   while True:

       # Get camera images
       success, frame = camera.read() # Read the webcam image (each loop)
       if not success:
         break  #  if cannot access the webcam feed, break


       # Call face recogniton using known images, draw bounding box and get recognized faces (name of person from trained face_encoding database.)

       frame = recognize_faces(frame, known_face_encodings, known_face_names)


       # frame in jpg encode
       ret, buffer = cv2.imencode('.jpg', frame) #Convert each processed frame to JPEG format

       _, buffer = cv2.imencode('.jpg', frame) #Convert each processed frame to JPEG format
       yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yields one frame of video for HTTP stream

   camera.release()  # Release resources. Important for proper cleanup.



# API route:

@app.route('/recognize_api', methods=['POST'])

def recognize_api():

   """API endpoint for face recognition.
       Expects an image in the request body and return known people JSON objects.

   """
   if 'file' not in request.files:
       return jsonify({'error': 'No file provided'}), 400

   file = request.files['file']

   if file.filename == '' or not allowed_file(file.filename):
       return jsonify({'error': 'Invalid file'}), 400

   image = face_recognition.load_image_file(file.stream) # Load from request data


   #Get all known people and get all the filepaths inside their individual, unique named file paths
   known_face_encodings, known_face_names = load_known_faces()  # Ensure these are loaded


    #  Find all face encodings within the image using face_recognition's
    # batch_face_locations: find all the locations of faces and use all these positions
    # to get face_encodings
   face_locations = face_recognition.batch_face_locations([image], number_of_times_to_upsample=0, batch_size=128)

    # Find the face encodings in the current image.

   face_encodings = [ face_encoding
                      for a in face_recognition.face_encodings(
                        image,
                        known_face_locations=face_locations,
                        model=MODEL,
                        num_jitters=1
                    )
                    for face_encoding in a]  # More jitters => more accurate (but slower)
                        # Find encodings

   # Loop through faces
   recognized_names = []
   for face_encoding in face_encodings:

       matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
       name = "Unknown"


       # Compare distances of all matches
       face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
       if face_distances.size > 0:  # Avoid error with empty distances
          best_match_index = np.argmin(face_distances)  # Find best match

          # Check if it is above minimum threshold
          if matches[best_match_index]:

             name = known_face_names[best_match_index] # get name
       recognized_names.append(name) # return list of names of found names in provided picture.



   if recognized_names:
        return jsonify({'names': recognized_names}), 200 # Success: Recognized
   else:
        return jsonify({'names': ['Unknown']}), 200 # No face recognized, 200 as technically not a failure


# -- Create Directories ----
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

if __name__ == '__main__':
    app.run(debug=True) # Start the Flask development server