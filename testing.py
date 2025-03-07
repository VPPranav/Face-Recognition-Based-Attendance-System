# STEP 2 : BUILDING OUR MACHINE LEARNING ALGORITHM FOR FACE RECOGNITION USING KNN CLASSIFIER USING PYTHON LIBRARY SCIKIT-LEARN

# K-Nearest Neighbors (KNN) Classifier:
# KNN is a supervised learning algorithm used for classification and regression.
# It works by finding the 'k' closest data points in the training set and 
# assigning the most common class among them to the new data point.
# In this case, KNN is used to classify detected faces based on stored face data.

from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier from scikit-learn
import cv2  # Import OpenCV for image processing
import pickle  # Import pickle for loading and saving data
import numpy as np  # Import NumPy for numerical operations
import os  # Import os to interact with the file system
import csv  # Import csv to handle attendance records
import time  # Import time module for timestamp generation
from datetime import datetime  # Import datetime for date and time formatting

# Import Windows speech API for text-to-speech functionality
from win32com.client import Dispatch   # type: ignore

def speak(text):  
    """Converts text to speech using Windows SAPI."""
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Capture video from the default camera
video = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier('data/frontalface_haarcascade.xml')

# Ensure the face detection model is loaded correctly
if facedetect.empty():
    print("Error: Could not load Haar cascade model!")
    exit()

# Load stored labels (all_names) from the pickle file
with open('data/all_names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

# Load stored face data from the pickle file
with open('data/all_faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)  # Print the shape of the loaded face data

# Initialize the KNN classifier with k=5 (5 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN classifier using the loaded face data and corresponding labels
knn.fit(FACES, LABELS)

# Load a background image for displaying results
imgBackground = cv2.imread("background.png")

# Ensure the background image is loaded correctly
if imgBackground is None:
    print("Error: Background image not found!")
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)  # Use a black background

# Ensure the camera is working
if not video.isOpened():
    print("Error: Could not open webcam!")
    exit()

# Column names for the attendance CSV file
COL_NAMES = ['NAME', 'TIME']

# Ensure Attendance folder exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

while True:  # Continuous loop to process video frames
    ret, frame = video.read()  # Read a frame from the camera
    if not ret:
        print("Error: Failed to capture frame from camera!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
    
    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:  # Loop through detected faces
        crop_img = frame[y:y+h, x:x+w, :]  # Crop the detected face region
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Resize and flatten
        
        output = knn.predict(resized_img)  # Predict the person's name using KNN
        name = str(output[0])
        
        # Generate timestamp for attendance
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")  # Format the date
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")  # Format the time
        
        attendance_file = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(attendance_file)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(frame, name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    # Display the frame with the background
    if imgBackground is not None:
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", imgBackground)
    else:
        cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    
    if k == ord('o'):  # Press 'o' to take attendance
        speak(f"Attendance taken for {name}")
        with open(attendance_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write header if file does not exist
            writer.writerow([name, timestamp])
        print(f"Attendance taken for {name}")

    if k == ord('q'):  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()