#STEP 1 : COLLECTING THE DATA

import cv2  # Import OpenCV for computer vision tasks
import pickle  # Import pickle to save and load serialized data
import numpy as np  # Import NumPy for numerical operations
import os  # Import os to interact with the file system

# Capture video from the default camera (0 represents the default webcam)
video = cv2.VideoCapture(0)

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier('data/frontalface_haarcascade.xml')

# List to store captured face images
faces_data = []

# Counter variable to control data collection
i = 0

# Ask the user to enter their name
name = input("Enter Your Name: ")

while True:  # Infinite loop to continuously capture video frames
    ret, frame = video.read()  # Read a frame from the camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    
    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:  # Loop through detected faces
        crop_img = frame[y:y+h, x:x+w, :]  # Crop the detected face region
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize the face image to 50x50 pixels
        
        # Store only up to 100 face images, capturing one every 10 frames
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)  # Append the resized face image to the list
        
        i = i + 1  # Increment the counter
        
        # Display the number of collected face images on the video frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (50, 50, 255), 1)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
    
    # Show the frame with detected faces
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)  # Wait for a key press for 1 millisecond
    if k == ord('q') or len(faces_data) == 100:  # Exit loop if 'q' is pressed or 100 images are collected
        break

# Release the webcam and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert the list of face images to a NumPy array
faces_data = np.asarray(faces_data)

# Reshape the array to store 100 face images in a flattened format
faces_data = faces_data.reshape(100, -1)

# Check if 'all_names.pkl' file exists in the 'data/' directory
if 'all_names.pkl' not in os.listdir('data/'):
    names = [name] * 100  # Create a list of 100 entries with the user's name
    with open('data/all_names.pkl', 'wb') as f:
        pickle.dump(names, f)  # Save the list to a pickle file
else:
    with open('data/all_names.pkl', 'rb') as f:
        names = pickle.load(f)  # Load existing names from the file
    
    names = names + [name] * 100  # Append the new user's name 100 times
    
    with open('data/all_names.pkl', 'wb') as f:
        pickle.dump(names, f)  # Save the updated list

# Check if 'all_faces_data.pkl' file exists in the 'data/' directory
if 'all_faces_data.pkl' not in os.listdir('data/'):
    with open('data/all_faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)  # Save the face data if the file does not exist
else:
    with open('data/all_faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)  # Load existing face data
    
    # Append new face data to the existing dataset
    faces = np.append(faces, faces_data, axis=0)
    
    with open('data/all_faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)  # Save the updated face dataset
