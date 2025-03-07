# Facial Recognition-Based Attendance System

![Project Banner](background.png)

## 📌 Project Overview

The **Facial Recognition-Based Attendance System** is a machine learning-powered solution designed to automate attendance tracking using facial recognition. It utilizes **OpenCV** for image processing, **scikit-learn** for face classification with **K-Nearest Neighbors (KNN)**, and **Streamlit** for a web-based attendance dashboard.

## 🚀 Features

- **Face Detection & Recognition**: Uses OpenCV's Haar cascade classifiers for face detection and KNN for recognition.
- **Automated Attendance Logging**: Saves attendance records with timestamps in CSV format.
- **Text-to-Speech Alerts**: Uses Windows SAPI to announce attendance confirmation.
- **Real-Time Processing**: Captures faces using a webcam and updates attendance instantly.
- **Web-Based Dashboard**: Streamlit-powered interface to view and manage attendance logs.

## 📂 Project Structure

```
📦 Facial-Recognition-Attendance-System
├── 📂 data                 # Stores trained face data and labels
│   ├── all_faces_data.pkl  # Encoded face embeddings
│   ├── all_names.pkl       # Names associated with embeddings
│   ├── frontalface_haarcascade.xml  # Haar cascade model for face detection
├── 📂 Attendance           # Stores CSV attendance logs
│   ├── Attendance_07-03-2025.csv  # Example attendance file
├── adding_faces.py         # Script to add new faces to the system
├── testing.py              # Facial recognition and attendance tracking logic
├── app.py                  # Streamlit-based web application for attendance logs
├── requirements.txt        # Required Python libraries
├── background.png          # Background image for UI
├── README.md               # Project documentation
```

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Facial-Recognition-Attendance-System.git
cd Facial-Recognition-Attendance-System
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add New Faces
Run the script to register new faces:
```bash
python adding_faces.py
```
Follow the instructions to capture face data.

### 4️⃣ Run the Attendance System
```bash
python testing.py
```
Press **"O"** to take attendance.

### 5️⃣ View Attendance on Web Dashboard
```bash
streamlit run app.py
```
This will launch a web app showing attendance records.

## 📸 Working Demo

1. The system detects and recognizes faces in real-time.
2. It logs attendance with a timestamp when a registered face is detected.
3. A voice announcement confirms attendance.
4. The attendance log is stored in a CSV file and displayed on the web app.

## 🔧 Technologies Used

- **Python**
- **OpenCV** (Face detection)
- **scikit-learn** (KNN for classification)
- **Streamlit** (Web dashboard)
- **NumPy, Pandas** (Data processing)
- **Windows SAPI** (Text-to-Speech)

## 📜 License

This project is open-source and available under the **MIT License**.
