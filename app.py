import os
import cv2
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pandas as pd
from playsound import playsound  # Import the playsound library

# Initialize Flask app
app = Flask(__name__)

# Set date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initialize the face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create required directories if they don't exist
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Check available cameras
def get_camera_index():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    raise Exception("No available camera found.")

camera_index = get_camera_index()

# Function to count registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract faces from an image
def extract_faces(img):
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        return faces
    return []

# Load or train model
def load_or_train_model():
    model_path = 'static/face_recognition_cnn_model.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        model.compile()  # Ensure model is compiled if needed
        return model
    else:
        model = train_model()
        model.save(model_path)
        return model

# CNN model training function
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    label_map = {user: idx for idx, user in enumerate(userlist)}

    for user, label in label_map.items():
        user_folder = os.path.join('static/faces', user)
        if os.path.isdir(user_folder):
            for imgname in os.listdir(user_folder):
                img = cv2.imread(os.path.join(user_folder, imgname))
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face)
                labels.append(label)

    faces = np.array(faces).astype('float32') / 255.0
    labels = to_categorical(labels, num_classes=len(userlist))

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(userlist), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(faces, labels, epochs=10, batch_size=8)
    return model

# Identify face using CNN model
def identify_face(facearray):
    model = load_or_train_model()
    predictions = model.predict(facearray)
    return np.argmax(predictions, axis=1)

# Extract attendance data
def extract_attendance():
    try:
        # Load the attendance file
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

        # Ensure required columns exist
        if 'Time' not in df.columns:
            df['Time'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(len(df))]
        if 'Name' not in df.columns or 'Roll' not in df.columns:
            raise KeyError("Required columns 'Name' or 'Roll' are missing.")

        # Return required data
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        l = len(df)
        return names, rolls, times, l

    except FileNotFoundError:
        print("Error: 'attendance.csv' file not found.")
        return [], [], [], 0
    except KeyError as e:
        print(f"KeyError: {e}")
        return [], [], [], 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], [], 0

# Add attendance record
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        playsound('static/sounds/thank_you.mp3')  # Play sound after recording attendance

# Main page route
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Route for attendance taking
@app.route('/start', methods=['GET'])
def start():
    try:
        model = load_or_train_model()
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise Exception("Could not open video device")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Could not read frame.")
                break

            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50)).reshape(1, 50, 50, 3) / 255.0
                identified_person_index = identify_face(face)

                if identified_person_index is not None:
                    userlist = os.listdir('static/faces')
                    if len(userlist) > identified_person_index[0]:
                        identified_person = userlist[identified_person_index[0]]
                        add_attendance(identified_person)
                        cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)
    
    except Exception as e:
        print("Error:", e)
        return render_template('error.html', message=str(e))

# Route for adding a new user
@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(camera_index)
    i = 0
    while i < 200:  # Changed from 50 to 200 images
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            cv2.imwrite(f'{userimagefolder}/{i}.jpg', face)
            i += 1
            if i >= 200:  # Stop after capturing 200 images
                break

        cv2.imshow("Capture Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Retrain the model after adding a new user
    model = train_model()
    model.save('static/face_recognition_cnn_model.h5')

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
