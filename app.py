from flask import Flask, render_template, url_for, Response, redirect, request, session
from utils import get_face_landmarks
import openai
import pickle
import cv2
import os
import time
import sqlite3
import cv2
import keras
import tensorflow as tf
from deepface import DeepFace
print("keras version:", keras.__version__)
print("tensorflow version:", tf.__version__)

emotions = ['HAPPY', 'SAD', 'SURPRISED']

app = Flask(__name__)

cap = cv2.VideoCapture(0)

# Define the path to the reference image
reference_image_path = "reference.jpg"

# Load the reference image
reference_image = cv2.imread(reference_image_path)

with open('./model', 'rb') as f:
    model = pickle.load(f)


# Routes
@app.route('/')
def hello_world():
    image_names = ['robot1.jpg', 'robot4.jpg', 'robot10.jpg',
                   'robot16.jpg', 'robot4.jpg', 'robot10.jpg',
                   'robot1.jpg', 'robot4.jpg', 'robot10.jpg']
    return render_template('Index.html', image_names=image_names, user="guest, please log in!")


@app.route('/video_feed')
def video_feed():
    global cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    generate_emotions()
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam')
def webcam_route():
    return render_template('webcam.html')


@app.route('/home')
def home_route():
    if 'user' in session:
        user = session["user"]
    else:
        user = "guest, please log in!"
    return render_template('index.html', user=user)


@app.route('/stop-video', methods=['POST'])
def stop_video():
    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('hello_world'))


# Logic

# Streams webcam input on website, not vital for robot


# Define a function to perform face recognition
def check_face_match(frame, reference_image):
    try:
        # Resize reference image to match webcam frame size
        reference_image_resized = cv2.resize(reference_image, (frame.shape[1], frame.shape[0]))

        # Perform face recognition
        result = DeepFace.verify(reference_image_resized, frame)

        # Extract the result of face recognition
        verified = result["verified"]

        return verified
    except Exception as e:
        print("Error:", str(e))
        return False


# Modify the generate_frames function to include face recognition
def generate_frames():
    while True:
        check, frame = cap.read()
        if not check:
            break
        else:
            # Check for face landmarks and predict emotion
            face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
            if len(face_landmarks) > 0:
                output = model.predict([face_landmarks])
            else:
                continue

            # Perform face recognition
            face_matched = check_face_match(frame, reference_image)

            # Display emotion and face match status
            cv2.putText(frame,
                        emotions[int(output[0])],
                        (10, frame.shape[0] - 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0, 255, 0),
                        5)

            match_text = "Face Matched" if face_matched else "Face Not Matched"
            cv2.putText(frame,
                        match_text,
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255) if not face_matched else (0, 255, 0),
                        2)

            # Encode frame and yield
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Wait 10 seconds, save an emotion, and then send it to the chat bot to initiate a convo
def generate_emotions():
    emotiondetected = True
    time.sleep(10)
    while emotiondetected:
        check, frame = cap.read()
        if not check:
            break
        else:
            face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
            if len(face_landmarks) > 0:
                output = model.predict([face_landmarks])
                print(emotions[int(output[0])])  # replace with code that stores this in a var and sends to chatbot
                print('emotion detected, breaking process')
                emotiondetected = False
            else:
                continue


# Chatbot is configured for manual input from website right now
# @app.route('/get-answer', methods=['GET', 'POST'])  # Allow both GET and POST methods
# def get_answer():
#     if request.method == 'POST':
#         print('reached')
#         question = request.form['text']
#         if question.lower() == "quit":
#             return "Goodbye!"
#         answer = generate_answer(question)
#         return render_template('webcam.html', sample_output=answer)
#     else:
#         return "Please use a POST request to get an answer."
#
#
# # answer generation given prompt
# def generate_answer(prompt):
#     response = openai.Completion.create(
#         engine="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         max_tokens=100,
#         temperature=0.5
#     )
#
#     answer = response.choices[0].text.strip()
#     return answer


