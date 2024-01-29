import os
import random

import streamlit as st
from deepface import DeepFace
import face_recognition
import cv2
import numpy as np
from utility.sqlite_db import FaceEmotionDB

video_capture = cv2.VideoCapture(0)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
db = FaceEmotionDB()
known_face_encodings = []
known_face_names = []
known_face_ids = []

face_data = db.fetch_all_face_data()

face_paths = 'faces'
emotions_paths = 'emotions'

for face_id, name, face_path in face_data:
    full_path = os.path.join('faces', face_path)
    if os.path.isfile(full_path):
        image = face_recognition.load_image_file(full_path)
        # Check if at least one face is found in the image
        if face_recognition.face_encodings(image):
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            known_face_ids.append(face_id)

print('FRAME')
temp_emotion_label = ''
temp_face_id = ''
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)

        name = "Unknown"
        face_id = -1
        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            face_id = known_face_ids[best_match_index]
        else:
            image_id = random.randint(0, 999999)
            image_name = str(image_id) + '_' + name + '.jpg'
            cv2.imwrite(os.path.join('faces', image_name), frame)
            face_id = db.insert_face_unknown({
                'x1': right,
                'y1': top,
                'x2': left,
                'y2': bottom,
                'image_path': image_name
            })
            name = f'unknown_{face_id}'
            known_face_encodings.append(face_encoding)
            known_face_ids.append(face_id)
            known_face_names.append(name)

        # emotion
        face_roi = frame[top:bottom, left:right]
        emotions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        print(emotions)
        emotion_label = emotions[0]['dominant_emotion']
        if emotion_label != temp_emotion_label or face_id != temp_face_id:
            emotion_confidences = emotions[0]['emotion']
            image_id = random.randint(0, 999999)
            image_name = str(image_id) + '_' + str(name).replace(' ', '-') + '_' + str(emotion_label) + '.jpg'
            image_path = os.path.join('emotions', image_name)
            cv2.imwrite(image_path, frame)
            db.insert_emotion({
                'x1': right,
                'y1': top,
                'x2': left,
                'y2': bottom,
                'image_path': image_name,
                'face_id': face_id,
                'angry_conf': int(emotion_confidences['angry']),
                'disgust_conf': int(emotion_confidences['disgust']),
                'fear_conf': int(emotion_confidences['fear']),
                'happy_conf': int(emotion_confidences['happy']),
                'sad_conf': int(emotion_confidences['sad']),
                'surprise_conf': int(emotion_confidences['surprise']),
                'neutral_conf': int(emotion_confidences['neutral']),
                'dominant_emotion': emotion_label,
            })
        temp_face_id = face_id
        temp_emotion_label = emotion_label

        cv2.putText(frame, emotion_label, (right, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
