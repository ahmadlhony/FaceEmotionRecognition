import random

from deepface import DeepFace
import streamlit as st
from PIL import Image
import os

import face_recognition
import cv2
import numpy as np

from utility.sqlite_db import FaceEmotionDB

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

st.title("Face Recognition App")
uploaded_images = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
if uploaded_images:
    img = Image.open(uploaded_images)
    new_image = np.array(img, dtype=np.uint8)

    face_locations = face_recognition.face_locations(new_image)
    face_encodings = face_recognition.face_encodings(new_image, face_locations)
    y1, x2, y2, x1 = face_locations[0]
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"
    fid = -1

    if known_face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            fid = known_face_ids[best_match_index]

    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    face_roi = new_image_bgr[y1:y2, x1:x2]
    emotions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    emotion_label = emotions[0]['dominant_emotion']
    # emotion_label = ''
    if name == 'Unknown':
        col_name, col_add = st.columns([4, 1])
        name = col_name.text_input("Name")
        add = col_add.button("Add")
        st.image(new_image, caption='Face')
        col_emotion, col_add_emotion = st.columns([4, 1])
        col_emotion.write("Emotion: " + emotion_label)
        if add:
            image_id = random.randint(0, 999999)
            image_name = str(image_id) + '_' + str(name.strip()).replace(' ', '-') + '.jpg'
            cv2.imwrite(os.path.join('faces', image_name), new_image_bgr)  # Save image in 'faces' directory
            face_id = db.insert_face({
                'name': name,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'image_path': image_name  # Saving only image name, not the full path
            })
            fid = face_id
            st.session_state.id = face_id
            st.rerun()
    else:
        col_name, col_add = st.columns([4, 1])
        col_name.write("Name: " + name)
        st.image(new_image, caption='Face')
        col_emotion, col_add_emotion = st.columns([4, 1])
        col_emotion.write("Emotion: " + emotion_label)
        add_emotion = col_add_emotion.button("Save emotion for this person")
        if fid == -1:
            fid = int(st.session_state.id)
        if add_emotion:
            emotion_confidences = emotions[0]['emotion']
            image_id = random.randint(0, 999999)
            image_name = str(image_id) + '_' + str(name.strip()).replace(' ', '-') + '_' + str(emotion_label) + '.jpg'
            image_path = os.path.join('emotions', image_name)
            cv2.imwrite(image_path, new_image_bgr)
            db.insert_emotion({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'image_path': image_name,
                'face_id': fid,
                'angry_conf': int(emotion_confidences['angry']),
                'disgust_conf': int(emotion_confidences['disgust']),
                'fear_conf': int(emotion_confidences['fear']),
                'happy_conf': int(emotion_confidences['happy']),
                'sad_conf': int(emotion_confidences['sad']),
                'surprise_conf': int(emotion_confidences['surprise']),
                'neutral_conf': int(emotion_confidences['neutral']),
                'dominant_emotion': emotion_label,
            })
            uploaded_images = []
            st.rerun()

# print(uploaded_images)
# image = cv2.imread(uploaded_images.name)

# print(image.shape)
