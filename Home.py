import random

from deepface import DeepFace
import streamlit as st
from PIL import Image
import os

import face_recognition
import cv2
import numpy as np

# Get a reference to webcam #0 (the default one)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

face_paths = 'faces'
emotions_paths = 'emotions'

for i in os.listdir(face_paths):
    face_path = os.path.join(face_paths, i)
    image = face_recognition.load_image_file(face_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    name = i.split('.')[0]
    known_face_names.append(name)

st.title("Face Recognition App")
uploaded_images = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
if uploaded_images:
    img = Image.open(uploaded_images)
    new_image = np.array(img, dtype=np.uint8)

    face_locations = face_recognition.face_locations(new_image)
    face_encodings = face_recognition.face_encodings(new_image, face_locations)
    y1, x2, y2, x1 = face_locations[0]
    print(face_locations)
    # (top, right, bottom, left), face_encoding = zip(face_locations[0], face_encodings[0])
    face_encoding = face_encodings[0]
    print('face_encoding')
    print(face_encoding)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
    print('matches')
    print(matches)

    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        print(known_face_names[best_match_index])
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    face_roi = new_image_bgr[y1:y2, x1:x2]
    emotions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    print(emotions)
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
            image_name = str(name.strip()).replace(' ', '-') + '.jpg'
            image_path = os.path.join(face_paths, image_name)
            cv2.imwrite(image_path, new_image_bgr)
            name = str(name.strip())
    else:
        col_name, col_add = st.columns([4, 1])
        col_name.write("Name: " + name)
        st.image(new_image, caption='Face')
        col_emotion, col_add_emotion = st.columns([4, 1])
        col_emotion.write("Emotion: " + emotion_label)
        add_emotion = col_add_emotion.button("Save emotion for this person")
        if add_emotion:
            image_id = random.randint(0, 999999)
            image_name = str(image_id) + '_' + str(name.strip()).replace(' ', '-')+ '_' + str(emotion_label) + '.jpg'
            image_path = os.path.join(emotions_paths, image_name)
            cv2.imwrite(image_path, new_image_bgr)
            uploaded_images = []

# print(uploaded_images)
# image = cv2.imread(uploaded_images.name)

# print(image.shape)
