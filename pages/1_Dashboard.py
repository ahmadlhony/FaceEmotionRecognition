import os
import pandas as pd
import streamlit as st
from PIL import Image

from utility.sqlite_db import FaceEmotionDB

db = FaceEmotionDB()

# Function to load images from a directory
# Function to load images and emotion data from the database
def load_images_and_emotions_from_db():
    emotion_data = db.fetch_all_emotion_data()
    images = []
    image_names = []
    person_names = []
    emotions = []
    confidences = []

    for name, emotion, angry_conf, fear_conf, neutral_conf, sad_conf, disgust_conf, happy_conf, surprise_conf, image_path in emotion_data:
        try:
            images.append(Image.open(os.path.join('emotions', image_path)))
            image_names.append(image_path)
            person_names.append(name)
            emotions.append(emotion)
            confidences.append({
                'Angry': angry_conf,
                'Fear': fear_conf,
                'Neutral': neutral_conf,
                'Sad': sad_conf,
                'Disgust': disgust_conf,
                'Happy': happy_conf,
                'Surprise': surprise_conf
            })
        except IOError:
            continue  # Skip if the image file is not found

    return images, image_names, person_names, emotions, confidences


# Function to filter data based on emotion and person-name
def filter_data(data, emotion_filter, person_name_filter):
    filtered_data = data

    if emotion_filter:
        filtered_data = filtered_data[filtered_data['Emotion'] == emotion_filter]

    if person_name_filter:
        filtered_data = filtered_data[filtered_data['Person Name'] == person_name_filter]

    return filtered_data


# Function to display the data table and images
def display_data_table(data, page_size=3):
    # Pagination
    items_per_page = page_size
    n_data = len(data)
    n_pages = (n_data - 1) // items_per_page + 1
    page = st.number_input(label="Page", min_value=1, max_value=n_pages, value=1, step=1)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    # Display images and data in the table
    for index, row in data.iloc[start_idx:end_idx].iterrows():
        # Create two columns: one for data and one for the image
        col1, col2 = st.columns([3, 1])

        # Display data in the first column
        data_table = pd.DataFrame({
            'Person Name': [row['Person Name']],
            'Emotion': [row['Emotion']],
            **{k: [v] for k, v in row['Confidences'].items()}  # Expand confidence scores
        })
        col1.table(data_table)

        # Display larger image in the second column
        col2.image(row['Image'], caption=row['Image Name'], use_column_width=True)


st.title("Emotions Data Table")

# Load images and emotions data
images, image_names, person_names, emotions, confidences = load_images_and_emotions_from_db()
data = pd.DataFrame({'Image': images, 'Image Name': image_names, 'Person Name': person_names, 'Emotion': emotions, 'Confidences': confidences})

# Filter options
emotion_filter = st.sidebar.selectbox('Filter by Emotion', ['', *data['Emotion'].unique()])
person_name_filter = st.sidebar.selectbox('Filter by Person Name', ['', *data['Person Name'].unique()])

# Filter the data based on user input
filtered_data = filter_data(data, emotion_filter, person_name_filter)

# Display the data table and images
if filtered_data.empty:
    st.warning("No data found with the given filters.")
else:
    display_data_table(filtered_data)

# Close the database connection
db.close()