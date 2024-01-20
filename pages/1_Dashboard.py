import os
import pandas as pd
import streamlit as st
from PIL import Image


# Function to load images from a directory
def load_images(directory):
    images = []
    image_names = []

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            images.append(Image.open(os.path.join(directory, filename)))
            image_names.append(filename)

    return images, image_names


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

    # Display images in the table
    for index, row in data.iloc[start_idx:end_idx].iterrows():
        # Create two columns: one for data and one for the image
        col1, col2 = st.columns([1, 1])

        # Display data in the first column
        col1.table(row[[ 'Person Name', 'Emotion']].to_frame().T)

        # Display larger image in the second column when a row is clicked
        col2.image(row['Image'], caption=row['Image Name'], use_column_width=True)


st.title("Emotions Data Table")

# Load images and create a DataFrame
images, image_names = load_images('emotions')
data = pd.DataFrame({'Image': images, 'Image Name': image_names})

# Extract emotion and person-name from image names
data['Emotion'] = data['Image Name'].apply(lambda x: x.split('_')[2].split('.')[0])
data['Person Name'] = data['Image Name'].apply(lambda x: x.split('_')[1])

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
