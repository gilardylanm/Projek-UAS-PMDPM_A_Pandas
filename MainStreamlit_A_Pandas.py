import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model(r'BestModel_MobileNet_Pandas.h5')
class_names = ['Jeruk Lemon', 'Jeruk Nipis', 'Jeruk Sunkist']

def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        return result.numpy()
    except Exception as e:
        return "Error", str(e)

def custom_progress_bar(confidence_scores, class_names, colors):
    for i, score in enumerate(confidence_scores):
        percentage = score * 100
        color = colors[i]
        progress_html = f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px; margin-bottom: 5px;">
            <div style="width: {percentage:.2f}%; background: {color}; color: white; text-align: center; height: 24px;">
                {class_names[i]}: {percentage:.2f}%
            </div>
        </div>
        """
        st.sidebar.markdown(progress_html, unsafe_allow_html=True)

st.title("Prediksi Jenis Buah Jeruk - Pandas")

uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            confidence_scores = classify_image(uploaded_file.name)
            if confidence_scores != "Error":
                class_idx = np.argmax(confidence_scores)
                predicted_label = class_names[class_idx]
                colors = ["#007BFF", "#FFAA00", "#28A745"]
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.write(f"**Prediksi Utama:** {predicted_label} ({confidence_scores[class_idx] * 100:.2f}%)")
                st.sidebar.write("**Confidence Scores:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence_scores[i] * 100:.2f}%")
                custom_progress_bar(confidence_scores, class_names, colors)
                st.sidebar.write(f"**Total Confidence:** {sum(confidence_scores) * 100:.2f}%")
                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence_scores[1]}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
