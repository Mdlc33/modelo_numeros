import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo
model = load_model('numerosCNN.keras')

# Crear interfaz de usuario
st.title('Clasificador de dígitos MNIST')
st.write('Sube una imagen para clasificarla como un dígito del 0 al 9.')

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen en escala de grises de 28x28 píxeles.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Procesar la imagen
        image = Image.open(uploaded_file).convert('L')  # Convertir a blanco y negro
        image = image.resize((28, 28))  # Redimensionar la imagen
        img_array = np.array(image)  # Convertir la imagen a un array
        img_array = img_array / 255.0  # Normalizar la imagen

        # Aplanar la imagen: 1 imagen, 784 dimensiones
        img_array = img_array.reshape(1, 784)

        # Mostrar la imagen subida
        st.image(image, caption='Imagen subida', use_column_width=True)

        # Predicción
        prediction = model.predict(img_array)

        # Clases para los dígitos del 0 al 9
        classes = [str(i) for i in range(10)]

        st.write('Predicción:', classes[np.argmax(prediction)])

        # Mostrar probabilidades
        for i, prob in enumerate(prediction[0]):
            st.write(f"{classes[i]}: {prob:.2%}")

        # Clase con mayor probabilidad
        st.write("Predicción:", classes[np.argmax(prediction)])

    except ValueError as e:
        st.write(f"Error al hacer la predicción: {e}")
