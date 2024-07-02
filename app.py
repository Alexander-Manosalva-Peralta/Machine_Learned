import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Configuración de la interfaz de Streamlit
st.title("Sistema de Detección de Retinopatía Diabética")
st.sidebar.title("Menú")

# Definir ruta al modelo preentrenado
model_path = 'model.h5'

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((256, 256))  # Reajustar tamaño de la imagen
    image = np.array(image) / 255.0  # Normalizar valores de píxeles
    image = np.expand_dims(image, axis=0)  # Agregar dimensión para lote único
    return image

# Opciones del menú lateral
menu_options = ["Inicio", "Detección de Retinopatía", "Equipo", "Acerca de"]
choice = st.sidebar.selectbox("Selecciona una opción", menu_options)

# Manejo de opciones del menú
if choice == "Inicio":
    st.subheader("Inicio")
    st.write("Bienvenido a la aplicación de detección de retinopatía diabética. Utiliza el menú de la barra lateral para navegar entre las secciones.")
    st.image("retinopatia.jpg", caption="Imagen ilustrativa", use_column_width=True)

elif choice == "Detección de Retinopatía":
    st.subheader("Detección de Retinopatía Diabética")
    st.write("Sube una imagen del ojo (JPG o PNG) para detectar si tiene retinopatía diabética.")

    # Subir archivo de imagen
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', use_column_width=True)
        
        st.write("")
        st.write("Procesando la imagen...")
        
        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image)
        
        if os.path.exists(model_path):
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)

            # Realizar la predicción
            prediction = model.predict(preprocessed_image)
            score = prediction[0][0]
            
            # Mostrar el resultado
            if score > 0.5:
                st.success(f"Resultado: Retinopatía diabética detectada con una confianza del {score*100:.2f}%.")
            else:
                st.info(f"Resultado: No se detectó retinopatía diabética con una confianza del {(1-score)*100:.2f}%.")
        else:
            st.warning("El modelo no está disponible. Por favor, asegúrate de que el archivo 'model.h5' está en el directorio correcto.")

elif choice == "Equipo":
    st.subheader("Equipo")
    st.write("Este proyecto ha sido desarrollado por un equipo multidisciplinario dedicado a la detección de retinopatía diabética.")
    st.sidebar.subheader("Integrantes del Equipo")
    st.sidebar.write("Lila Zaray Huanca Ampuero")
    st.sidebar.write("Yojan Alexander Manosalva Peralta")
    st.sidebar.write("Michael Gavino Isidro")
    st.sidebar.write("Sebastián Marcelo Pacheco Vidalon")

elif choice == "Acerca de":
    st.subheader("Acerca de")
    st.write("Esta aplicación utiliza un modelo de aprendizaje profundo para detectar retinopatía diabética en imágenes del fondo de ojo. El modelo está integrado con Streamlit para proporcionar una interfaz de usuario interactiva y accesible.")
    st.write("Desarrollado utilizando TensorFlow para el aprendizaje profundo y Streamlit para la visualización web.")

