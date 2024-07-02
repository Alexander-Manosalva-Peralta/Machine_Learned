import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings

# Configuración de la interfaz de Streamlit
st.set_page_config(page_title="Sistema de Detección de Retinopatía Diabética", page_icon=":eye:", layout="wide")
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
    st.image("retinopatia.jpg", caption="Imagen ilustrativa de retina con retinopatía", use_column_width=True)

elif choice == "Detección de Retinopatía":
    st.subheader("Detección de Retinopatía Diabética")
    st.write("Sube una imagen del ojo (JPG o PNG) o captura una foto desde la cámara para detectar si tiene retinopatía diabética.")

    # Subir archivo de imagen o capturar desde la cámara
    capture_option = st.radio("Selecciona una opción:", ("Subir imagen", "Tomar foto con la cámara"))

    if capture_option == "Subir imagen":
        uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen subida', use_column_width=True)
            st.write("")
            st.write("Procesando la imagen...")
            preprocessed_image = preprocess_image(image)
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                prediction = model.predict(preprocessed_image)
                score = prediction[0][0]
                if score > 0.5:
                    st.success(f"Resultado: Retinopatía diabética detectada con una confianza del {score*100:.2f}%.")
                    st.image("retinopatia_futurista.jpg", caption="Ejemplo de retina con retinopatía", use_column_width=True)
                    st.write("Esta imagen muestra una visión futurista de cómo la inteligencia artificial ayuda a detectar problemas de salud visual con precisión.")
                else:
                    st.info(f"Resultado: No se detectó retinopatía diabética con una confianza del {(1-score)*100:.2f}%.")
                    st.image("ojo_sano_futurista.jpg", caption="Ejemplo de retina sana", use_column_width=True)
                    st.write("Esta imagen representa un futuro donde la tecnología puede confirmar rápidamente la salud ocular sin necesidad de intervención médica.")
            else:
                st.warning("El modelo no está disponible. Por favor, asegúrate de que el archivo 'model.h5' está en el directorio correcto.")

    elif capture_option == "Tomar foto con la cámara":
        st.write("Presiona el botón para iniciar la cámara:")
        webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=None)
        if webrtc_ctx.video_processor:
            st.subheader("Vista previa de la cámara")
            st.write("Haz clic en 'Tomar foto' cuando estés listo.")
            if st.button("Tomar foto"):
                image = webrtc_ctx.video_processor.get_image()
                st.image(image, caption='Foto tomada desde la cámara', use_column_width=True)
                st.write("")
                st.write("Procesando la imagen...")
                preprocessed_image = preprocess_image(image)
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path)
                    prediction = model.predict(preprocessed_image)
                    score = prediction[0][0]
                    if score > 0.5:
                        st.success(f"Resultado: Retinopatía diabética detectada con una confianza del {score*100:.2f}%.")
                        st.image("retinopatia_futurista.jpg", caption="Ejemplo de retina con retinopatía", use_column_width=True)
                        st.write("Esta imagen muestra una visión futurista de cómo la inteligencia artificial ayuda a detectar problemas de salud visual con precisión.")
                    else:
                        st.info(f"Resultado: No se detectó retinopatía diabética con una confianza del {(1-score)*100:.2f}%.")
                        st.image("ojo_sano_futurista.jpg", caption="Ejemplo de retina sana", use_column_width=True)
                        st.write("Esta imagen representa un futuro donde la tecnología puede confirmar rápidamente la salud ocular sin necesidad de intervención médica.")
                else:
                    st.warning("El modelo no está disponible. Por favor, asegúrate de que el archivo 'model.h5' está en el directorio correcto.")

elif choice == "Equipo":
    st.subheader("Equipo")
    st.write("Este proyecto ha sido desarrollado por un equipo multidisciplinario comprometido con la salud visual y la inteligencia artificial.")
    st.sidebar.subheader("Integrantes del Equipo")
    st.sidebar.write("Lila Zaray Huanca Ampuero")
    st.sidebar.write("Yojan Alexander Manosalva Peralta")
    st.sidebar.write("Michael Gavino Isidro")
    st.sidebar.write("Sebastián Marcelo Pacheco Vidalon")

elif choice == "Acerca de":
    st.subheader("Acerca de")
    st.write("Esta aplicación utiliza un modelo de aprendizaje profundo para detectar retinopatía diabética en imágenes del fondo de ojo. La integración con Streamlit proporciona una interfaz accesible y educativa para los usuarios.")
    st.write("Desarrollado utilizando TensorFlow para el aprendizaje profundo y Streamlit para la visualización web.")
    st.image("diabetic_retinopathy_futurista.jpg", caption="Ilustración futurista de retinopatía diabética", use_column_width=True)
    st.write("En un futuro cercano, herramientas como esta podrían ser comunes en consultorios médicos y hogares, mejorando el cuidado de la salud ocular globalmente.")
