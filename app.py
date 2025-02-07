import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings

# Configuración de la interfaz de Streamlit
st.set_page_config(
    page_title="Detección de Retinopatía Diabética",
    page_icon=":eye:",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("Sistema de Detección de Retinopatía Diabética")

# Ruta al modelo preentrenado
model_path = 'model.h5'

# Función para preprocesar la imagen
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Crear menú en la barra lateral
menu = ["Inicio", "Detección de Retinopatía", "Captura desde Cámara", "Equipo", "Acerca de"]
choice = st.sidebar.selectbox("Menú", menu)

if choice == "Inicio":
    st.subheader("Inicio")
    st.write("Bienvenido a la aplicación de detección de retinopatía diabética. Utiliza el menú de la barra lateral para navegar entre las secciones.")
    st.image("retinopatia.jpg")

elif choice == "Detección de Retinopatía":
    st.subheader("Detección de Retinopatía Diabética")
    st.write("Sube una imagen del ojo (JPG o PNG) para detectar si tiene retinopatía diabética.")

    # Verificar si el archivo del modelo existe
    model_exists = os.path.exists(model_path)

    if model_exists:
        model = tf.keras.models.load_model(model_path)

    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', use_column_width=True)
        
        st.write("")
        st.write("Procesando la imagen...")
        
        # Preprocesar la imagen
        preprocessed_image = preprocess_image(image)
        
        if model_exists:
            # Realizar la predicción
            prediction = model.predict(preprocessed_image)
            score = prediction[0][0]
            
            # Mostrar el resultado
            if score > 0.5:
                st.write(f"**Resultado:** Retinopatía diabética detectada con una confianza del {score*100:.2f}%.")
            else:
                st.write(f"**Resultado:** No se detectó retinopatía diabética con una confianza del {(1-score)*100:.2f}%.")

elif choice == "Captura desde Cámara":
    st.subheader("Captura desde Cámara")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            super().__init__()

        def recv(self, frame: np.ndarray) -> np.ndarray:
            return frame

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        async_processing=True,
    )

elif choice == "Equipo":
    st.subheader("Equipo")
    st.sidebar.header("Integrantes del Equipo")
    st.sidebar.write("Lila Zaray Huanca Ampuero")
    st.sidebar.write("Manosalva Peralta Yojan Alexander")
    st.sidebar.write("Michael Gavino Isidro")
    st.sidebar.write("Pacheco Vidalon Sebastián Marcelo")
    
elif choice == "Acerca de":
    st.subheader("Acerca de")
    st.write("Esta es una aplicación de detección de retinopatía diabética desarrollada utilizando Streamlit y TensorFlow.")
