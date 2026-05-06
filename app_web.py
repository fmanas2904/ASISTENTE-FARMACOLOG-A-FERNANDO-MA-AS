import streamlit as st
import os
from groq import Groq

st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Error en la configuración de la clave.")
    st.stop()

# --- SELECCIÓN DE GUÍA ---
directorio = "documentos"
archivos = sorted([f for f in os.listdir(directorio) if f.lower().endswith(".txt")])

if archivos:
    guia_seleccionada = st.selectbox("📚 Seleccioná la guía para consultar:", archivos)
    
    # Leer solo la guía elegida
    try:
        with open(os.path.join(directorio, guia_seleccionada), "r", encoding="utf-8", errors="ignore") as f:
            contenido_guia = f.read()
        st.success(f"✅ Consultando: {guia_seleccionada}")
    except:
        st.error("No se pudo leer el archivo.")
        contenido_guia = None
else:
    st.warning("No hay archivos .txt en la carpeta documentos.")
    contenido_guia = None

# --- CONSULTA ---
pregunta = st.text_input("Escribí tu duda técnica:")

if pregunta and contenido_guia:
    with st.spinner("El profesor está respondiendo..."):
        try:
            # Enviamos solo el contenido de UNA guía. Esto es ultra rápido.
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Sos Profesor de Farmacología. Respondé usando solo la guía provista. Si no está ahí, decilo."},
                    {"role": "user", "content": f"GUÍA: {contenido_guia}\n\nPREGUNTA: {pregunta}"}
                ],
                temperature=0.0
            )
            st.subheader("📌 Respuesta:")
            st.write(res.choices[0].message.content)
        except:
            st.error("Hubo un error de conexión. Probá de nuevo.")
