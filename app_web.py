import streamlit as st
import os
from groq import Groq

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

# --- CONEXIÓN ---
try:
    # Usamos la clave de los secrets que ya configuraste
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Error: Revisa la clave en los Secrets.")
    st.stop()

# --- CARGA DE GUÍAS (MÉTODO LIVIANO) ---
def leer_guias():
    texto_acumulado = ""
    dir_documentos = "documentos"
    if not os.path.exists(dir_documentos):
        return None
    
    archivos = [f for f in os.listdir(dir_documentos) if f.lower().endswith(".txt")]
    for arch in archivos:
        try:
            # Leemos ignorando errores para que no se trabe por un acento
            with open(os.path.join(dir_documentos, arch), "r", encoding="utf-8", errors="ignore") as f:
                contenido = f.read()
                if len(contenido) > 10:
                    # Solo tomamos los primeros 5000 caracteres de cada guía para no saturar la IA
                    texto_acumulado += f"\n--- {arch} ---\n{contenido[:5000]}\n"
        except:
            continue
    return texto_acumulado

contexto = leer_guias()

if contexto:
    st.success("✅ Material de cátedra cargado correctamente.")
else:
    st.warning("⚠️ No se encontraron archivos de texto en la carpeta documentos.")

# --- PREGUNTA ---
pregunta = st.text_input("Escribe tu duda (ej: ranitidina):")

if pregunta and contexto:
    with st.spinner("Consultando guías..."):
        try:
            # Usamos Llama 3.1 8b: es el más rápido y estable para el aula
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Eres Profesor de Farmacología. Responde dudas técnicas usando solo el material provisto. Sé breve y preciso."},
                    {"role": "user", "content": f"MATERIAL:\n{contexto}\n\nPREGUNTA: {pregunta}"}
                ],
                temperature=0.0
            )
            st.subheader("📌 Respuesta:")
            st.write(res.choices[0].message.content)
        except Exception as e:
            st.error("La IA está tardando en responder. Intenta de nuevo.")
