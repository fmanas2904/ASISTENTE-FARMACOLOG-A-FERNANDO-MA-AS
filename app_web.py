import streamlit as st
import os
from groq import Groq

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

# --- CONEXIÓN ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Error: Revisa la Key en los Secrets de Streamlit.")
    st.stop()

# --- CARGA DE TEXTO DIRECTA ---
def leer_todo():
    texto_total = ""
    directorio = "documentos"
    if not os.path.exists(directorio):
        return None
    
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
    for archivo in archivos:
        ruta = os.path.join(directorio, archivo)
        # Intentamos leer ignoreando errores de caracteres raros
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(ruta, "r", encoding=encoding, errors="ignore") as f:
                    contenido = f.read()
                    if len(contenido.strip()) > 10:
                        texto_total += f"\n--- GUIA: {archivo} ---\n{contenido}\n"
                        break
            except:
                continue
    return texto_total

with st.spinner("Cargando guías..."):
    contexto_completo = leer_todo()

if not contexto_completo:
    st.error("No se pudo leer ninguna guía. Verifica que los archivos .txt tengan texto adentro.")
else:
    st.success("✅ Guías cargadas y listas para la clase.")

# --- CONSULTA ---
pregunta = st.text_input("Escribe tu duda técnica:")

if pregunta and contexto_completo:
    with st.spinner("El profesor está pensando..."):
        # Le enviamos todo el bloque de texto directamente a Llama 3.3
        # Este modelo tiene memoria suficiente para leer todas tus guías de una vez
        prompt = f"""
        Eres el Profesor Titular de Farmacología. 
        Usa el siguiente material de la cátedra para responder la duda del alumno.
        Si la respuesta no está en el material, di que no figura en las guías.

        MATERIAL DE LA CÁTEDRA:
        {contexto_completo}

        PREGUNTA: {pregunta}
        """
        
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "Responde con rigor científico basado en el texto."},
                          {"role": "user", "content": prompt}],
                temperature=0.1
            )
            st.subheader("📌 Respuesta:")
            st.write(res.choices[0].message.content)
        except Exception as e:
            st.error(f"Error al conectar con la IA: {e}")
