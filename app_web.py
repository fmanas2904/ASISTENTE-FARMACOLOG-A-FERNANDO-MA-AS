import streamlit as st
import os
from groq import Groq

st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

# --- CONEXIÓN DIRECTA ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Error en la configuración de la clave.")
    st.stop()

# --- LECTURA RÁPIDA DE ARCHIVOS ---
@st.cache_data
def obtener_fragmentos():
    fragmentos = []
    directorio = "documentos"
    if not os.path.exists(directorio): return []
    
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
    for archivo in archivos:
        try:
            with open(os.path.join(directorio, archivo), "r", encoding="utf-8", errors="ignore") as f:
                contenido = f.read()
                # Dividimos por párrafos para buscar con precisión
                parrafos = [p.strip() for p in contenido.split("\n\n") if len(p.strip()) > 20]
                for p in parrafos:
                    fragmentos.append({"texto": p, "fuente": archivo})
        except: continue
    return fragmentos

material = obtener_fragmentos()

if material:
    st.sidebar.success(f"✅ {len(os.listdir('documentos'))} guías listas.")
else:
    st.error("No se encontraron guías en la carpeta 'documentos'.")

# --- BUSCADOR Y CONSULTA ---
pregunta = st.text_input("Escribe tu duda técnica (ej: glucocorticoides):")

if pregunta and material:
    with st.spinner("Buscando en las guías..."):
        # Buscador de palabras clave ultra-rápido
        palabras_clave = pregunta.lower().split()
        contexto_relevante = ""
        encontrados = 0
        
        for f in material:
            if any(palabra in f["texto"].lower() for palabra in palabras_clave):
                contexto_relevante += f"\n--- De {f['fuente']} ---\n{f['texto']}\n"
                encontrados += 1
            if encontrados > 8: break # Límite para no saturar la conexión

        if not contexto_relevante:
            # Si no hay coincidencia exacta, mandamos una muestra general
            contexto_relevante = "\n".join([f["texto"] for f in material[:5]])

        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Eres Profesor de Farmacología. Responde usando exclusivamente el material de cátedra provisto. Sé técnico y preciso."},
                    {"role": "user", "content": f"MATERIAL DE CÁTEDRA:\n{contexto_relevante}\n\nPREGUNTA: {pregunta}"}
                ],
                temperature=0.0
            )
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
        except Exception as e:
            st.error("La conexión con el servidor de IA falló. Por favor, intenta de nuevo.")
