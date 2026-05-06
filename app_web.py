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
# --- BUSCADOR Y CONSULTA MEJORADO ---
pregunta = st.text_input("Escribe tu duda técnica:")

if pregunta and material:
    with st.spinner("Buscando en todas las guías..."):
        # Buscamos coincidencias de forma más amplia
        palabras_pregunta = pregunta.lower().replace("?", "").split()
        contexto_relevante = ""
        
        # Ordenamos fragmentos por relevancia (cuántas palabras coinciden)
        puntuados = []
        for f in material:
            coincidencias = sum(1 for p in palabras_pregunta if p in f["texto"].lower())
            if coincidencias > 0:
                puntuados.append((coincidencias, f))
        
        puntuados.sort(key=lambda x: x[0], reverse=True)
        
        # Tomamos los 10 mejores fragmentos de cualquier guía
        for _, f in puntuados[:10]:
            contexto_relevante += f"\n--- De {f['fuente']} ---\n{f['texto']}\n"

        if not contexto_relevante:
            # Si no hay nada, mandamos un resumen de las primeras guías
            contexto_relevante = "\n".join([f["texto"] for f in material[:5]])

        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Eres Profesor de Farmacología Veterinaria. Responde usando SOLO el material provisto. Si la información no está en el material, indícalo."},
                    {"role": "user", "content": f"MATERIAL DE CÁTEDRA:\n{contexto_relevante}\n\nPREGUNTA DEL ALUMNO: {pregunta}"}
                ],
                temperature=0.0
            )
            
            # --- MOSTRAR RESPUESTA ---
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
            
            # --- MOSTRAR FUENTES (ESTO TE DA TRANQUILIDAD) ---
            with st.expander("🔍 Ver fuentes de las guías utilizadas"):
                st.write("El asistente extrajo información de los siguientes fragmentos:")
                st.info(contexto_relevante)
                
        except Exception as e:
            st.error("Error de conexión. Intenta de nuevo.")
