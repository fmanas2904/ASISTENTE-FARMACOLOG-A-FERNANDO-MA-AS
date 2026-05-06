import streamlit as st
import os
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

# --- LLAVE Y MODELOS ---
@st.cache_resource
def inicializar_modelos():
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
    # Modelo robusto para español
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return client, modelo

try:
    client, modelo = inicializar_modelos()
except:
    st.error("Error en la API Key o Modelos. Revisa los Secrets.")
    st.stop()

@st.cache_resource
def cargar_datos():
    chunks = []
    directorio = "documentos"
    if not os.path.exists(directorio):
        return None, "No se encontró la carpeta 'documentos'."
    
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
    if not archivos:
        return None, "No hay archivos .txt en la carpeta."

    for archivo in archivos:
        try:
            with open(os.path.join(directorio, archivo), "r", encoding="utf-8") as f:
                # Leemos todo y dividimos en bloques de 500 caracteres para no fallar
                contenido = f.read()
                if len(contenido.strip()) < 10: continue
                
                # Dividimos por párrafos o líneas largas
                partes = contenido.split("\n")
                temp_chunk = ""
                for p in partes:
                    temp_chunk += " " + p.strip()
                    if len(temp_chunk) > 600: # Tamaño ideal para veterinaria
                        chunks.append({"texto": temp_chunk.strip(), "doc": archivo})
                        temp_chunk = ""
                if temp_chunk:
                    chunks.append({"texto": temp_chunk.strip(), "doc": archivo})
        except:
            continue
    
    if not chunks: return None, "Error crítico: No se pudo procesar el texto de los archivos."

    textos = [c["texto"] for c in chunks]
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return (index, chunks), f"✅ ¡Listo! {len(archivos)} guías cargadas."

resultado, mensaje_carga = cargar_datos()
st.sidebar.success(mensaje_carga)

# --- INTERFAZ ---
pregunta = st.text_input("Escribe tu duda técnica (ej: amoxicilina):")

if pregunta and resultado:
    index, chunks = resultado
    with st.spinner("Buscando en las guías..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=12)
        
        contexto_lista = []
        for i, score in zip(idx[0], scores[0]):
            if i != -1 and score > 0.2:
                contexto_lista.append(f"GUÍA: {chunks[i]['doc']}\n{chunks[i]['texto']}")
        
        if contexto_lista:
            contexto_unido = "\n\n---\n\n".join(contexto_lista)
            prompt = f"""
            Eres el Profesor Titular de Farmacología. 
            Responde de forma técnica y profesional usando SOLO este contexto de las guías:
            
            {contexto_unido}
            
            Pregunta del alumno: {pregunta}
            """
            
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "Sé preciso y académico. Cita la guía si es posible."},
                          {"role": "user", "content": prompt}],
                temperature=0.0
            )
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
        else:
            st.warning("No encontré información específica en las guías actuales.")
