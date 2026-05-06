import streamlit as st
import os
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

@st.cache_resource
def inicializar():
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    # Modelo pequeño y rápido
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return client, modelo

try:
    client, modelo = inicializar()
except:
    st.error("Revisa la API Key en los Secrets.")
    st.stop()

@st.cache_resource
def cargar_datos():
    chunks = []
    directorio = "documentos"
    if not os.path.exists(directorio): return None, "No existe carpeta documentos."
    
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
    for archivo in archivos:
        contenido = ""
        for enc en ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(os.path.join(directorio, archivo), "r", encoding=enc, errors="ignore") as f:
                    contenido = f.read()
                if len(contenido.strip()) > 10: break
            except: continue
        
        if contenido:
            # Cortamos en pedazos de 800 letras para no pasarnos de tokens
            for i in range(0, len(contenido), 800):
                chunks.append({"texto": contenido[i:i+1000], "doc": archivo})
    
    if not chunks: return None, "Archivos vacíos."
    
    textos = [c["texto"] for c in chunks]
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return (index, chunks), f"✅ ¡Listo! {len(archivos)} guías operativas."

resultado, mensaje = cargar_datos()
st.sidebar.info(mensaje)

# --- CONSULTA ---
pregunta = st.text_input("Escribe tu duda técnica:")

if pregunta and resultado:
    index, chunks = resultado
    with st.spinner("Buscando en las guías..."):
        # Buscamos solo los 5 fragmentos más relevantes
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=5)
        
        contexto = ""
        for i in idx[0]:
            if i != -1: contexto += f"\n{chunks[i]['texto']}\n"
        
        # Usamos el modelo 8b que es más rápido y tiene límites más amplios
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[
                    {"role": "system", "content": "Eres Profesor de Farmacología Veterinaria. Responde breve y técnico usando el texto provisto."},
                    {"role": "user", "content": f"Guías:\n{contexto}\n\nPregunta: {pregunta}"}
                ],
                temperature=0.0
            )
            st.subheader("📌 Respuesta:")
            st.write(res.choices[0].message.content)
        except Exception as e:
            st.error(f"Error de conexión. Intenta de nuevo en unos segundos.")
