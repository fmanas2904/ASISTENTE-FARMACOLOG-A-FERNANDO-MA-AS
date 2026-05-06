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
    # Usamos un modelo más liviano para evitar el cuelgue
    modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return client, modelo

try:
    client, modelo = inicializar_modelos()
except Exception as e:
    st.error("Error al cargar modelos o API Key. Revisa los Secrets.")
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
                contenido = f.read()
                # Dividimos por párrafos para mayor claridad
                parrafos = [p.strip() for p in contenido.split("\n\n") if len(p.strip()) > 10]
                for p in parrafos:
                    chunks.append({"texto": p, "doc": archivo})
        except:
            continue
    
    if not chunks: return None, "No se pudo extraer texto de las guías."

    textos = [c["texto"] for c in chunks]
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return (index, chunks), f"✅ ¡Listo! {len(archivos)} guías operativas."

resultado, mensaje_carga = cargar_datos()
st.sidebar.success(mensaje_carga)

# --- INTERFAZ ---
pregunta = st.text_input("Escribe tu duda técnica (ej: fenilbutazona):")

if pregunta and resultado:
    index, chunks = resultado
    with st.spinner("Buscando en las guías..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=10)
        
        contexto_lista = []
        for i, score in zip(idx[0], scores[0]):
            if i != -1 and score > 0.3: # Umbral de relevancia ajustado
                contexto_lista.append(f"ORIGEN: {chunks[i]['doc']}\n{chunks[i]['texto']}")
        
        if contexto_lista:
            contexto_unido = "\n\n---\n\n".join(contexto_lista)
            prompt = f"Eres Profesor de Farmacología Veterinaria. Responde la duda usando este contexto de tus guías:\n{contexto_unido}\n\nPregunta: {pregunta}"
            
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile", # Modelo más potente para mejores respuestas
                messages=[{"role": "system", "content": "No des respuestas genéricas. Usa el texto provisto."},
                          {"role": "user", "content": prompt}],
                temperature=0.0
            )
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
        else:
            st.warning("No encontré ese detalle específico en los archivos subidos.")
