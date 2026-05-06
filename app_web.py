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
                contenido = f.read()
                if len(contenido.strip()) > 5:
                    # Guardamos el archivo completo como un solo bloque para no fallar
                    chunks.append({"texto": contenido.strip(), "doc": archivo})
        except:
            continue
    
    if not chunks: return None, "Error: Los archivos están vacíos."

    textos = [c["texto"] for c in chunks]
    embeddings = modelo.encode(textos, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return (index, chunks), f"✅ ¡Listo! {len(archivos)} guías cargadas."

resultado, mensaje_carga = cargar_datos()
st.sidebar.success(mensaje_carga)

# --- INTERFAZ ---
pregunta = st.text_input("Escribe tu duda técnica:")

if pregunta and resultado:
    index, chunks = resultado
    with st.spinner("Buscando en las guías..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=5)
        
        contexto_unido = ""
        for i in idx[0]:
            if i != -1:
                contexto_unido += f"\n--- ARCHIVO: {chunks[i]['doc']} ---\n{chunks[i]['texto']}\n"
        
        prompt = f"Eres Profesor de Farmacología. Responde usando este contexto:\n{contexto_unido}\n\nPregunta: {pregunta}"
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Sé técnico y usa el texto provisto."},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        st.subheader("📌 Respuesta:")
        st.write(res.choices[0].message.content)
