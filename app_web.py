import streamlit as st
import os
import numpy as np
import pdfplumber
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Farmaco UNRC", page_icon="💊")
st.title("🧠 Asistente de Farmacología")
st.info("Consulta las guías de la cátedra mediante IA.")

# --- LLAVE Y MODELOS ---
# Nota: En la nube usaremos "Secrets" para no dejar la llave a la vista
api_key = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=api_key)
modelo = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def cargar_datos():
    chunks = []
    if os.path.exists("documentos"):
        archivos = [f for f in os.listdir("documentos") if f.endswith(".pdf")]
        if not archivos:
            st.error("No se encontraron archivos PDF en la carpeta 'documentos'.")
            return None, None
            
        for archivo in archivos:
            try:
                with pdfplumber.open(f"documentos/{archivo}") as pdf:
                    texto_completo = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texto_completo += page_text + "\n"
                
                # Si el PDF está vacío o es una imagen/escaneo, esto fallará
                if len(texto_completo.strip()) < 50:
                    st.warning(f"El archivo {archivo} parece no tener texto legible (¿es un escaneo?).")
                    continue

                # Corte de texto con solapamiento (Overlap)
                partes = texto_completo.split("\n")
                buffer = ""
                for p in partes:
                    if len(buffer) < 700:
                        buffer += " " + p
                    else:
                        chunks.append({"texto": buffer.strip(), "doc": archivo})
                        buffer = p
                if buffer:
                    chunks.append({"texto": buffer.strip(), "doc": archivo})
            except Exception as e:
                st.error(f"Error al leer {archivo}: {e}")
        
        if not chunks:
            return None, None

        textos = [c["texto"] for c in chunks]
        embeddings = modelo.encode(textos, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return index, chunks
    return None, None
index, chunks = cargar_datos()

# --- INTERFAZ DE USUARIO ---
pregunta = st.text_input("¿Qué quieres consultar?")

if pregunta and index:
    with st.spinner("Buscando en los apuntes..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=10)
        
        contexto = ""
        for i, score in zip(idx[0], scores[0]):
            if i != -1 and score > 0.25:
                contexto += f"DOC: {chunks[i]['doc']}\nTEXTO: {chunks[i]['texto']}\n\n"
        
        if contexto:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": "Eres un profesor de farmacología veterinaria."},
                          {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}]
            )
            st.subheader("📌 Respuesta:")
            st.write(res.choices[0].message.content)
        else:
            st.warning("No encontré eso en mis guías.")
