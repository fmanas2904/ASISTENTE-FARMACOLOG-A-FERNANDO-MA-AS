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
try:
    api_key = st.secrets["GROQ_API_KEY"] 
    client = Groq(api_key=api_key)
except:
    st.error("⚠️ Error: No se encontró la clave GROQ_API_KEY en los Secrets.")
    st.stop()

modelo = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def cargar_datos():
    chunks = []
    directorio = "documentos"
    if os.path.exists(directorio):
        # Aceptamos .txt y .TXT
        archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
        
        if not archivos:
            return None, "No se encontraron archivos .txt en la carpeta documentos."
            
        for archivo in archivos:
            try:
                with open(os.path.join(directorio, archivo), "r", encoding="utf-8") as f:
                    texto_completo = f.read()
                
                # Dividir en fragmentos
                lineas = texto_completo.split("\n")
                buffer = ""
                for linea in lineas:
                    if len(buffer) < 1200:
                        buffer += " " + linea
                    else:
                        chunks.append({"texto": buffer.strip(), "doc": archivo})
                        buffer = linea
                if buffer:
                    chunks.append({"texto": buffer.strip(), "doc": archivo})
            except Exception as e:
                continue
        
        if not chunks: return None, "Los archivos están vacíos o no se pudieron leer."

        textos = [c["texto"] for c in chunks]
        embeddings = modelo.encode(textos, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return (index, chunks), f"✅ ¡Listo! Se cargaron {len(archivos)} guías correctamente."
    return None, "No se encontró la carpeta 'documentos'."

resultado, mensaje_carga = cargar_datos()
st.sidebar.write(mensaje_carga)

# --- INTERFAZ ---
pregunta = st.text_input("Escribe tu duda técnica (ej: amoxicilina):")

if pregunta:
    if resultado:
        index, chunks = resultado
        with st.spinner("Buscando respuesta..."):
            q = modelo.encode([pregunta], normalize_embeddings=True)
            scores, idx = index.search(np.array(q).astype("float32"), k=15)
            
            contexto_lista = []
            for i, score in zip(idx[0], scores[0]):
                if i != -1 and score > 0.02:
                    contexto_lista.append(f"ARCHIVO: {chunks[i]['doc']}\n{chunks[i]['texto']}")
            
            if contexto_lista:
                contexto_unido = "\n\n---\n\n".join(contexto_lista)
                prompt = f"Responde la duda del alumno usando solo este contexto:\n{contexto_unido}\n\nPregunta: {pregunta}"
                
                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                st.subheader("📌 Respuesta:")
                st.write(res.choices[0].message.content)
            else:
                st.warning("No encontré información sobre eso en las guías.")
    else:
        st.error(mensaje_carga)
