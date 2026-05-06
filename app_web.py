import streamlit as st
import os
import numpy as np
import pdfplumber
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")
st.info("Consulta las guías de la cátedra mediante IA.")

# --- LLAVE Y MODELOS ---
api_key = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=api_key)
modelo = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def cargar_datos():
    chunks = []
    if os.path.exists("documentos"):
        archivos = [f for f in os.listdir("documentos") if f.endswith(".pdf")]
        if not archivos:
            return None, None
            
        for archivo in archivos:
            try:
                with pdfplumber.open(f"documentos/{archivo}") as pdf:
                    texto_completo = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texto_completo += page_text + "\n"
                
                if len(texto_completo.strip()) < 50:
                    continue

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
            except Exception:
                continue
        
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
    with st.spinner("Buscando en los apuntes de la cátedra..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=12)
        
        contexto = ""
        for i, score in zip(idx[0], scores[0]):
            # Bajamos el filtro a 0.20 para que sea más sensible
            if i != -1 and score > 0.20:
                contexto += f"DOC: {chunks[i]['doc']}\nTEXTO: {chunks[i]['texto']}\n\n"
        
        if contexto:
            prompt_estricto = f"""
            Actúa como un profesor de farmacología veterinaria. 
            Responde ÚNICAMENTE usando el CONTEXTO proporcionado.
            
            REGLAS:
            1. Si la respuesta no está en el CONTEXTO, di: "No encontré información en las guías de la cátedra."
            2. No inventes ni uses info de internet.
            3. Cita el nombre del archivo (DOC) al final de tu respuesta.

            CONTEXTO:
            {contexto}
            
            PREGUNTA:
            {pregunta}
            """

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Eres un asistente académico estricto."},
                    {"role": "user", "content": prompt_estricto}
                ],
                temperature=0.1
            )
            st.subheader("📌 Respuesta:")
            st.write(res.choices[0].message.content)
            
            with st.expander("🔍 Ver fragmentos analizados"):
                st.text(contexto)
        else:
            st.warning("No encontré información relevante en los apuntes cargados.")
