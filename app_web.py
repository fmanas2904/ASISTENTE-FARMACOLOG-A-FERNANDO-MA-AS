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
# Una sola vez el input para evitar el error de DuplicateElementId
pregunta = st.text_input("Escribe tu duda técnica aquí:")

if pregunta and index:
    with st.spinner("Analizando guías de cátedra..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=12)
        
        contexto_lista = []
        for i, score in zip(idx[0], scores[0]):
            if i != -1 and score > 0.18:
                contexto_lista.append(f"ARCHIVO: {chunks[i]['doc']}\nCONTENIDO: {chunks[i]['texto']}")
        
        if contexto_lista:
            contexto_unido = "\n\n---\n\n".join(contexto_lista)
            
            prompt_blindado = f"""
            ESTRICTAMENTE PROHIBIDO USAR INFORMACIÓN EXTERNA.
            RESPONDE ÚNICAMENTE USANDO EL CONTEXTO ABAJO.
            
            SI NO ESTÁ EN EL CONTEXTO:
            Responde: "Lo siento, la información no está en las guías cargadas."
            
            CONTEXTO:
            {contexto_unido}
            
            PREGUNTA:
            {pregunta}
            """

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Eres un extractor de texto académico literal."},
                    {"role": "user", "content": prompt_blindado}
                ],
                temperature=0.0  # Mínima creatividad, máxima precisión
            )
            
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
            
            with st.expander("🔍 Ver fragmentos analizados"):
                for f in contexto_lista:
                    st.markdown(f"**{f}**")
        else:
            st.warning("⚠️ No se encontró referencia a ese tema en los PDFs.")
