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
        for archivo in archivos:
            try:
                with pdfplumber.open(f"documentos/{archivo}") as pdf:
                    texto_completo = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            texto_completo += page_text + "\n"
                
                if len(texto_completo.strip()) < 50: continue

                # Fragmentos de 1000 caracteres para no perder contexto
                partes = texto_completo.split("\n")
                buffer = ""
                for p in partes:
                    if len(buffer) < 1000:
                        buffer += " " + p
                    else:
                        chunks.append({"texto": buffer.strip(), "doc": archivo})
                        buffer = p
                if buffer: chunks.append({"texto": buffer.strip(), "doc": archivo})
            except Exception: continue
        
        if not chunks: return None, None
        textos = [c["texto"] for c in chunks]
        embeddings = modelo.encode(textos, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return index, chunks
    return None, None

index, chunks = cargar_datos()

# --- INTERFAZ DE USUARIO ---
pregunta = st.text_input("Escribe tu duda técnica aquí:")

if pregunta and index:
    with st.spinner("Buscando en las guías..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        # k=15 es el punto justo para no saturar la API
        scores, idx = index.search(np.array(q).astype("float32"), k=15)
        
        contexto_lista = []
        for i, score in zip(idx[0], scores[0]):
            # Filtro muy sensible para encontrar términos específicos
            if i != -1 and score > 0.05:
                contexto_lista.append(f"ARCHIVO: {chunks[i]['doc']}\nCONTENIDO: {chunks[i]['texto']}")
        
        if contexto_lista:
            contexto_unido = "\n\n---\n\n".join(contexto_lista)
            
            prompt_estricto = f"""
            Eres profesor de Farmacología Veterinaria. RESPONDE SOLO CON EL CONTEXTO.
            CONTEXTO:
            {contexto_unido}
            PREGUNTA: {pregunta}
            """

            try:
                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Si la info no está en el contexto, di que no está en las guías."},
                        {"role": "user", "content": prompt_estricto}
                    ],
                    temperature=0.0
                )
                st.subheader("📌 Respuesta:")
                st.write(res.choices[0].message.content)
            except Exception as e:
                st.error("Error de conexión con el motor de IA. Intenta de nuevo en unos segundos.")
            
            with st.expander("🔍 Ver fragmentos analizados"):
                for f in contexto_lista: st.markdown(f"**{f}**")
        else:
            st.warning("No se encontró referencia a ese tema en los PDFs.")
