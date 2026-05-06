import streamlit as st
import os
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")
st.info("Consulta las guías de la cátedra mediante archivos de texto.")

# --- LLAVE Y MODELOS ---
api_key = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=api_key)
modelo = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def cargar_datos():
    chunks = []
    # Buscamos en la carpeta 'documentos'
    if os.path.exists("documentos"):
        archivos = [f for f in os.listdir("documentos") if f.endswith(".txt")]
        if not archivos:
            return None, None
            
        for archivo in archivos:
            try:
                # Leemos el archivo TXT directamente
                with open(f"documentos/{archivo}", "r", encoding="utf-8") as f:
                    texto_completo = f.read()
                
                if len(texto_completo.strip()) < 10: continue

                # Dividimos en fragmentos grandes para no perder la explicación
                lineas = texto_completo.split("\n")
                buffer = ""
                for linea in lineas:
                    if len(buffer) < 1500:
                        buffer += " " + linea
                    else:
                        chunks.append({"texto": buffer.strip(), "doc": archivo})
                        buffer = linea
                if buffer:
                    chunks.append({"texto": buffer.strip(), "doc": archivo})
            except Exception:
                continue
        
        if not chunks: return None, None

        textos = [c["texto"] for c in chunks]
        embeddings = modelo.encode(textos, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return index, chunks
    return None, None

index, chunks = cargar_datos()

# --- INTERFAZ DE USUARIO ---
pregunta = st.text_input("Escribe tu duda técnica (ej: Ranitidina):")

if pregunta and index:
    with st.spinner("Buscando en las guías de texto..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        # Traemos 20 fragmentos para asegurar que encuentre el tema
        scores, idx = index.search(np.array(q).astype("float32"), k=20)
        
        contexto_lista = []
        for i, score in zip(idx[0], scores[0]):
            # Filtro de sensibilidad extrema para términos técnicos
            if i != -1 and score > 0.05:
                contexto_lista.append(f"ARCHIVO: {chunks[i]['doc']}\nCONTENIDO: {chunks[i]['texto']}")
        
        if contexto_lista:
            contexto_unido = "\n\n---\n\n".join(contexto_lista)
            
            prompt_estricto = f"""
            Eres el Asistente de Cátedra de Farmacología Veterinaria. 
            Responde la duda del alumno usando EXCLUSIVAMENTE el CONTEXTO proporcionado.
            
            REGLAS:
            1. Si la respuesta no está en el CONTEXTO, di: "No encontré ese detalle en las guías".
            2. Cita siempre el nombre del ARCHIVO al final.
            3. Mantén la precisión técnica (ej. si es bactericida o bacteriostático).

            CONTEXTO:
            {contexto_unido}
            
            PREGUNTA: {pregunta}
            """

            try:
                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Eres un extractor de información literal de guías académicas."},
                        {"role": "user", "content": prompt_estricto}
                    ],
                    temperature=0.0
                )
                st.subheader("📌 Respuesta de la Cátedra:")
                st.write(res.choices[0].message.content)
            except Exception:
                st.error("Error temporal de conexión. Intenta de nuevo en unos segundos.")
            
            with st.expander("🔍 Ver texto analizado"):
                for f in contexto_lista: st.markdown(f"**{f}**")
        else:
            st.warning("⚠️ No encontré ninguna referencia a ese tema en los archivos .txt")
