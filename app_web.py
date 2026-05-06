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
# --- INTERFAZ DE USUARIO ---
pregunta = st.text_input("¿Qué quieres consultar?")

if pregunta and index:
    with st.spinner("Analizando guías de cátedra..."):
        q = modelo.encode([pregunta], normalize_embeddings=True)
        scores, idx = index.search(np.array(q).astype("float32"), k=12)
        
        contexto_lista = []
        for i, score in zip(idx[0], scores[0]):
            # Umbral de seguridad: si el fragmento no se parece a la pregunta, se descarta
            if i != -1 and score > 0.18:
                contexto_lista.append(f"ARCHIVO: {chunks[i]['doc']}\nCONTENIDO: {chunks[i]['texto']}")
        
        if contexto_lista:
            contexto_unido = "\n\n---\n\n".join(contexto_lista)
            
            # PROMPT DE BLOQUEO TOTAL
            prompt_blindado = f"""
            ESTRICTAMENTE PROHIBIDO USAR INFORMACIÓN EXTERNA.
            SOLO PUEDES RESPONDER USANDO EL 'CONTEXTO DE LAS GUÍAS' PROPORCIONADO ABAJO.
            
            SI LA RESPUESTA NO ESTÁ EXPLÍCITA EN EL CONTEXTO:
            Responde exactamente: "Lo siento, la información solicitada no se encuentra en las guías de estudio cargadas actualmente."
            
            REGLAS DE ORO:
            1. No menciones nada que no esté escrito en los fragmentos de abajo.
            2. No corrijas ni añadidas datos de internet aunque creas que el contexto está incompleto.
            3. Cita siempre el nombre del ARCHIVO al finalizar la respuesta.

            CONTEXTO DE LAS GUÍAS:
            {contexto_unido}
            
            PREGUNTA DEL ALUMNO:
            {pregunta}
            """

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Eres un extractor de texto literal. No tienes conocimientos generales, solo acceso a los documentos proporcionados."},
                    {"role": "user", "content": prompt_blindado}
                ],
                temperature=0.0  # MÁXIMA PRECISIÓN, CERO INVENCIÓN
            )
            
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
            
            with st.expander("🔍 Ver párrafos originales analizados"):
                for fragmento in contexto_lista:
                    st.markdown(f"**{fragmento}**")
        else:
            st.warning("⚠️ No se encontró ninguna referencia a ese tema en los PDFs de la carpeta 'documentos'.")
