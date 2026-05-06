import streamlit as st
import os
from groq import Groq

st.set_page_config(page_title="Asistente Farmaco", page_icon="💊")
st.title("🧠 Asistente de Farmacología")

# --- CONEXIÓN DIRECTA ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Error en la configuración de la clave.")
    st.stop()

# --- LECTURA RÁPIDA DE ARCHIVOS ---
@st.cache_data
def obtener_fragmentos():
    fragmentos = []
    directorio = "documentos"
    if not os.path.exists(directorio): return []
    
    archivos = [f for f in os.listdir(directorio) if f.lower().endswith(".txt")]
    for archivo in archivos:
        try:
            with open(os.path.join(directorio, archivo), "r", encoding="utf-8", errors="ignore") as f:
                contenido = f.read()
                # Dividimos por párrafos para buscar con precisión
                parrafos = [p.strip() for p in contenido.split("\n\n") if len(p.strip()) > 20]
                for p in parrafos:
                    fragmentos.append({"texto": p, "fuente": archivo})
        except: continue
    return fragmentos

material = obtener_fragmentos()

if material:
    st.sidebar.success(f"✅ {len(os.listdir('documentos'))} guías listas.")
else:
    st.error("No se encontraron guías en la carpeta 'documentos'.")

# --- BUSCADOR Y CONSULTA ---
# --- BUSCADOR Y CONSULTA REFORZADO ---
pregunta = st.text_input("Escribe tu duda técnica (ej: mecanismo de glucocorticoides):")

if pregunta and material:
    with st.spinner("Buscando en las guías de la cátedra..."):
        # Limpieza de pregunta para búsqueda
        palabras_pregunta = pregunta.lower().replace("?", "").replace("¿", "").split()
        
        # Puntuación por relevancia
        puntuados = []
        for f in material:
            # Buscamos coincidencias de palabras completas
            coincidencias = sum(1 for p in palabras_pregunta if p in f["texto"].lower())
            if coincidencias > 0:
                puntuados.append((coincidencias, f))
        
        # Ordenar y seleccionar los mejores 15 (equilibrio entre contexto y velocidad)
        puntuados.sort(key=lambda x: x[0], reverse=True)
        fragmentos_finales = puntuados[:15]
        
        contexto_relevante = ""
        for _, f in fragmentos_finales:
            contexto_relevante += f"\n--- FUENTE: {f['fuente']} ---\n{f['texto']}\n"

        if not contexto_relevante:
            contexto_relevante = "No se encontraron fragmentos específicos. Usa el conocimiento general solo para indicar que no figura en las guías."

        try:
            # Llamada al modelo 70b con instrucciones estrictas
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "Eres un Asistente Académico de Farmacología Veterinaria extremadamente riguroso. "
                            "Tu única fuente de verdad es el MATERIAL DE CÁTEDRA provisto. "
                            "REGLAS CRÍTICAS:\n"
                            "1. Si la respuesta no está en el material, di: 'Esta información no figura en las guías actuales'.\n"
                            "2. PROHIBIDO inventar prodrogas, mecanismos o ejemplos que no estén escritos en el texto.\n"
                            "3. Usa terminología técnica y mantén un tono profesional docente.\n"
                            "4. Si encuentras palabras con errores de tildes (ej. 'accin'), corrígelas en tu respuesta (ej. 'acción')."
                        )
                    },
                    {"role": "user", "content": f"MATERIAL DE CÁTEDRA:\n{contexto_relevante}\n\nPREGUNTA DEL ALUMNO: {pregunta}"}
                ],
                temperature=0.0 # Cero creatividad para evitar errores médicos
            )
            
            # --- MOSTRAR RESPUESTA ---
            st.subheader("📌 Respuesta de la Cátedra:")
            st.write(res.choices[0].message.content)
            
            # --- MOSTRAR FUENTES ---
            with st.expander("🔍 Ver fragmentos originales analizados"):
                st.write("El asistente utilizó estos párrafos para construir la respuesta:")
                st.info(contexto_relevante)
                
        except Exception as e:
            st.error("Error de conexión con el servidor de IA. Por favor, intenta de nuevo en unos segundos.")
