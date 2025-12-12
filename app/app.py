"""
CLASIFICADOR DE URGENCIAS MÉDICAS - Chatbot Web
Sistema de triaje inteligente basado en IA para orientar a pacientes sobre especialidades médicas

IMPORTANTE: Este sistema es solo orientativo y NO reemplaza la consulta médica profesional.
En caso de emergencia médica real, contacte inmediatamente al 911 o acuda al hospital más cercano.
"""

import gradio as gr
import pickle
import spacy
import numpy as np
from datetime import datetime
import os

# ============================================================
# CONFIGURACIÓN Y CARGA DE MODELOS
# ============================================================

print("🔧 Inicializando sistema de triaje médico...")

# Cargar modelo de lenguaje español
try:
    nlp = spacy.load("es_core_news_sm")
    print("Modelo de lenguaje español cargado")
except OSError:
    print("Descargando modelo de lenguaje español...")
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

# Cargar modelos entrenados (prioridad: pipeline -> archivos separados)


# Opción 1: Modelos actuales (Pipeline - PRIORIDAD)
modelo_pipeline = '../models/modelo_triaje_svm.pkl'
encoder_actual = '../models/label_encoder_final.pkl'

# Opción 2: Modelos de la celda 4 del notebook (archivos separados)
modelo_separado = '../models/svm_model.pickle'
vectorizador_separado = '../models/tfidf_vectorizer.pickle'
encoder_separado = '../models/label_encoder_svm.pickle'

try:
    # Intentar cargar modelos .pickle primero (PRIORIDAD para comparación)
    if os.path.exists(modelo_separado) and os.path.exists(vectorizador_separado) and os.path.exists(encoder_separado):
        with open(modelo_separado, 'rb') as f:
            svm_model = pickle.load(f)
        
        with open(vectorizador_separado, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        
        with open(encoder_separado, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print("Modelos cargados: Archivos .pickle (celda 4 del notebook)")
        usar_pipeline = False
    
    # Fallback: Cargar pipeline
    elif os.path.exists(modelo_pipeline) and os.path.exists(encoder_actual):
        with open(modelo_pipeline, 'rb') as f:
            svm_model = pickle.load(f)
        
        with open(encoder_actual, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print("Modelos cargados: Pipeline .pkl (modelo_triaje_svm.pkl)")
        usar_pipeline = True
    
    else:
        raise FileNotFoundError("No se encontraron modelos entrenados")

except FileNotFoundError as e:
    print("\nERROR: No se encontraron los modelos entrenados")
    print("\nOpciones para generar los modelos:")
    print("1. Ejecutar notebook: notebooks/3_entrenamiento_modelos.ipynb (celdas 1-3)")
    print("2. Ejecutar script: python src/train.py")
    print(f"\nArchivo faltante: {e}")
    exit(1)

# Configuración de negaciones (importantes en contexto médico)
negaciones = {'no', 'sin', 'ni', 'nunca', 'jamás', 'tampoco'}
for palabra in negaciones:
    nlp.vocab[palabra].is_stop = False

# ============================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================

def procesar_texto_medico(texto):
    """
    Procesa el texto del usuario aplicando NLP:
    - Tokenización
    - Eliminación de stopwords (preservando negaciones)
    - Lematización
    """
    if not texto or texto.strip() == "":
        return ""
    
    doc = nlp(texto.lower())
    tokens_limpios = []
    
    for token in doc:
        if not token.is_punct and not token.is_stop and token.is_alpha:
            tokens_limpios.append(token.lemma_)
    
    return " ".join(tokens_limpios)


def obtener_recomendaciones_especialidad(especialidad):
    """
    Proporciona recomendaciones específicas según la especialidad detectada
    """
    recomendaciones = {
        'CARDIOLOGÍA/CIRCULATORIO': {
            'emoji': '❤️',
            'urgencia': 'ALTA',
            'consejo': 'Si experimentas dolor en el pecho, dificultad para respirar o palpitaciones intensas, acude INMEDIATAMENTE a urgencias.',
            'medidas': [
                'Mantente en reposo',
                'No realices esfuerzos físicos',
                'Monitorea tu presión arterial si es posible',
                'En caso de dolor agudo: llama al 911'
            ]
        },
        'RESPIRATORIO/NEUMOLOGÍA': {
            'emoji': '🫁',
            'urgencia': 'MEDIA-ALTA',
            'consejo': 'Los problemas respiratorios pueden agravarse rápidamente. Consulta pronto a un especialista.',
            'medidas': [
                'Ventila bien los espacios',
                'Evita irritantes (humo, polvo)',
                'Si hay dificultad respiratoria severa: acude a urgencias',
                'Mantente hidratado'
            ]
        },
        'GASTROENTEROLOGÍA/DIGESTIVO': {
            'emoji': '🩺',
            'urgencia': 'MEDIA',
            'consejo': 'Los síntomas digestivos pueden indicar diversas condiciones. Se recomienda consulta médica.',
            'medidas': [
                'Dieta blanda y ligera',
                'Hidratación constante',
                'Si hay sangrado o dolor intenso: urgencias',
                'Evita alimentos irritantes'
            ]
        },
        'NEUROLOGÍA': {
            'emoji': '🧠',
            'urgencia': 'ALTA',
            'consejo': 'Los síntomas neurológicos requieren atención especializada urgente.',
            'medidas': [
                'No conduzcas ni operes maquinaria',
                'Reposo en lugar seguro',
                'Si hay confusión, parálisis o pérdida de conciencia: 911',
                'Anota cuándo comenzaron los síntomas'
            ]
        },
        'TRAUMATOLOGÍA/MUSCULAR': {
            'emoji': '🦴',
            'urgencia': 'MEDIA',
            'consejo': 'Las lesiones musculoesqueléticas necesitan evaluación para evitar complicaciones.',
            'medidas': [
                'Reposo de la zona afectada',
                'Aplicar hielo (primeras 48h)',
                'Inmovilizar si hay sospecha de fractura',
                'Si hay deformidad o dolor severo: urgencias'
            ]
        },
        'DERMATOLOGÍA': {
            'emoji': '🩹',
            'urgencia': 'BAJA',
            'consejo': 'Los problemas de piel generalmente no son urgentes, pero requieren diagnóstico profesional.',
            'medidas': [
                'No rascar ni tocar excesivamente',
                'Mantener la zona limpia y seca',
                'Evitar productos irritantes',
                'Consulta si empeora o se extiende'
            ]
        },
        'UROLOGÍA/RENAL': {
            'emoji': '🫘',
            'urgencia': 'MEDIA',
            'consejo': 'Los problemas urinarios o renales pueden ser serios. Consulta médica necesaria.',
            'medidas': [
                'Aumenta la ingesta de agua',
                'Evita retener la orina',
                'Si hay sangre en orina o dolor intenso: urgencias',
                'Monitorea la frecuencia urinaria'
            ]
        },
        'OFTALMOLOGÍA/ORL': {
            'emoji': '👁️',
            'urgencia': 'MEDIA',
            'consejo': 'Los problemas de visión, oído o garganta requieren evaluación especializada.',
            'medidas': [
                'No te frotes los ojos',
                'Evita sonidos muy fuertes',
                'Si hay pérdida súbita de visión/audición: urgencias',
                'Mantén buena higiene'
            ]
        },
        'PSIQUIATRÍA/MENTAL': {
            'emoji': '🧘',
            'urgencia': 'MEDIA',
            'consejo': 'La salud mental es igual de importante. Busca apoyo profesional.',
            'medidas': [
                'Habla con alguien de confianza',
                'Evita el aislamiento',
                'Si hay pensamientos de autolesión: llama a línea de crisis',
                'Mantén rutinas saludables'
            ]
        },
        'ONCOLOGÍA (TUMORES)': {
            'emoji': '🎗️',
            'urgencia': 'ALTA',
            'consejo': 'Cualquier sospecha de tumor requiere evaluación médica inmediata.',
            'medidas': [
                'Programa cita con especialista pronto',
                'No ignores síntomas persistentes',
                'Mantén un registro de síntomas',
                'Busca apoyo familiar y profesional'
            ]
        },
        'INFECCIOSAS/PARASITARIAS': {
            'emoji': '🦠',
            'urgencia': 'MEDIA-ALTA',
            'consejo': 'Las infecciones pueden propagarse o agravarse. Consulta médica necesaria.',
            'medidas': [
                'Aíslate si es contagioso',
                'Hidratación constante',
                'Monitorea la temperatura',
                'Si hay fiebre alta persistente: urgencias'
            ]
        }
    }
    
    # Buscar coincidencia parcial si no hay coincidencia exacta
    for key in recomendaciones:
        if key in especialidad.upper() or especialidad.upper() in key:
            return recomendaciones[key]
    
    # Recomendación genérica si no se encuentra la especialidad
    return {
        'emoji': '🏥',
        'urgencia': 'MEDIA',
        'consejo': 'Se recomienda consulta médica general para evaluación apropiada.',
        'medidas': [
            'Consulta con tu médico de cabecera',
            'Lleva un registro de tus síntomas',
            'No te automediques',
            'Busca atención si los síntomas empeoran'
        ]
    }


def predecir_especialidad(sintomas_usuario):
    """
    Función principal de predicción usando el modelo SVM
    """
    # Validación de entrada
    if not sintomas_usuario or sintomas_usuario.strip() == "":
        return "Por favor, describe tus síntomas para poder ayudarte."
    
    # Procesamiento del texto
    texto_procesado = procesar_texto_medico(sintomas_usuario)
    
    if not texto_procesado or len(texto_procesado.split()) < 2:
        return "No pude entender tus síntomas. Por favor, describe con más detalle qué sientes."
    
    # Predicción según el tipo de modelo cargado
    if usar_pipeline:
        # Pipeline: vectorización + clasificación en un solo paso
        prediccion_index = svm_model.predict([texto_procesado])[0]
        probabilidades = svm_model.predict_proba([texto_procesado])[0]
    else:
        # Modelos separados: vectorizar primero, luego clasificar
        texto_vectorizado = tfidf_vectorizer.transform([texto_procesado]).toarray()
        prediccion_index = svm_model.predict(texto_vectorizado)[0]
        probabilidades = svm_model.predict_proba(texto_vectorizado)[0]
    
    confianza = np.max(probabilidades) * 100
    
    # Decodificar especialidad
    especialidad = label_encoder.inverse_transform([prediccion_index])[0]
    
    # Obtener recomendaciones
    info = obtener_recomendaciones_especialidad(especialidad)
    
    # Construir respuesta
    respuesta = f"""
**ANÁLISIS COMPLETADO**

{info['emoji']} **Especialidad Recomendada:** {especialidad.upper()}
**Nivel de Confianza:** {confianza:.1f}%
**Nivel de Urgencia:** {info['urgencia']}

**Recomendación:**
{info['consejo']}

**Medidas Sugeridas:**
"""
    for i, medida in enumerate(info['medidas'], 1):
        respuesta += f"\n{i}. {medida}"
    
    respuesta += f"""

---
**RECORDATORIO IMPORTANTE:**
Este sistema es solo orientativo y utiliza Inteligencia Artificial.
NO reemplaza el diagnóstico médico profesional.
En caso de emergencia real, llama al 911 o acude a urgencias.

Análisis realizado: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
"""
    
    return respuesta


# ============================================================
# FUNCIÓN DE CHATBOT CONVERSACIONAL
# ============================================================

def chatbot_respuesta(mensaje, historial):
    """
    Función que maneja la conversación del chatbot
    Retorna el historial actualizado en formato compatible con Gradio 6.0
    """
    # Saludos y despedidas
    mensaje_lower = mensaje.lower().strip()
    
    saludos = ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'hey', 'saludos']
    despedidas = ['gracias', 'adios', 'adiós', 'hasta luego', 'chao', 'bye']
    
    if any(saludo in mensaje_lower for saludo in saludos) and len(mensaje_lower.split()) <= 3:
        respuesta = """¡Hola! Soy tu asistente médico virtual basado en IA.

Puedo ayudarte a orientarte sobre qué especialidad médica consultar según tus síntomas.

**¿Cómo funciono?**
Simplemente describe tus síntomas con el mayor detalle posible y te daré una recomendación.

Por ejemplo:
- "Tengo dolor fuerte en el pecho y me cuesta respirar"
- "Me duele mucho la cabeza y tengo náuseas"
- "Tengo fiebre alta y dolor de garganta"

**IMPORTANTE:** Soy una herramienta de orientación. En emergencias reales, llama al 911.

¿Qué síntomas estás experimentando?"""
    
    elif any(despedida in mensaje_lower for despedida in despedidas):
        respuesta = """¡Cuídate mucho y espero que te sientas mejor pronto!

Recuerda:
Consulta siempre con un médico profesional
En emergencias, llama al 911
No te automediques

¡Hasta pronto! """
    
    else:
        # Procesamiento de síntomas
        respuesta = predecir_especialidad(mensaje)
    
    # Inicializar historial si es None
    if historial is None:
        historial = []
    
    # Formato para Gradio 6.0: lista de diccionarios con 'role' y 'content'
    historial.append({"role": "user", "content": mensaje})
    historial.append({"role": "assistant", "content": respuesta})
    
    return historial


# ============================================================
# INTERFAZ GRADIO
# ============================================================

# Ejemplos predefinidos para guiar al usuario
ejemplos = [
    "Tengo un dolor muy fuerte en el pecho que se irradia al brazo izquierdo y me cuesta respirar",
    "Me salió una mancha roja en la piel que me pica muchísimo y está creciendo",
    "Tengo visión borrosa en el ojo derecho y me duele la cabeza del mismo lado",
    "Me caí y creo que me fracturé la pierna, está muy hinchada y no puedo apoyarla",
    "Siento ardor al orinar y dolor en la espalda baja cerca de los riñones",
    "Llevo tres días con vómitos constantes, fiebre y dolor abdominal intenso",
    "Tengo mareos muy fuertes, me duele la cabeza y siento náuseas",
    "Me siento muy ansioso, tengo palpitaciones y no puedo dormir desde hace semanas"
]

# Crear interfaz de chatbot
with gr.Blocks(title="Clasificador de Urgencias Médicas") as demo:
    
    gr.Markdown("""
    # Clasificador de Urgencias Médicas IA
    ### Sistema Inteligente de Triaje basado en Machine Learning
    
    ---
    
    **Describe tus síntomas y te orientaré sobre qué especialidad médica consultar**
    
    **DISCLAIMER:** Este sistema es solo orientativo y NO reemplaza la consulta médica profesional.
    En caso de emergencia médica real, contacta al **911** o acude al hospital más cercano.
    """)
    
    chatbot_interface = gr.Chatbot(
        label="Conversación Médica",
        height=500,
        show_label=True,
        avatar_images=(None, "https://em-content.zobj.net/thumbs/120/google/350/health-worker_1f9d1-200d-2695-fe0f.png")
    )
    
    with gr.Row():
        mensaje_input = gr.Textbox(
            label="Describe tus síntomas aquí",
            placeholder="Ejemplo: Tengo dolor fuerte en el pecho y dificultad para respirar...",
            lines=3,
            scale=4
        )
        enviar_btn = gr.Button("Enviar ", variant="primary", scale=1)
    
    # Función para limpiar el input después de enviar
    def responder_y_limpiar(mensaje, historial):
        nuevo_historial = chatbot_respuesta(mensaje, historial if historial else [])
        return nuevo_historial, ""  # Retorna historial actualizado y limpia el textbox
    
    gr.Markdown("### Ejemplos de consultas:")
    gr.Examples(
        examples=ejemplos,
        inputs=mensaje_input,
        label="Haz clic en algún ejemplo o escribe tu propia consulta"
    )
    
    gr.Markdown("""
    ---
    ### ℹInformación del Sistema
    - **Modelo:** Support Vector Machine (SVM) con kernel lineal
    - **Técnica NLP:** TF-IDF + Lematización con SpaCy
    - **Idioma:** Español
    - **Precisión del modelo:** >85% en datos de validación
    - **Dataset:** CodiEsp (casos clínicos reales en español)
    
    Desarrollado con fines educativos y de orientación médica general.
    """)
    
    # Eventos - Ahora usa la función que limpia el input
    mensaje_input.submit(
        responder_y_limpiar, 
        [mensaje_input, chatbot_interface], 
        [chatbot_interface, mensaje_input]
    )
    enviar_btn.click(
        responder_y_limpiar, 
        [mensaje_input, chatbot_interface], 
        [chatbot_interface, mensaje_input]
    )

# ============================================================
# LANZAMIENTO
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" LANZANDO INTERFAZ WEB DEL CLASIFICADOR MÉDICO")
    print("="*60)
    print("La aplicación se abrirá en tu navegador automáticamente")
    print(" URL local: http://127.0.0.1:7861")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,  # Cambia a True si quieres compartir públicamente
        show_error=True,
        theme=gr.themes.Soft()  # En Gradio 6.0, theme va en launch()
    )
