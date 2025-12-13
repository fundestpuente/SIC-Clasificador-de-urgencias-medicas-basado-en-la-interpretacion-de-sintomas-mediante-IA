import gradio as gr
import sys
import os
from datetime import datetime

# ============================================================
# CONFIGURACIÓN DE RUTAS E IMPORTACIONES
# ============================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.predict import load_artifacts, predict_single

# ============================================================
# CARGA DE MODELOS
# ============================================================
print("🔧 Inicializando sistema de triaje médico...")

try:
    svm_model, label_encoder = load_artifacts()
    print(f"✅ Modelos cargados correctamente.")
except Exception as e:
    print(f"\n❌ ERROR FATAL: {e}")
    print("Ejecuta 'python src/train.py' para generar los modelos.")
    exit(1)


# ============================================================
# LÓGICA DE NEGOCIO (RECOMENDACIONES)
# ============================================================

def obtener_recomendaciones(especialidad):
    """Retorna consejos basados en la especialidad predicha."""
    recomendaciones = {
        'CARDIOLOGÍA': {'emoji': '❤️', 'urgencia': 'ALTA',
                        'consejo': 'Dolor de pecho o dificultad respiratoria requieren atención inmediata.'},
        'NEUMOLOGÍA': {'emoji': '🫁', 'urgencia': 'MEDIA-ALTA',
                       'consejo': 'Vigila la saturación de oxígeno y dificultad para respirar.'},
        'GASTROENTEROLOGÍA': {'emoji': '🩺', 'urgencia': 'MEDIA',
                              'consejo': 'Hidratación constante. Acude a urgencias si hay dolor agudo.'},
        'NEUROLOGÍA': {'emoji': '🧠', 'urgencia': 'ALTA',
                       'consejo': 'Pérdida de fuerza, habla o visión requieren activación de emergencia (911).'},
        'TRAUMATOLOGÍA': {'emoji': '🦴', 'urgencia': 'MEDIA', 'consejo': 'Inmovilizar la zona. Aplicar frío local.'},
        'DERMATOLOGÍA': {'emoji': '🩹', 'urgencia': 'BAJA',
                         'consejo': 'Evita rascar o aplicar remedios caseros sin receta.'},
        'UROLOGÍA': {'emoji': '🫘', 'urgencia': 'MEDIA',
                     'consejo': 'Beber agua. Si hay fiebre alta o sangrado, consultar urgente.'},
        'OFTALMOLOGÍA': {'emoji': '👁️', 'urgencia': 'MEDIA',
                         'consejo': 'No frotar los ojos. Lavar con agua limpia si cayó sustancia.'},
        'PSIQUIATRÍA': {'emoji': '🧘', 'urgencia': 'MEDIA',
                        'consejo': 'Busca compañía de confianza o llama a líneas de ayuda.'},
        'ONCOLOGÍA': {'emoji': '🎗️', 'urgencia': 'ALTA', 'consejo': 'Consulta prioritaria con especialista.'},
        'INFECCIOSAS': {'emoji': '🦠', 'urgencia': 'MEDIA-ALTA',
                        'consejo': 'Aislamiento preventivo y control de fiebre.'}
    }

    # Búsqueda parcial (ej: "CARDIOLOGÍA/CIRCULATORIO" -> Match con "CARDIOLOGÍA")
    especialidad_upper = especialidad.upper()
    for key, val in recomendaciones.items():
        if key in especialidad_upper:
            return val

    return {'emoji': '🏥', 'urgencia': 'MEDIA', 'consejo': 'Consulta a tu médico general.'}


def generar_respuesta_texto(mensaje):
    """Genera el texto de respuesta del bot."""
    if not mensaje or not mensaje.strip():
        return "Por favor, describe tus síntomas."

    # Predicción
    especialidad, confianza, _ = predict_single(mensaje, svm_model, label_encoder)

    if especialidad is None:
        return "No entendí los síntomas. Intenta ser más descriptivo."

    # Formateo de respuesta
    info = obtener_recomendaciones(especialidad)
    return (
        f"### Análisis {info['emoji']}\n\n"
        f"**Especialidad:** {especialidad}\n"
        f"**Confianza:** {confianza:.1%} | **Urgencia:** {info['urgencia']}\n\n"
        f"*⚠️ IA Orientativa - No reemplaza consulta médica.*"
    )


# =========
# CHATBOT
# =========

def responder(mensaje, historial):

    if historial is None:
        historial = []

    respuesta_bot = generar_respuesta_texto(mensaje)


    historial.append([mensaje, respuesta_bot])

    # Retornamos historial actualizado y limpiamos el input (string vacío)
    return historial, ""


# ============================================================
# INTERFAZ
# ============================================================

with gr.Blocks(title="Triaje Médico IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 Clasificador de Urgencias Médicas")
    gr.Markdown("Describe tus síntomas y la IA te sugerirá la especialidad médica.")

    # Chatbot clásico (sin type="messages", usa el defecto que es tuplas/listas)
    chatbot = gr.Chatbot(
        label="Asistente Virtual",
        height=450,
        avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/3774/3774299.png")
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Tus síntomas",
            placeholder="Ej: Tengo fiebre y dolor de cabeza...",
            scale=4
        )
        btn = gr.Button("Enviar", variant="primary", scale=1)

    # Manejo de eventos
    # input: [msg, chatbot] -> output: [chatbot, msg]
    msg.submit(responder, [msg, chatbot], [chatbot, msg])
    btn.click(responder, [msg, chatbot], [chatbot, msg])

    gr.Examples(
        examples=[
            "Dolor fuerte en el pecho y falta de aire",
            "Manchas rojas en la piel que pican mucho",
            "Golpe en la pierna, está hinchada y duele al caminar"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)