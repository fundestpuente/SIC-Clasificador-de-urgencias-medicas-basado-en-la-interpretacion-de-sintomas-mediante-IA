# 🚑 Interfaz Web - Clasificador de Urgencias Médicas

Aplicación web de chatbot conversacional para clasificación de urgencias médicas mediante IA.

## 📋 Descripción

Este módulo contiene la interfaz web interactiva del sistema de triaje médico. Utiliza el modelo SVM entrenado para proporcionar orientación sobre qué especialidad médica consultar según los síntomas descritos por el usuario.

## 🗂️ Archivos

- **`app.py`**: Interfaz web principal con Gradio (chatbot conversacional)
- **`train_svm.py`**: Script para entrenar y guardar el modelo SVM
- **`README.md`**: Este archivo

## 🚀 Instalación y Uso

### 📌 Requisitos Previos

**Versión de Python requerida:**
- ✅ **Recomendado:** Python 3.11.x (versión del proyecto)
- ✅ **Compatible:** Python 3.9.x, 3.10.x, 3.11.x
- ⚠️ **No recomendado:** Python 3.12.x (puede tener issues menores)
- ❌ **Incompatible:** Python 3.13+ (TensorFlow no funciona)

**Verificar tu versión:**
```bash
python --version
```

Si tienes una versión incompatible, considera usar un entorno virtual con Python 3.11.

### 1️⃣ Instalar dependencias

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

**Nota:** Si usas Python 3.12 y TensorFlow falla, intenta:
```bash
pip install tensorflow==2.15.0
```

### 2️⃣ Entrenar el modelo SVM

**IMPORTANTE:** Este paso es necesario solo la primera vez o cuando quieras reentrenar el modelo.

```bash
cd view
python train_svm.py
```

Este comando:
- ✅ Carga los datos procesados
- ✅ Entrena el modelo SVM optimizado
- ✅ Guarda los modelos en `../models/`
- ✅ Muestra métricas de rendimiento

### 3️⃣ Lanzar la aplicación web

```bash
python app.py
```

La aplicación se abrirá automáticamente en tu navegador en: **http://127.0.0.1:7860**

## 💡 Características

### ✨ Funcionalidades del Chatbot

- 🗣️ **Conversación natural**: Interacción fluida en español
- 🎯 **Predicción de especialidad**: Basada en modelo SVM (>85% accuracy)
- 📊 **Nivel de confianza**: Muestra probabilidad de la predicción
- ⚠️ **Niveles de urgencia**: Clasifica según gravedad (ALTA/MEDIA/BAJA)
- 💊 **Recomendaciones personalizadas**: Consejos específicos por especialidad
- 📋 **Medidas sugeridas**: Pasos a seguir según el caso
- 🕐 **Registro temporal**: Marca fecha/hora del análisis

### 🏥 Especialidades Detectadas

El sistema puede clasificar en las siguientes especialidades:

1. ❤️ Cardiología/Circulatorio
2. 🫁 Respiratorio/Neumología
3. 🩺 Gastroenterología/Digestivo
4. 🧠 Neurología
5. 🦴 Traumatología/Muscular
6. 🩹 Dermatología
7. 🫘 Urología/Renal
8. 👁️ Oftalmología/ORL
9. 🧘 Psiquiatría/Mental
10. 🎗️ Oncología
11. 🦠 Infecciosas/Parasitarias

## 🔧 Tecnologías Utilizadas

- **Gradio 6.x**: Framework de interfaz web
- **scikit-learn**: Modelo SVM
- **SpaCy**: Procesamiento de lenguaje natural (modelo es_core_news_sm)
- **NumPy & Pandas**: Operaciones numéricas y manipulación de datos
- **Pickle**: Serialización de modelos
- **Python 3.11.5**: Versión base del proyecto

## ⚠️ Disclaimer Legal

**IMPORTANTE:** Este sistema es **SOLO ORIENTATIVO** y utiliza Inteligencia Artificial para sugerencias generales.

❌ **NO reemplaza**:
- Diagnóstico médico profesional
- Consulta con especialistas
- Atención médica de emergencia

✅ **En caso de emergencia real**:
- Llama inmediatamente al **911**
- Acude al hospital más cercano
- No dependas únicamente de esta herramienta

## 📊 Rendimiento del Modelo

- **Algoritmo**: SVM (Support Vector Machine) con kernel lineal
- **Vectorización**: TF-IDF (5000 features)
- **Precisión esperada**: >85% en datos de validación
- **Dataset**: CodiEsp (casos clínicos reales en español)

## 🎨 Personalización

### Cambiar el puerto

Edita `app.py` en la sección de lanzamiento:

```python
demo.launch(
    server_port=7860,  # Cambia este número
    ...
)
```

### Compartir públicamente

Para generar un link público temporal (útil para demos):

```python
demo.launch(
    share=True,  # Cambia a True
    ...
)
```

## 🐛 Solución de Problemas

### Error: "No se encontraron los modelos entrenados"

**Solución**: Ejecuta primero `python train_svm.py`

### Error: "Modelo de SpaCy no encontrado"

**Solución**: 
```bash
python -m spacy download es_core_news_sm
```

### Error: "ModuleNotFoundError: No module named 'gradio'"

**Solución**:
```bash
pip install gradio
```

### La aplicación no se abre en el navegador

**Solución**: Abre manualmente http://127.0.0.1:7860

### Error: "TypeError: Chatbot.__init__() got an unexpected keyword argument"

**Causa**: Incompatibilidad de versión de Gradio  
**Solución**: Asegúrate de tener Gradio 6.x instalado
```bash
pip install --upgrade gradio
```

### Error con TensorFlow en Python 3.12+

**Solución**: Usa Python 3.11 o crea un entorno virtual:
```bash
# Crear entorno virtual con Python específico
python -m venv venv_triaje
.\venv_triaje\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 🌐 Entornos Virtuales (Recomendado)

Para evitar conflictos de versiones entre compañeros de equipo:

```bash
# Crear entorno virtual
python -m venv venv_triaje

# Activar entorno
.\venv_triaje\Scripts\activate          # Windows PowerShell
venv_triaje\Scripts\activate.bat        # Windows CMD
source venv_triaje/bin/activate         # Mac/Linux

# Instalar dependencias en el entorno
pip install -r requirements.txt
python -m spacy download es_core_news_sm

# Desactivar cuando termines
deactivate
```

## 📞 Soporte

Para reportar problemas o sugerencias, contacta al equipo de desarrollo del proyecto SIC.

---

**Desarrollado con ❤️ para mejorar el acceso a orientación médica básica**
