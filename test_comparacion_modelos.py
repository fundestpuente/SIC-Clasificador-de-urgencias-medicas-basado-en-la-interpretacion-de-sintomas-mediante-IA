"""
Script de Comparación: Modelos Pipeline vs Pickle Separados

Este script compara objetivamente ambas implementaciones del modelo de triaje médico.
"""

import pickle
import pandas as pd
import time
from pathlib import Path

# ==================================================
# CASOS DE PRUEBA ESTÁNDAR
# ==================================================
casos_test = [
    "Paciente presenta dolor de pecho intenso, sudoración y dificultad para respirar",
    "Tengo dolor de cabeza muy fuerte, mareos y visión borrosa",
    "Fractura de brazo por caída, dolor intenso e hinchazón",
    "Dolor abdominal agudo con vómitos y fiebre",
    "Dolor en las rodillas al caminar y rigidez articular",
    "Tos persistente, fiebre y fatiga hace dos semanas",
    "Confusión mental, problemas para hablar y debilidad en un lado del cuerpo",
    "Sangrado vaginal anormal y dolor pélvico",
    "Dolor al orinar con sangre en la orina",
    "Lesión en la piel que ha crecido y cambiado de color",
    "Ansiedad severa, pensamientos intrusivos y problemas para dormir",
    "Dificultad para ver de un ojo y dolor ocular",
    "Dolor de espalda crónico que irradia a las piernas",
    "Náuseas, vómitos y heces con sangre",
    "Tumor palpable en el cuello y pérdida de peso"
]

especialidades_esperadas = [
    "CARDIOLOGÍA/CIRCULATORIO",
    "NEUROLOGÍA",
    "TRAUMATOLOGÍA/MUSCULAR",
    "GASTROENTEROLOGÍA/DIGESTIVO",
    "TRAUMATOLOGÍA/MUSCULAR",
    "CARDIOLOGÍA/CIRCULATORIO",
    "NEUROLOGÍA",
    "GINECOLOGÍA/OBSTETRICIA",
    "UROLOGÍA/RENAL",
    "ONCOLOGÍA (TUMORES)",
    "PSIQUIATRÍA/MENTAL",
    "OFTALMOLOGÍA/ORL",
    "TRAUMATOLOGÍA/MUSCULAR",
    "GASTROENTEROLOGÍA/DIGESTIVO",
    "ONCOLOGÍA (TUMORES)"
]

# ==================================================
# FUNCIÓN: CARGAR MODELO PIPELINE
# ==================================================
def cargar_modelo_pipeline():
    """Carga el modelo pipeline (versión optimizada)"""
    try:
        with open('models/modelo_triaje_svm.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        with open('models/label_encoder_final.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return pipeline, encoder, True
    except Exception as e:
        print(f" Error al cargar modelo pipeline: {e}")
        return None, None, False

# ==================================================
# FUNCIÓN: CARGAR MODELO PICKLE SEPARADO
# ==================================================
def cargar_modelo_pickle():
    """Carga el modelo con pickle separados (versión antigua)"""
    try:
        with open('models/svm_model.pickle', 'rb') as f:
            svm = pickle.load(f)
        with open('models/tfidf_vectorizer.pickle', 'rb') as f:
            tfidf = pickle.load(f)
        with open('models/label_encoder_svm.pickle', 'rb') as f:
            encoder = pickle.load(f)
        return svm, tfidf, encoder, True
    except Exception as e:
        print(f"Error al cargar modelo pickle: {e}")
        return None, None, None, False

# ==================================================
# FUNCIÓN: PREDICCIÓN CON PIPELINE
# ==================================================
def predecir_pipeline(texto, pipeline, encoder):
    """Realiza predicción usando el modelo pipeline"""
    inicio = time.time()
    pred_num = pipeline.predict([texto])[0]
    probas = pipeline.predict_proba([texto])[0]
    tiempo = time.time() - inicio
    
    especialidad = encoder.inverse_transform([pred_num])[0]
    confianza = max(probas) * 100
    
    return especialidad, confianza, tiempo

# ==================================================
# FUNCIÓN: PREDICCIÓN CON PICKLE SEPARADO
# ==================================================
def predecir_pickle(texto, svm, tfidf, encoder):
    """Realiza predicción usando modelos pickle separados"""
    inicio = time.time()
    texto_vec = tfidf.transform([texto]).toarray()
    pred_num = svm.predict(texto_vec)[0]
    probas = svm.predict_proba(texto_vec)[0]
    tiempo = time.time() - inicio
    
    especialidad = encoder.inverse_transform([pred_num])[0]
    confianza = max(probas) * 100
    
    return especialidad, confianza, tiempo

# ==================================================
# COMPARACIÓN PRINCIPAL
# ==================================================
def main():
    print("="*80)
    print(" COMPARACIÓN DE MODELOS: Pipeline vs Pickle Separados")
    print("="*80)
    
    # Cargar modelos
    print("\nCargando modelos...")
    pipeline, encoder_p, ok_pipeline = cargar_modelo_pipeline()
    svm, tfidf, encoder_s, ok_pickle = cargar_modelo_pickle()
    
    if not ok_pipeline and not ok_pickle:
        print("\n No se pudo cargar ningún modelo. Verifica que existan en /models/")
        return
    
    # Resultados
    resultados = []
    
    print(f"\n Ejecutando {len(casos_test)} casos de prueba...\n")
    print("-"*80)
    
    for i, (caso, esp_esperada) in enumerate(zip(casos_test, especialidades_esperadas), 1):
        resultado = {
            'caso': i,
            'sintomas': caso[:50] + "...",
            'esperada': esp_esperada
        }
        
        # Predicción con Pipeline
        if ok_pipeline:
            esp_p, conf_p, tiempo_p = predecir_pipeline(caso, pipeline, encoder_p)
            resultado['pipeline_pred'] = esp_p
            resultado['pipeline_conf'] = conf_p
            resultado['pipeline_tiempo'] = tiempo_p
            resultado['pipeline_correcto'] = (esp_p == esp_esperada)
        
        # Predicción con Pickle
        if ok_pickle:
            esp_s, conf_s, tiempo_s = predecir_pickle(caso, svm, tfidf, encoder_s)
            resultado['pickle_pred'] = esp_s
            resultado['pickle_conf'] = conf_s
            resultado['pickle_tiempo'] = tiempo_s
            resultado['pickle_correcto'] = (esp_s == esp_esperada)
        
        resultados.append(resultado)
        
        # Mostrar resultado
        print(f"Caso {i}: {caso[:60]}...")
        if ok_pipeline:
            check_p = "" if resultado['pipeline_correcto'] else ""
            print(f"Pipeline: {esp_p} ({conf_p:.1f}%) - {tiempo_p*1000:.2f}ms {check_p}")
        if ok_pickle:
            check_s = "" if resultado['pickle_correcto'] else ""
            print(f"Pickle:   {esp_s} ({conf_s:.1f}%) - {tiempo_s*1000:.2f}ms {check_s}")
        print()
    
    print("="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    
    # Estadísticas Pipeline
    if ok_pipeline:
        aciertos_p = sum(r['pipeline_correcto'] for r in resultados)
        precision_p = (aciertos_p / len(resultados)) * 100
        conf_prom_p = sum(r['pipeline_conf'] for r in resultados) / len(resultados)
        tiempo_prom_p = sum(r['pipeline_tiempo'] for r in resultados) / len(resultados) * 1000
        
        print(f"\n MODELO PIPELINE:")
        print(f"   ├─ Precisión: {aciertos_p}/{len(resultados)} ({precision_p:.1f}%)")
        print(f"   ├─ Confianza promedio: {conf_prom_p:.1f}%")
        print(f"   └─ Tiempo promedio: {tiempo_prom_p:.2f} ms")
    
    # Estadísticas Pickle
    if ok_pickle:
        aciertos_s = sum(r['pickle_correcto'] for r in resultados)
        precision_s = (aciertos_s / len(resultados)) * 100
        conf_prom_s = sum(r['pickle_conf'] for r in resultados) / len(resultados)
        tiempo_prom_s = sum(r['pickle_tiempo'] for r in resultados) / len(resultados) * 1000
        
        print(f"\nMODELO PICKLE SEPARADO:")
        print(f"   ├─ Precisión: {aciertos_s}/{len(resultados)} ({precision_s:.1f}%)")
        print(f"   ├─ Confianza promedio: {conf_prom_s:.1f}%")
        print(f"   └─ Tiempo promedio: {tiempo_prom_s:.2f} ms")
    
    # Comparación directa
    if ok_pipeline and ok_pickle:
        print(f"\nGANADOR:")
        if precision_p > precision_s:
            print(f"Pipeline es {precision_p - precision_s:.1f}% más preciso")
        elif precision_s > precision_p:
            print(f"Pickle es {precision_s - precision_p:.1f}% más preciso")
        else:
            print(f"Ambos tienen la misma precisión")
        
        if tiempo_prom_p < tiempo_prom_s:
            mejora = ((tiempo_prom_s - tiempo_prom_p) / tiempo_prom_s) * 100
            print(f" Pipeline es {mejora:.1f}% más rápido")
        else:
            mejora = ((tiempo_prom_p - tiempo_prom_s) / tiempo_prom_p) * 100
            print(f" Pickle es {mejora:.1f}% más rápido")
    
    # Guardar resultados en CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('resultados_comparacion_modelos.csv', index=False)
    print(f"\nResultados guardados en: resultados_comparacion_modelos.csv")
    print("="*80)

if __name__ == "__main__":
    main()
