from sklearn.metrics import classification_report

def generate_full_report(y_true, y_pred, target_names=None):
    """
    Genera e imprime un reporte de clasificación completo.

    :param y_true: Etiquetas verdaderas (ground truth).
    :param y_pred: Predicciones del modelo.
    :param target_names: Lista opcional de nombres de etiquetas (ej: especialidades).
    """
    print("\n--- REPORTE DE CLASIFICACIÓN DETALLADO ---")
    
    # Genera el reporte usando la función de sklearn
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names, 
        zero_division=0 # Para evitar warnings si una clase no tiene predicciones
    )
    
    print(report)
    print("------------------------------------------\n")
    return report