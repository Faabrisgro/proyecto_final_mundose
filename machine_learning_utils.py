from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def make_prediction(model, X_train, y_train, X_test):
    """
    Realiza la predicción utilizando un modelo dado.

    Parámetros:
    - model: El modelo de clasificación entrenado.
    - X_train: Datos de entrenamiento.
    - y_train: Etiquetas de entrenamiento.
    - X_test: Datos de prueba.

    Devuelve:
    - predictions: Las predicciones del modelo en los datos de prueba.
    """
    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Realizar predicciones en los datos de prueba
    predictions = model.predict(X_test)

    return predictions

def verify_results(y_true, y_pred):
    """
    Muestra un mapa de calor de la matriz de confusión y otros resultados de clasificación.

    Parámetros:
    - y_true: Etiquetas reales.
    - y_pred: Predicciones del modelo.
    """
    # Calcular la matriz de confusión
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Mostrar otros resultados de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred))

    # Crear un mapa de calor de la matriz de confusión utilizando Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Clase {}".format(i) for i in range(confusion_mat.shape[0])],
                yticklabels=["Clase {}".format(i) for i in range(confusion_mat.shape[0])])
    plt.xlabel("Predicciones")
    plt.ylabel("Etiquetas reales")
    plt.title("Matriz de Confusión")
    plt.show()