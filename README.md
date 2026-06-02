# Clasificador de Diabetes — Redes Neuronales con PyTorch

Este repositorio contiene la implementación completa de un sistema de aprendizaje profundo para predecir si un paciente padece diabetes basándose en métricas clínicas de entrada. El proyecto abarca un pipeline completo de Machine Learning (ML), desde el análisis exploratorio preliminar y el balanceo de clases, hasta el benchmarking exhaustivo de **10 arquitecturas de Redes Neuronales (Modelos A–J)** utilizando **PyTorch**.

El código original está estructurado y optimizado en forma de libreta interactiva o script de ejecución monolítica para entornos de cómputo acelerado (como Google Colab).

---

## Arquitectura General del Proyecto

El flujo de trabajo se divide rigurosamente en 7 bloques secuenciales y modulares:

* **Bloque 0 — EDA (Exploratory Data Analysis):** Análisis estadístico descriptivo (`.describe()`), visualización de distribuciones mediante histogramas y mapeo de relaciones mediante una matriz de correlación de Pearson (`seaborn.heatmap`) para identificar los predictores clínicos de mayor peso.
* **Bloque 1 — Preparación de Datos:** 
  * Carga de datos desde un archivo Excel (`diab.xlsx`).
  * Análisis visual del **desbalance de clases** y justificación del uso de métricas alternativas al Accuracy (F1-Score, AUC-ROC).
  * Partición estratificada del dataset en **70% Entrenamiento, 15% Validación y 15% Prueba** (`train_test_split(stratify=y)`).
  * Normalización estadística (Z-Score) mediante `StandardScaler` ajustado exclusivamente con datos de entrenamiento para prevenir el filtrado de información (*data leakage*).
  * Conversión a tensores de PyTorch y empaquetado optimizado en objetos `TensorDataset` y `DataLoader`.
* **Bloque 2 — Arquitectura de Red y Motor de Entrenamiento:** Definición de una clase de red neuronal totalmente dinámica (`RedDiabetes`) que hereda de `nn.Module`. Permite definir el número de capas ocultas, cantidad de neuronas por capa y funciones de activación de manera paramétrica. Implementa bucles robustos de optimización con **Early Stopping** (basado en la pérdida de validación con paciencia de 20 épocas) y cálculo automático de métricas.
* **Bloque 3 y 3B — Benchmarking de Modelos:** Entrenamiento secuencial de los 10 modelos base (A–J) definidos por el PDF guía y su contraparte directa **aplicando regularización $L_2$ (Weight Decay)**.
* **Bloque 4 y 4B — Visualización y Reporte:** Generación automatizada de curvas de pérdida (learning curves), mapas de calor para matrices de confusión individuales y comparación de curvas ROC individuales y consolidadas.
* **Bloque 5 — Análisis Clínico y Selección:** Consolidación de tablas comparativas y lógica interna de selección de candidatos óptimos según el criterio médico/técnico.
* **Bloque 6 — Sensibilidad de Hiperparámetros y Robustez:** Análisis del impacto del tamaño de lote (Batch Size $\in [4, 8, 16, 32, 64]$) y validación cruzada estratificada (**K-Fold Cross Validation**, $K=5$) con cálculo de intervalos de confianza al 95%.

---

##  Benchmarking de Modelos (Configuraciones A–J)

El proyecto evalúa empíricamente cómo impactan la normalización, las funciones de activación (`ReLU`, `Tanh`, `LeakyReLU`, `ELU`), la profundidad de la red y la tasa de aprendizaje (`learning_rate`) en la métrica final:

| Modelo | Normalizado | Estructura Oculta | Activación | Learning Rate | Regularización $L_2$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | No | `[16]` | `ReLU` | $0.01$ | $0.0$ o $10^{-4}$ |
| **B** | No | `[32, 16]` | `ReLU` | $0.001$ | $0.0$ o $10^{-4}$ |
| **C** | No | `[64, 32]` | `Tanh` | $0.001$ | $0.0$ o $10^{-4}$ |
| **D** | No | `[128, 64, 32]` | `ReLU` | $0.0001$ | $0.0$ o $10^{-4}$ |
| **E** | Sí | `[16]` | `ReLU` | $0.01$ | $0.0$ o $10^{-4}$ |
| **F** | Sí | `[32, 16]` | `ReLU` | $0.001$ | $0.0$ o $10^{-4}$ |
| **G** | Sí | `[64, 32]` | `LeakyReLU` | $0.001$ | $0.0$ o $10^{-4}$ |
| **H** | Sí | `[128, 64, 32]` | `ELU` | $0.0001$ | $0.0$ o $10^{-4}$ |
| **I** | Sí | `[64, 32, 16, 8]` | `ReLU` | $0.0005$ | $0.0$ o $10^{-4}$ |
| **J** | Sí | `[32]` | `Tanh` | $0.01$ | $0.0$ o $10^{-4}$ |

---

##  Aspectos Técnicos Relevantes

### Estabilidad Numérica y Funciones de Pérdida
En lugar de añadir una función de activación sigmoide en la última capa de la red, la arquitectura de la clase `RedDiabetes` expone directamente los **logits**. El motor de entrenamiento utiliza la función de pérdida **`nn.BCEWithLogitsLoss`**. Esto asegura una óptima estabilidad numérica al procesar los gradientes de la clasificación binaria (evitando subdesbordamientos o desbordamientos matemáticos que ocurren comúnmente al calcular de forma aislada $\log(	ext{Sigmoid}(x))$).

### Criterio Clínico de Evaluación (Trade-off Sensibilidad vs Especificidad)
En un clasificador de diagnóstico de enfermedades, los costes de cometer un error son asimétricos:
* **Falso Positivo (FP):** Clasificar a un paciente sano como diabético. Causa ansiedad y requiere pruebas de confirmación adicionales, pero no pone en riesgo la vida.
* **Falso Negativo (FN):** Clasificar a un paciente enfermo como sano. Evita que el paciente reciba un tratamiento oportuno, desencadenando complicaciones clínicas graves.

Por consiguiente, el proyecto prioriza el **Recall (Sensibilidad)** sobre la Precisión estricta o el Accuracy general, maximizando la tasa de verdaderos positivos (TPR) sin comprometer drásticamente la especificidad.

---

##  Requisitos e Instalación

Para replicar este pipeline, asegúrate de contar con un entorno de Python 3.8+ e instalar los siguientes paquetes fundamentales:

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### Configuración de la Ruta del Dataset
El código asume una integración con Google Drive para entornos Google Colab. Si deseas ejecutarlo de forma local, simplemente remueve las líneas de montura de almacenamiento e indica la ruta local del archivo `.xlsx`:

```python
# Versión original en Colab:
# df = pd.read_excel('/content/drive/MyDrive/Proyecto_Diabetes/diab.xlsx')

# Modificación para entorno local:
df = pd.read_excel('datos/diab.xlsx')
```

---

##  Conclusiones Clave del Experimento

1. **Impacto Crítico de la Normalización:** Los modelos entrenados con datos crudos sin normalizar (**A–D**) exhiben un comportamiento inestable en sus gradientes y requieren un mayor número de épocas para converger. Los modelos normalizados (**E–J**) logran converger más rápido debido a que eliminan las discrepancias de magnitudes métricas entre variables (por ejemplo, los valores absolutos de Glucosa vs. Edad).
2. **Mitigación del Sobreajuste:** La inclusión de una penalización regularizadora $L_2$ (`weight_decay=1e-4`) suaviza las fronteras de decisión y optimiza significativamente las métricas en el conjunto de datos de Prueba, disminuyendo la brecha típica entre las pérdidas de entrenamiento y validación.
3. **Robustez mediante Validación Cruzada:** El bloque final de **K-Fold** remueve el sesgo inherente de una sola partición aleatoria, proveyendo medias e intervalos de confianza estadísticamente representativos del verdadero desempeño del modelo candidato.
