# 🛡️ Detección de Transacciones Bancarias Fraudulentas

**Caso 1 — Examen Complexivo | Minería de Datos**  
Universidad Central del Ecuador · Facultad de Ingeniería y Ciencias Aplicadas  
Carrera de Ingeniería en Sistemas de Información

---

## 📋 Descripción

Prueba de concepto de un sistema de detección de fraude bancario basado en modelos de clasificación supervisada. El proyecto abarca desde la exploración de datos hasta el despliegue de un prototipo interactivo, aplicando técnicas de preparación de datos sin fuga de información, tratamiento del desbalance de clases y validación cruzada estratificada.

| Campo | Detalle |
|---|---|
| **Autor** | Araya Leonardo |
| **Componente** | Minería de Datos — Métodos Predictivos de Clasificación |
| **Fecha** | Octubre 2026 |

---

## 🗂️ Estructura del repositorio

```
DETECCION_FRAUDEZ/
├── fraude.ipynb                  # Notebook principal (análisis completo)
├── app.py                        # Aplicación web interactiva (Streamlit)
├── generador_dataset.py          # Script que genera el dataset sintético
├── transacciones_bancarias_fraude.csv   # Dataset utilizado
├── dataset_procesado.csv         # Dataset tras limpieza (sin duplicados)
├── modelo_fraude.joblib          # Modelo entrenado + umbral de decisión
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

---

## 📊 Resultados principales

| Modelo | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|
| Regresión Logística ✅ | 0,8552 | 0,7386 | 0,2359 | 0,3576 |
| Random Forest | 0,88 | 0,67 | 0,42 | 0,52 |
| Árbol de Decisión | 0,85 | 0,75 | 0,23 | 0,35 |

**Modelo final:** Regresión Logística con `class_weight='balanced'` y umbral 0,5082.  
**Criterio de selección:** maximizar Recall (minimizar fraudes no detectados) con Recall ≥ 0,80 en validación cruzada.

### Matriz de confusión (holdout — 6.000 transacciones)

|  | Pred. Legítima | Pred. Fraude |
|---|---|---|
| **Real Legítima** | ✅ VN = 5.481 | ⚠️ FP = 366 |
| **Real Fraude** | ❌ FN = 40 | ✅ VP = 113 |

---

## 🚀 Cómo ejecutar

### 1. Clonar el repositorio
```bash
git clone https://github.com/[TU-USUARIO]/DETECCION_FRAUDEZ.git
cd DETECCION_FRAUDEZ
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el notebook
Abre `fraude.ipynb` en VS Code o Jupyter y ejecuta **Restart + Run All**.

### 4. Ejecutar la aplicación web
```bash
python -m streamlit run app.py
```
Se abre en el navegador en `http://localhost:8501`.

---

## 🔬 Metodología resumida

1. **Dataset sintético** generado con `generador_dataset.py` (semilla 42, 30.090 registros, 2,56 % de fraude).
2. **EDA** univariado y bivariado: identificación de nulos, duplicados, distribuciones y patrones de fraude por variable.
3. **Preparación sin data leakage:** división estratificada 80/20 antes de cualquier ajuste; preprocesamiento dentro de pipelines.
4. **Desbalance:** `class_weight='balanced'` como estrategia principal; SMOTE descartado por no mejorar los resultados.
5. **Validación cruzada** estratificada de 5 folds con métricas AUC-ROC, Recall, Precision, F1 y Accuracy.
6. **Selección del modelo** por umbral óptimo calculado con predicciones fuera de muestra (solo entrenamiento).
7. **Evaluación final** una única vez sobre el conjunto de prueba, con intervalos de confianza por bootstrap.

---

## ⚠️ Limitaciones

- El dataset es **sintético**: las relaciones entre variables y fraude las define el generador, no comportamientos reales.
- Solo hay 153 fraudes en el conjunto de prueba, lo que genera intervalos de confianza amplios.
- Las probabilidades no están calibradas: deben usarse como puntaje de riesgo relativo.
- Es una **prueba de concepto** académica, no un sistema listo para producción.

---

## 🏷️ Hitos del proyecto

| Tag | Contenido |
|---|---|
| `hito-1-eda` | Análisis exploratorio completo |
| `hito-2-preparacion` | Preparación de datos sin data leakage |
| `hito-3-comparacion-modelos` | Validación cruzada y tabla comparativa |
| `hito-4-modelo-final` | Modelo final, holdout y matriz de confusión |
| `hito-5-entrega-final` | Predicciones, prototipo, app web y conclusiones |