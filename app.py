"""Prototipo web: detección de transacciones bancarias fraudulentas (Caso 1 - Minería de Datos).
Ejecutar con:  streamlit run app.py"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
UMBRAL_DEFECTO = 0.5082  # umbral elegido con predicciones fuera de muestra (solo train)

st.set_page_config(page_title="Detección de fraude bancario", page_icon="🛡️", layout="wide")


# ----------------------------------------------------------------------------- modelo
def reentrenar():
    """Plan B: si el .joblib no es compatible con las librerías instaladas, se reentrena
    el mismo modelo (Regresión Logística balanceada) con dataset_procesado.csv."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(BASE / "dataset_procesado.csv", encoding="utf-8-sig").drop_duplicates()
    X, y = df.drop(columns=["id_transaccion", "fraude"]), df["fraude"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num = ["monto", "hora_transaccion", "distancia_ubicacion", "frecuencia_transacciones",
           "monto_promedio_cliente", "antiguedad_cliente_meses"]
    cat = ["tipo_transaccion", "canal", "metodo_autenticacion"]
    binarias = ["dispositivo_nuevo", "ubicacion_nueva"]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("bin", "passthrough", binarias)])
    modelo = Pipeline([("prep", prep), ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    return modelo.fit(X_tr, y_tr)


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    try:
        paquete = joblib.load(BASE / "modelo_fraude.joblib")
        return paquete["modelo"], float(paquete["umbral"]), "modelo_fraude.joblib"
    except Exception:
        return reentrenar(), UMBRAL_DEFECTO, "reentrenado con dataset_procesado.csv"


modelo, umbral, origen_modelo = cargar_modelo()

# ----------------------------------------------------------------------------- ejemplos
EJEMPLOS = {
    "legitima": dict(monto=45.9, hora=14, distancia=2.5, frecuencia=2, promedio=60.0, disp=False,
                     ubic=False, tipo="Compra", canal="POS", metodo="Tarjeta", antig=48),
    "sospechosa": dict(monto=950.0, hora=2, distancia=180.0, frecuencia=8, promedio=70.0, disp=True,
                       ubic=True, tipo="Transferencia", canal="Web", metodo="Clave", antig=6),
}


def cargar_ejemplo(nombre):
    for k, v in EJEMPLOS[nombre].items():
        st.session_state[k] = v


for k, v in EJEMPLOS["legitima"].items():
    st.session_state.setdefault(k, v)

# ----------------------------------------------------------------------------- cabecera
st.title("🛡️ Detección de transacciones bancarias fraudulentas")
st.caption("Prueba de concepto · Minería de Datos · Métodos predictivos de clasificación · "
           "Universidad Central del Ecuador")
tab1, tab2, tab3 = st.tabs(["🔎 Clasificar transacción", "📊 Resultados del modelo", "ℹ️ Metodología"])

# ----------------------------------------------------------------------------- pestaña 1
with tab1:
    izq, der = st.columns([3, 2], gap="large")
    with izq:
        st.subheader("Datos de la transacción")
        b1, b2, _ = st.columns([1, 1, 2])
        b1.button("Cargar ejemplo habitual", on_click=cargar_ejemplo, args=("legitima",))
        b2.button("Cargar ejemplo sospechoso", on_click=cargar_ejemplo, args=("sospechosa",))

        c1, c2 = st.columns(2)
        monto = c1.number_input("Monto (USD)", min_value=0.0, step=10.0, key="monto")
        promedio = c2.number_input("Monto promedio del cliente (USD)", min_value=0.0, step=10.0, key="promedio")
        hora = c1.slider("Hora de la transacción", 0, 23, key="hora")
        distancia = c2.number_input("Distancia a la ubicación habitual (km)", min_value=0.0, step=5.0, key="distancia")
        frecuencia = c1.number_input("Transacciones en las últimas 24 h", min_value=0, max_value=50, step=1, key="frecuencia")
        antig = c2.number_input("Antigüedad del cliente (meses)", min_value=0, max_value=600, step=1, key="antig")
        tipo = c1.selectbox("Tipo de transacción", ["Compra", "Pago", "Transferencia", "Retiro"], key="tipo")
        canal = c2.selectbox("Canal", ["Aplicación móvil", "Cajero", "Web", "POS"], key="canal")
        metodo = c1.selectbox("Método de autenticación", ["Clave", "Biometría", "Token", "Tarjeta"], key="metodo")
        disp = c1.toggle("Dispositivo nuevo", key="disp")
        ubic = c2.toggle("Ubicación nueva", key="ubic")

    fila = pd.DataFrame([{
        "monto": monto, "hora_transaccion": hora, "distancia_ubicacion": distancia,
        "frecuencia_transacciones": frecuencia, "monto_promedio_cliente": promedio,
        "dispositivo_nuevo": int(disp), "ubicacion_nueva": int(ubic), "tipo_transaccion": tipo,
        "canal": canal, "metodo_autenticacion": metodo, "antiguedad_cliente_meses": antig}])
    prob = float(modelo.predict_proba(fila)[0, 1])

    with der:
        st.subheader("Resultado")
        if prob >= umbral:
            st.error("### 🚨 Transacción potencialmente fraudulenta")
        else:
            st.success("### ✅ Transacción legítima")
        st.metric("Puntaje de riesgo del modelo", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f"Umbral de decisión: {umbral:.4f}. El puntaje no es una probabilidad calibrada: "
                   "sirve para ordenar transacciones por riesgo.")

        try:  # explicabilidad: aporte de cada variable (válido para Regresión Logística)
            x = modelo[:-1].transform(fila)
            x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
            aportes = pd.Series(modelo[-1].coef_[0] * x[0],
                                index=[n.split("__", 1)[-1] for n in modelo[:-1].get_feature_names_out()])
            top = aportes.reindex(aportes.abs().sort_values(ascending=False).index).head(6)
            st.markdown("**Factores que más influyen** (positivo = sube el riesgo)")
            st.bar_chart(top.sort_values(), horizontal=True)
        except Exception:
            pass

# ----------------------------------------------------------------------------- pestaña 2
with tab2:
    st.subheader("Evaluación final en el conjunto de prueba (holdout)")
    st.caption("6.000 transacciones no vistas durante el entrenamiento (153 fraudulentas). Modelo: "
               "Regresión Logística con class_weight='balanced', umbral 0,5082.")
    m = st.columns(5)
    for col, (n, v) in zip(m, [("AUC-ROC", "0,8552"), ("Recall", "0,7386"), ("Precision", "0,2359"),
                               ("F1-Score", "0,3576"), ("Accuracy", "0,9323")]):
        col.metric(n, v)

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("**Matriz de confusión**")
        cm = np.array([[5481, 366], [40, 113]])
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        etq = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{etq[i][j]}\n{cm[i, j]:,}\n({pct:.1%})", ha="center", va="center",
                        color="white" if pct > 0.5 else "black", fontsize=11)
        ax.set_xticks([0, 1], ["Legítima", "Fraude"])
        ax.set_yticks([0, 1], ["Legítima", "Fraude"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
        st.pyplot(fig)
        st.caption("Detecta 113 de 153 fraudes (73,9 %), deja pasar 40 y genera 366 falsas alarmas (6,3 % de las legítimas).")
    with g2:
        st.markdown("**Comparación de modelos (validación cruzada estratificada, 5 folds)**")
        cv = pd.DataFrame({
            "AUC-ROC": [0.8837, 0.8851, 0.8530, 0.8844, 0.8837],
            "Recall": [0.8007, 0.7958, 0.7533, 0.6700, 0.5850],
            "Precision": [0.2663, 0.2449, 0.2315, 0.4236, 0.4222],
            "F1": [0.3996, 0.3744, 0.3539, 0.5179, 0.4891],
            "Accuracy": [0.9386, 0.9321, 0.9296, 0.9681, 0.9688]},
            index=["Reg. Logística", "Reg. Logística + SMOTE", "Árbol de Decisión",
                   "Random Forest", "Random Forest + SMOTE"])
        st.dataframe(cv.style.format("{:.4f}").highlight_max(axis=0, color="#1f6f4a"), width="stretch")
        st.caption("Accuracy no sirve para decidir: clasificar todo como legítimo ya daría 97,45 %.")

    st.subheader("Qué distingue al fraude en los datos (tasa de fraude por categoría, %)")
    e = st.columns(3)
    datos = {
        "Ubicación nueva": {"No": 1.12, "Sí": 17.63},
        "Dispositivo nuevo": {"No": 1.52, "Sí": 13.72},
        "Tipo de transacción": {"Pago": 0.73, "Compra": 1.83, "Retiro": 5.25, "Transferencia": 5.60},
        "Canal": {"POS": 1.06, "App móvil": 2.42, "Cajero": 4.01, "Web": 4.46},
        "Autenticación": {"Biometría": 1.06, "Tarjeta": 2.32, "Token": 2.56, "Clave": 4.60}}
    for i, (nombre, d) in enumerate(datos.items()):
        with e[i % 3]:
            st.markdown(f"**{nombre}**")
            st.bar_chart(pd.Series(d, name="Tasa de fraude (%)"))
    st.caption("Tasa global de fraude: 2,55 %. Además, el fraude se concentra en la madrugada (≈35 % a las 0 h).")

# ----------------------------------------------------------------------------- pestaña 3
with tab3:
    st.subheader("Metodología")
    st.markdown("""
1. **Datos:** 30.090 transacciones sintéticas con las 13 variables del diccionario del caso; tras eliminar 90 duplicados, 30.000. Desbalance: 2,55 % de fraude.
2. **EDA:** valores faltantes (<2 %), duplicados, distribuciones, valores atípicos (se conservan: son señal de fraude) y análisis de fraude por variable.
3. **Preparación sin fuga de datos:** división estratificada 80/20 **antes** de cualquier ajuste; imputación, escalado y One-Hot dentro de un `Pipeline` que se ajusta solo con entrenamiento.
4. **Desbalance:** `class_weight="balanced"` como estrategia principal; SMOTE (dentro del pipeline) como alternativa. No mejoró.
5. **Modelos y validación:** Regresión Logística, Árbol de Decisión y Random Forest con validación cruzada estratificada de 5 folds.
6. **Selección:** umbral elegido con predicciones fuera de muestra (solo entrenamiento) exigiendo Recall ≥ 0,80; ganó la Regresión Logística por desempate (más simple y explicable).
7. **Evaluación final:** una sola vez sobre el conjunto de prueba, con intervalos de confianza por bootstrap (Recall 95 %: 0,67–0,80).
""")
    st.warning("**Limitaciones:** el dataset es **sintético** (las relaciones las define un generador), "
               "solo hay 153 fraudes en el conjunto de prueba, la Precision es baja (≈ 3 falsas alarmas por cada "
               "fraude detectado) y el puntaje no es una probabilidad calibrada. Es una prueba de concepto: en un "
               "sistema real debe apoyar —no reemplazar— la revisión humana.")
    st.caption(f"Modelo cargado desde: {origen_modelo}")
