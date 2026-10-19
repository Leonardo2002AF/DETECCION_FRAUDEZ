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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Fondo oscuro fijo en toda la app */
    html, body, .stApp, .main, .block-container, section.main, header[data-testid="stHeader"] {
        background-color: #0d1117 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Fuerza texto blanco / gris claro en todos los elementos */
    html, body, .stApp, p, span, label, h1, h2, h3, h4, h5, h6, li, td, th {
        color: #e6edf3;
    }

    /* Tarjetas de métricas */
    [data-testid="metric-container"], div[data-testid="stMetric"] {
        background: #161b22 !important;
        border: 1px solid #2563eb !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 0 12px rgba(37, 99, 235, 0.15) !important;
        animation: glow 3s infinite !important;
    }

    /* Valor de métrica */
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Etiqueta de métrica */
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] label, [data-testid="stMetricLabel"] p {
        color: #8b949e !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] > div, .stDataFrame, div[data-testid="stTable"] {
        background: #161b22 !important;
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }
    div[data-testid="stDataFrame"] table {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
    }

    /* Inputs, selectbox, number_input */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stSelectbox"] > div > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > input,
    div[data-testid="stTextInput"] input,
    input, select {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #e6edf3 !important;
        background-color: #21262d !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
    }

    /* Slider */
    div[data-testid="stSlider"], div[data-baseweb="slider"] {
        accent-color: #2563eb !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
    }

    /* Botones de ejemplo */
    div[data-testid="stButton"] > button,
    .stButton > button {
        background: #1d4ed8 !important;
        background-color: #1d4ed8 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stButton"] > button:hover,
    .stButton > button:hover {
        background: #2563eb !important;
        background-color: #2563eb !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
        color: white !important;
    }
    div[data-testid="stButton"] > button p,
    .stButton > button p {
        color: white !important;
        font-weight: 600 !important;
    }

    /* Toggle */
    div[data-testid="stToggle"],
    div[data-testid="stToggle"] input,
    div[data-testid="stToggle"] label {
        accent-color: #2563eb !important;
    }
    div[data-testid="stToggle"] div[aria-checked="true"],
    div[data-testid="stToggle"] span[aria-checked="true"] {
        background-color: #2563eb !important;
    }

    /* Pestañas activas */
    button[data-baseweb="tab"] {
        color: #8b949e !important;
        background: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 2px solid #2563eb !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] p {
        color: #2563eb !important;
        font-weight: 600 !important;
    }

    /* st.success */
    div[data-testid="stAlert"]:has([data-testid="stAlertSuccessIcon"]),
    div[data-testid="stAlert"]:has([aria-label="success"]),
    .stAlert:has([data-testid="stAlertSuccessIcon"]) {
        background: #0d2818 !important;
        border: 1px solid #238636 !important;
        border-radius: 12px !important;
        color: #3fb950 !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stAlertSuccessIcon"]) *,
    .stAlert:has([data-testid="stAlertSuccessIcon"]) * {
        color: #3fb950 !important;
    }

    /* st.error */
    div[data-testid="stAlert"]:has([data-testid="stAlertErrorIcon"]),
    div[data-testid="stAlert"]:has([aria-label="error"]),
    .stAlert:has([data-testid="stAlertErrorIcon"]) {
        background: #2d1117 !important;
        border: 1px solid #da3633 !important;
        border-radius: 12px !important;
        color: #f85149 !important;
    }
    div[data-testid="stAlert"]:has([data-testid="stAlertErrorIcon"]) *,
    .stAlert:has([data-testid="stAlertErrorIcon"]) * {
        color: #f85149 !important;
    }

    /* st.divider */
    hr, [data-testid="stDivider"], div[data-testid="stDivider"] hr {
        border-color: #21262d !important;
        border-top-color: #21262d !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px !important;
        height: 6px !important;
        background: #161b22 !important;
    }
    ::-webkit-scrollbar-track {
        background: #161b22 !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #2563eb !important;
        border-radius: 3px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #1d4ed8 !important;
    }

    /* Footer text */
    .custom-footer {
        color: #484f58 !important;
        font-size: 0.75rem !important;
        text-align: center !important;
        margin-top: 2.5rem;
        padding: 1.2rem 0 0.5rem 0;
        border-top: 1px solid #21262d;
    }

    /* Animaciones CSS */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(37, 99, 235, 0.3); }
        50% { box-shadow: 0 0 20px rgba(37, 99, 235, 0.7); }
    }

    .main .block-container > div,
    [data-testid="stVerticalBlock"] > div {
        animation-name: fadeInUp;
        animation-duration: 0.7s;
        animation-fill-mode: both;
    }

    [data-testid="metric-container"] {
        animation: glow 3s infinite;
    }

    /* Resultado animado */
    .fraud-alert {
        animation: pulse 1.5s ease-in-out infinite;
        border-radius: 12px;
    }
    .safe-alert {
        animation: slideInLeft 0.8s ease forwards;
        border-radius: 12px;
    }

    /* Header banner */
    .header-banner {
        padding: 24px;
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border-bottom: 1px solid #21262d;
        border-radius: 12px;
        margin-top: 8px;
        margin-bottom: 20px;
        animation: fadeIn 1.2s ease forwards;
    }

    .fraud-alert {
      animation: fraudPulse 1.5s ease-in-out infinite !important;
    }

    @keyframes fraudPulse {
      0%   { box-shadow: 0 0 0px rgba(218,54,51,0); border-color: #da3633; }
      50%  { box-shadow: 0 0 22px 6px rgba(218,54,51,0.7); border-color: #ff6b6b; }
      100% { box-shadow: 0 0 0px rgba(218,54,51,0); border-color: #da3633; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FOOTER_HTML = """
<div class="custom-footer">
    Caso 1 · Minería de Datos · Ing. en Sistemas de Información · UCE · 2026
</div>
"""

SCROLL_REVEAL_SCRIPT = """
<script>
(function(){
  const style = document.createElement('style');
  style.textContent = '.reveal{opacity:0;transform:translateY(40px);transition:opacity 0.7s ease,transform 0.7s ease} .reveal.visible{opacity:1;transform:translateY(0)}';
  document.head.appendChild(style);
  function check(){
    document.querySelectorAll('[data-testid="stVerticalBlock"] > div').forEach((el,i)=>{
      const rect = el.getBoundingClientRect();
      if(rect.top < window.innerHeight*0.92){
        el.classList.add('reveal','visible');
      } else {
        el.classList.add('reveal');
      }
    });
  }
  window.addEventListener('scroll', check, {passive:true});
  setTimeout(check, 300);
})();
</script>
"""


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
escudo_path = BASE / "uce_escudo.png"
if escudo_path.exists():
    st.image(str(escudo_path), width=80)

st.title("Detección de transacciones bancarias fraudulentas")
st.markdown(
    """
    <div class="header-banner">
        <h4 style="margin: 0; color: #e6edf3; font-weight: 700;">Detección de Transacciones Bancarias Fraudulentas</h4>
        <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 0.95rem;">
            Prueba de concepto · Minería de Datos · <strong style="color: #e6edf3;">Universidad Central del Ecuador</strong> · 2026
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
tab1, tab2, tab3 = st.tabs(["🔎 Clasificar transacción", "📊 Resultados del modelo", "ℹ️ Metodología"])

# ----------------------------------------------------------------------------- pestaña 1
with tab1:
    st.components.v1.html(SCROLL_REVEAL_SCRIPT, height=0)
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
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #2d1117, #1a0a0a);
                    border: 1px solid #da3633;
                    border-radius: 12px;
                    padding: 24px 28px;
                    margin-bottom: 16px;
                    animation: fraudPulse 1.5s ease-in-out infinite;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    min-height: 80px;
                ">
                    <span style="font-size:2rem">🚨</span>
                    <span style="color:#f85149; font-size:1.3rem; font-weight:700; line-height:1.3">
                        Transacción potencialmente fraudulenta
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #0d2818, #081a10);
                    border: 1px solid #238636;
                    border-radius: 12px;
                    padding: 24px 28px;
                    margin-bottom: 16px;
                    animation: slideInLeft 0.8s ease forwards;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    min-height: 80px;
                ">
                    <span style="font-size:2rem">✅</span>
                    <span style="color:#3fb950; font-size:1.3rem; font-weight:700; line-height:1.3">
                        Transacción legítima
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.metric("Puntaje de riesgo del modelo", f"{prob:.1%}")

        st.markdown(
            f"""
            <div style="position:relative;height:28px;background:#21262d;border-radius:14px;overflow:hidden;margin:8px 0">
              <div style="height:100%;width:{prob*100:.1f}%;background:linear-gradient(90deg,#238636,#d29922,#da3633);border-radius:14px;transition:width 0.8s ease"></div>
              <span style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#e6edf3;font-weight:700;font-size:0.9rem">{prob:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- pestaña 2
with tab2:
    st.components.v1.html(SCROLL_REVEAL_SCRIPT, height=0)
    st.markdown(
        """
        <div class="subtitle-fadein">
            <h3 style="margin-bottom: 0.2rem; color: #e6edf3;">📊 Evaluación final en el conjunto de prueba (holdout)</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("6.000 transacciones no vistas durante el entrenamiento (153 fraudulentas). Modelo: "
               "Regresión Logística con class_weight='balanced', umbral 0,5082.")
    m = st.columns(5)
    for col, (n, v) in zip(m, [("AUC-ROC", "0,8552"), ("Recall", "0,7386"), ("Precision", "0,2359"),
                               ("F1-Score", "0,3576"), ("Accuracy", "0,9323")]):
        col.metric(n, v)

    st.divider()

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

    st.divider()

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

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- pestaña 3
with tab3:
    st.components.v1.html(SCROLL_REVEAL_SCRIPT, height=0)
    st.subheader("Metodología")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
            <div class="method-card" style="
              background:#161b22;
              border:1px solid #21262d;
              border-left:4px solid #2563eb;
              border-radius:12px;
              padding:20px 24px;
              margin-bottom:16px;
              animation: fadeInUp 0.6s ease 0.0s both;
              min-height:160px;
            ">
              <div style="font-size:2rem;margin-bottom:8px">📦</div>
              <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Datos</div>
              <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">30.090 transacciones sintéticas con 13 variables. Tras eliminar 90 duplicados quedan 30.000. Desbalance: 2,55 % de fraude (769 casos).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-card" style="
              background:#161b22;
              border:1px solid #21262d;
              border-left:4px solid #d29922;
              border-radius:12px;
              padding:20px 24px;
              margin-bottom:16px;
              animation: fadeInUp 0.6s ease 0.2s both;
              min-height:160px;
            ">
              <div style="font-size:2rem;margin-bottom:8px">⚙️</div>
              <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Preparación sin fuga de datos</div>
              <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">División estratificada 80/20 antes de cualquier ajuste. Imputación, escalado y One-Hot dentro de un Pipeline ajustado solo con entrenamiento.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-card" style="
              background:#161b22;
              border:1px solid #21262d;
              border-left:4px solid #ec4899;
              border-radius:12px;
              padding:20px 24px;
              margin-bottom:16px;
              animation: fadeInUp 0.6s ease 0.4s both;
              min-height:160px;
            ">
              <div style="font-size:2rem;margin-bottom:8px">🤖</div>
              <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Modelos evaluados</div>
              <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">Regresión Logística, Árbol de Decisión y Random Forest con validación cruzada estratificada de 5 folds y métricas AUC-ROC, Recall, Precision, F1.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="method-card" style="
              background:#161b22;
              border:1px solid #21262d;
              border-left:4px solid #238636;
              border-radius:12px;
              padding:20px 24px;
              margin-bottom:16px;
              animation: fadeInUp 0.6s ease 0.1s both;
              min-height:160px;
            ">
              <div style="font-size:2rem;margin-bottom:8px">🔍</div>
              <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Análisis exploratorio (EDA)</div>
              <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">Valores faltantes (&lt;2 %), duplicados, distribuciones, valores atípicos conservados (son señal de fraude) y tasas de fraude por variable.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-card" style="
              background:#161b22;
              border:1px solid #21262d;
              border-left:4px solid #8b5cf6;
              border-radius:12px;
              padding:20px 24px;
              margin-bottom:16px;
              animation: fadeInUp 0.6s ease 0.3s both;
              min-height:160px;
            ">
              <div style="font-size:2rem;margin-bottom:8px">⚖️</div>
              <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Tratamiento del desbalance</div>
              <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">class_weight='balanced' como estrategia principal. SMOTE como alternativa dentro del pipeline. SMOTE no mejoró y se descartó.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="method-card" style="
              background:#161b22;
              border:1px solid #21262d;
              border-left:4px solid #14b8a6;
              border-radius:12px;
              padding:20px 24px;
              margin-bottom:16px;
              animation: fadeInUp 0.6s ease 0.5s both;
              min-height:160px;
            ">
              <div style="font-size:2rem;margin-bottom:8px">🎯</div>
              <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Selección del modelo final</div>
              <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">Umbral elegido con predicciones fuera de muestra (solo entrenamiento) exigiendo Recall ≥ 0,80. Ganó Regresión Logística por desempate: más simple y explicable.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="method-card" style="
          background:#161b22;
          border:1px solid #21262d;
          border-left:4px solid #da3633;
          border-radius:12px;
          padding:20px 24px;
          margin-bottom:16px;
          animation: fadeInUp 0.6s ease 0.6s both;
          min-height:160px;
        ">
          <div style="font-size:2rem;margin-bottom:8px">⚠️</div>
          <div style="color:#e6edf3;font-weight:700;font-size:1rem;margin-bottom:6px">Limitaciones</div>
          <div style="color:#8b949e;font-size:0.875rem;line-height:1.6">Dataset sintético · Solo 153 fraudes en prueba · Precision baja (≈3 falsas alarmas por fraude) · Puntaje no calibrado · Es una prueba de concepto: debe apoyar, no reemplazar, la revisión humana.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Modelo cargado desde: {origen_modelo}")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)
