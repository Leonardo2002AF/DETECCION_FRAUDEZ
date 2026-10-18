"""Generador de dataset sintético: transacciones bancarias fraudulentas.
Respeta exactamente el diccionario de datos del Caso 1. Semilla fija = reproducible."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 30000
TASA_FRAUDE = 0.025

fraude = (rng.random(N) < TASA_FRAUDE).astype(int)
# 20% de los fraudes se "camuflan" (parecen legítimos) y 3% de los legítimos
# parecen sospechosos: evita separación perfecta y hace el problema realista
sospechoso = np.where(fraude == 1, rng.random(N) > 0.20, rng.random(N) < 0.03)
f = sospechoso

def elegir(opc, p_leg, p_sosp):
    out = np.empty(N, dtype=object)
    out[~f] = rng.choice(opc, (~f).sum(), p=p_leg)
    out[f] = rng.choice(opc, f.sum(), p=p_sosp)
    return out

antig = np.clip(rng.gamma(2.5, 22, N) * np.where(f, 0.6, 1.0), 1, 240).round().astype(int)
monto_prom = np.clip(rng.lognormal(4.2, 0.6, N), 5, 2000).round(2)
mult = np.where(f, rng.lognormal(1.2, 0.8, N), rng.lognormal(0.0, 0.55, N))
monto = np.clip(monto_prom * mult, 1, 5000).round(2)

hora = np.where(
    f,
    np.where(rng.random(N) < 0.45, rng.integers(0, 6, N), rng.integers(0, 24, N)),
    np.clip(rng.normal(14, 4.5, N), 0, 23).round().astype(int),
)
dist = np.clip(np.where(f, rng.exponential(65, N), rng.exponential(9, N)), 0, 500).round(2)
frec = np.clip(np.where(f, rng.poisson(6, N), rng.poisson(2, N)), 0, 20)
disp_nuevo = (rng.random(N) < np.where(f, 0.55, 0.06)).astype(int)
ubic_nueva = (rng.random(N) < np.where(f, 0.6, 0.05) + 0.25 * (dist > 40)).clip(0, 1).astype(int)

tipo = elegir(["Compra", "Pago", "Transferencia", "Retiro"], [.50, .25, .15, .10], [.30, .05, .40, .25])
canal = elegir(["Aplicación móvil", "Cajero", "Web", "POS"], [.35, .08, .22, .35], [.30, .15, .45, .10])
metodo = elegir(["Clave", "Biometría", "Token", "Tarjeta"], [.30, .30, .15, .25], [.50, .05, .10, .35])
# coherencia básica: POS y Cajero usan mayormente tarjeta
m_pos = (canal == "POS") & (rng.random(N) < 0.85)
m_caj = (canal == "Cajero") & (rng.random(N) < 0.70)
metodo[m_pos | m_caj] = "Tarjeta"

df = pd.DataFrame({
    "id_transaccion": [f"TX{i:06d}" for i in range(1, N + 1)],
    "monto": monto, "hora_transaccion": hora, "distancia_ubicacion": dist,
    "frecuencia_transacciones": frec, "monto_promedio_cliente": monto_prom,
    "dispositivo_nuevo": disp_nuevo, "ubicacion_nueva": ubic_nueva,
    "tipo_transaccion": tipo, "canal": canal, "metodo_autenticacion": metodo,
    "antiguedad_cliente_meses": antig, "fraude": fraude,
})

# Imperfecciones para que el EDA tenga qué encontrar
for col, p in [("monto_promedio_cliente", .015), ("distancia_ubicacion", .01), ("metodo_autenticacion", .008)]:
    df.loc[rng.random(N) < p, col] = np.nan
dup = df.sample(int(N * 0.003), random_state=1)
df = pd.concat([df, dup]).sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("transacciones_bancarias_fraude.csv", index=False, encoding="utf-8-sig")
print(df.shape, "| fraude:", round(df.fraude.mean() * 100, 2), "%")
