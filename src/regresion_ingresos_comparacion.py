# ============================================================
# Regresión lineal para predecir ingreso mensual
# Compara particiones 80/20, 90/10 y 70/30.
#
# Instalación:
#   pip install pandas openpyxl scikit-learn statsmodels matplotlib seaborn
#
# Ejecución:
#   Coloca data.xlsx junto a este script y ejecuta:
#   python regresion_ingresos_comparacion.py
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- CONFIGURACIÓN --------------------
ARCHIVO = Path("data/data.xlsx") if Path("data/data.xlsx").exists() else Path("data.xlsx")
HOJA = "Sheet1"
OBJETIVO = "ingreso_mensual"
PREDICTORES = ["edad", "anios_esc", "nivel_edu", "sex"]
SEMILLA = 42
PARTICIONES = {"80/20": 0.20, "90/10": 0.10, "70/30": 0.30}
SALIDA = Path("resultados_regresion")
SALIDA.mkdir(exist_ok=True)

# -------------------- CARGA Y VALIDACIÓN --------------------
if not ARCHIVO.exists():
    raise FileNotFoundError(
        f"No se encontró {ARCHIVO.resolve()}. Coloca data.xlsx junto al script."
    )

df = pd.read_excel(ARCHIVO, sheet_name=HOJA)
df.columns = df.columns.astype(str).str.strip()

columnas_requeridas = ["edad", "anios_esc", "nivel_edu", "sex", OBJETIVO]
faltantes = [c for c in columnas_requeridas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas requeridas: {faltantes}")

for c in ["edad", "anios_esc", OBJETIVO]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
n_original = len(df)
df = df.dropna(subset=[OBJETIVO]).copy()

# Las categorías se mantienen como texto; los valores ausentes se imputan
# dentro del pipeline, que se ajusta únicamente con el conjunto de entrenamiento.
for c in ["nivel_edu", "sex"]:
    df[c] = df[c].astype(object)
    df.loc[pd.isna(df[c]), c] = np.nan

X = df[PREDICTORES]
y = df[OBJETIVO]

print(f"Registros originales: {n_original:,}")
print(f"Registros con ingreso válido: {len(df):,}")
print(f"Registros excluidos por ingreso faltante/no válido: {n_original-len(df):,}")
print(f"Valores faltantes por columna después de filtrar ingreso:\n{df[PREDICTORES + [OBJETIVO]].isna().sum()}")

def crear_modelo():
    """Preprocesamiento y regresión dentro de un pipeline para evitar fuga de datos."""
    numericas = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categoricas = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", numericas, ["edad", "anios_esc"]),
            ("categoricas", categoricas, ["nivel_edu", "sex"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[
        ("preprocesar", preprocesador),
        ("regresion", LinearRegression()),
    ])

# -------------------- COMPARACIÓN DE PARTICIONES --------------------
filas = []
resultados_particion = {}

for nombre, test_size in PARTICIONES.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEMILLA
    )
    modelo = crear_modelo()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)

    fila = {
        "Partición": nombre,
        "Fracción prueba": test_size,
        "n_entrenamiento": len(X_train),
        "n_prueba": len(X_test),
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    }
    filas.append(fila)
    resultados_particion[nombre] = (modelo, X_train, X_test, y_train, y_test, pred)

comparacion = pd.DataFrame(filas)
comparacion.to_csv(SALIDA / "comparacion_particiones.csv", index=False)

print("\n=== COMPARACIÓN DE PARTICIONES ===")
print(comparacion.to_string(index=False, formatters={
    "Fracción prueba": "{:.2f}".format,
    "R2": "{:.4f}".format,
    "MAE": "{:,.2f}".format,
    "RMSE": "{:,.2f}".format,
}))

# Gráfica de R² y tabla/CSV de las métricas
sns.set_theme()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(comparacion["Partición"], comparacion["R2"])
ax.set_title("Comparación del R² en el conjunto de prueba")
ax.set_xlabel("Partición entrenamiento/prueba")
ax.set_ylabel("R²")
ax.set_ylim(0, max(comparacion["R2"]) * 1.15)
for i, valor in enumerate(comparacion["R2"]):
    ax.text(i, valor, f"{valor:.3f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(SALIDA / "comparacion_r2.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# -------------------- MODELO SIMPLE VS. MÚLTIPLE EN 80/20 --------------------
# Se mantiene la partición 80/20 como análisis principal.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEMILLA
)
modelo_multiple = crear_modelo()
modelo_multiple.fit(X_train, y_train)
pred_multiple = modelo_multiple.predict(X_test)

# Modelo de referencia que utiliza únicamente edad.
X_edad = df[["edad"]]
Xedad_train, Xedad_test, yedad_train, yedad_test = train_test_split(
    X_edad, y, test_size=0.20, random_state=SEMILLA
)
modelo_edad = Pipeline(steps=[
    ("preprocesar", ColumnTransformer(
        transformers=[(
            "numerica",
            Pipeline(steps=[
                ("imputar", SimpleImputer(strategy="median")),
                ("escalar", StandardScaler()),
            ]),
            ["edad"],
        )],
        remainder="drop",
    )),
    ("regresion", LinearRegression()),
])
modelo_edad.fit(Xedad_train, yedad_train)
pred_edad = modelo_edad.predict(Xedad_test)

metricas_modelos = pd.DataFrame([
    {
        "Modelo": "Solo edad",
        "R2": r2_score(yedad_test, pred_edad),
        "MAE": mean_absolute_error(yedad_test, pred_edad),
        "RMSE": np.sqrt(mean_squared_error(yedad_test, pred_edad)),
    },
    {
        "Modelo": "Regresión múltiple",
        "R2": r2_score(y_test, pred_multiple),
        "MAE": mean_absolute_error(y_test, pred_multiple),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_multiple)),
    },
])
metricas_modelos.to_csv(SALIDA / "metricas_modelos_80_20.csv", index=False)
print("\n=== SOLO EDAD VS. REGRESIÓN MÚLTIPLE (80/20) ===")
print(metricas_modelos.to_string(index=False, formatters={
    "R2": "{:.4f}".format,
    "MAE": "{:,.2f}".format,
    "RMSE": "{:,.2f}".format,
}))

# Guardar predicciones del modelo múltiple 80/20.
predicciones = pd.DataFrame({
    "ingreso_real": y_test,
    "ingreso_predicho": pred_multiple,
    "residuo": y_test.to_numpy() - pred_multiple,
}, index=y_test.index)
predicciones.to_csv(SALIDA / "predicciones_test_80_20.csv", index=True)

# Diagnóstico visual básico del modelo múltiple 80/20.
residuos = y_test.to_numpy() - pred_multiple
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(pred_multiple, residuos, alpha=0.35)
axes[0].axhline(0, linestyle="--")
axes[0].set_title("Residuos frente al ingreso predicho")
axes[0].set_xlabel("Ingreso predicho")
axes[0].set_ylabel("Residuo (real - predicho)")

axes[1].scatter(y_test, pred_multiple, alpha=0.35)
lim_min = min(float(y_test.min()), float(np.min(pred_multiple)))
lim_max = max(float(y_test.max()), float(np.max(pred_multiple)))
axes[1].plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
axes[1].set_title("Ingreso real frente a predicho")
axes[1].set_xlabel("Ingreso real")
axes[1].set_ylabel("Ingreso predicho")
fig.tight_layout()
fig.savefig(SALIDA / "diagnostico_80_20.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"\nArchivos guardados en: {SALIDA.resolve()}")
print("- comparacion_particiones.csv")
print("- comparacion_r2.png")
print("- metricas_modelos_80_20.csv")
print("- predicciones_test_80_20.csv")
print("- diagnostico_80_20.png")
