# TelecomX – Predicción de Churn (Customer Cancellation)

![status](https://img.shields.io/badge/status-active-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![sklearn](https://img.shields.io/badge/scikit--learn-1.x-orange)

Proyecto de *Machine Learning* para **predecir cancelación de clientes (churn)** en una telco, con enfoque en:

* pipeline reproducible de preparación de datos,
* entrenamiento de **dos modelos** (uno con normalización, otro sin),
* evaluación con métricas robustas a desbalance,
* interpretación/variables clave y **acciones de retención**.

> **Churn observado**: \~25.72% (moderado desbalance).
> **Modelos**: KNN (con escalado) y Random Forest (sin escalado).

---

## 📁 Estructura del repositorio

```
.
├── notebooks/
│   └── Telecom_2.ipynb                # Notebook principal (EDA, encoding, modelado)
├── data/
│   ├── TelecomX_Data.json             # Original expandido/curado (si aplica)
│   ├── TelecomX_Data_encoded.json     # One-Hot Encoding aplicado
│   └── TelecomX_Data_scaled.json      # Escalado (StandardScaler) para modelos sensibles
├── src/
│   └── train_models.py                # Script Colab-ready para entrenar KNN y RF
├── outputs/
│   ├── metrics_summary.json           # Métricas comparativas (test)
│   ├── knn_confusion_matrix.png
│   ├── knn_roc.png
│   ├── knn_pr.png
│   ├── rf_confusion_matrix.png
│   ├── rf_roc.png
│   ├── rf_pr.png
│   └── rf_feature_importances_top25.png
├── models/
│   ├── model_knn.joblib               # Pipeline/estimador KNN entrenado
│   └── model_random_forest.joblib     # Modelo RandomForest entrenado
├── REPORT.md                          # Informe ejecutivo de resultados y recomendaciones
└── README.md                          # Este archivo
```

> **Nota**: Las rutas pueden variar según ejecutes en **Google Colab** (`/content/...`) o localmente (`./data`, `./outputs`, etc.).

---

## 📊 Datos

* Fuente de referencia: dataset de churn tipo Telco (estructura anidada con `customer`, `phone`, `internet`, `account`).
* Pasos de preparación:

  1. **Expansión** de campos anidados (json → columnas planas).
  2. **One‑Hot Encoding** de todas las categóricas.
  3. Eliminación de **varianza cero** (ruido tras OHE masivo).
  4. **Estandarización** (solo para pipelines que lo requieren, p. ej., KNN).
* **Target**: `Churn_Yes` (binaria: 1=se va, 0=permanece).

---

## ⚡️ Quickstart

### Opción A) Google Colab

1. Sube `TelecomX_Data_encoded.json` (y `TelecomX_Data_scaled.json` si ya lo generaste) a `/content/`.
2. Abre el notebook `notebooks/Telecom_2.ipynb` o copia el script de `src/train_models.py` en una celda.
3. Ejecuta todo. Los resultados se guardarán en `/content/outputs` y los modelos en `/content/models`.

### Opción B) Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train_models.py --data data/TelecomX_Data_encoded.json \
                           --outputs outputs --models models
```

Parámetros comunes:

```bash
--data <ruta al json codificado>
--test-size 0.2
--random-state 42
--cv-splits 5
--n-estimators 400         # para RandomForest
```

---

## 🧠 Modelado

* **KNN** (con `StandardScaler` en `Pipeline`): *grid search* de `n_neighbors`, `weights`, `p` (Manhattan/Euclidiana).
* **Random Forest** (sin escalado): `class_weight='balanced'`, 400 árboles.
* División **estratificada** 80/20.
* **Métricas**: Accuracy, ROC‑AUC, PR‑AUC, Precision, Recall, F1.
* Gráficos: matrices de confusión, curvas ROC y PR; **importancias** (RF).

Ejemplo (salidas):

```json
{
  "knn": {"accuracy": 0.85, "roc_auc": 0.88, "ap": 0.62},
  "random_forest": {"accuracy": 0.87, "roc_auc": 0.91, "ap": 0.66},
  "n_features_after_vt": 3120
}
```

---

## 📈 Interpretación & Negocio

* Factores típicos asociados a mayor churn: **contrato mes a mes**, **tenure bajo**, **electronic check**, **cargos mensuales altos**, **sin tech support/online security**.
* Ver `outputs/rf_feature_importances_top25.png` para tu *Top‑N* real.
* Recomendaciones: upgrades por permanencia, onboarding proactivo, incentivos a medios de pago automáticos, bundles/paquetes, trials de servicios de soporte.

---

## 🔁 Reproducibilidad

* Fijar `random_state=42` en splits/modelos.
* Mantener **misma versión** de librerías (`requirements.txt`).
* Usar **validación estratificada** y separar **train/test** por cliente.

---

## 🧪 Predicción con modelos guardados

```python
import joblib, pandas as pd
X_nuevo = pd.read_json("data/TelecomX_Data_encoded.json").drop(columns=["Churn_Yes"], errors="ignore")
rf = joblib.load("models/model_random_forest.joblib")
proba = rf.predict_proba(X_nuevo)[:,1]  # probabilidad de churn
```

> Ajusta el **umbral** (p. ej., 0.35–0.5) según tu curva costo‑beneficio.

---

* Dataset de referencia tipo **Telco Customer Churn**.
* Comunidad de **scikit‑learn** por las herramientas de modelado y evaluación.
