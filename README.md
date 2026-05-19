# 📈 PFAVAL Stock Analysis — Regresión Lineal Múltiple

> Análisis econométrico del precio de la acción **Preferencial Grupo Aval (PFAVAL)** en la Bolsa de Valores de Colombia, usando cinco variables macroeconómicas como regresoras (2020–2025).

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Objetivo

Modelar el precio de cierre diario de PFAVAL (en COP) mediante regresión lineal múltiple, identificando y cuantificando el efecto de cinco variables macroeconómicas representativas del entorno colombiano e internacional.

---

## 📐 Variables del Modelo

| Variable | Ticker / Fuente | Rol en el modelo | Correlación esperada |
|----------|----------------|-----------------|----------------------|
| **PFAVAL** (COP) | `PFAVAL.CL` — BVC | Variable dependiente (Y) | — |
| **TRM** (COP/USD) | `COP=X` — SFC Colombia | Regresora X₁ | Negativa |
| **WTI** (USD/barril) | `CL=F` — EIA/NYMEX | Regresora X₂ | Positiva |
| **TPM BanRep** (%) | Banco de la República | Regresora X₃ | Negativa |
| **COLCAP** (puntos) | `^COLCAP` — BVC | Regresora X₄ | Positiva |
| **VIX** (puntos) | `^VIX` — CBOE | Regresora X₅ | Negativa |

### ¿Por qué estas variables?

- **TRM**: El precio del dólar frente al peso es el mayor driver de incertidumbre macroeconómica en Colombia. Dólar fuerte → peso débil → salida de capitales de emergentes → PFAVAL cae.

- **WTI**: Colombia es exportador de petróleo. WTI alto → mejora balanza comercial → COP se aprecia → sector financiero se beneficia por mayor actividad crediticia.

- **TPM BanRep**: El ciclo de tasas 2020–2023 (mínimo 1.75% → máximo 13.25%) fue el más dramático en décadas. Tasas altas comprimen la valuación de acciones por descuento de flujos.

- **COLCAP**: Índice bursátil de referencia de Colombia. Captura el sentimiento general del mercado doméstico, correlacionado naturalmente con PFAVAL.

- **VIX**: Índice global del miedo. Episodios de alta volatilidad (COVID, guerra Ucrania, crisis bancarias) generan fly-to-quality y salida de emergentes.

---

## 🏗 Estructura del Proyecto

```
pfaval-regresion/
│
├── app.py                  ← Dashboard interactivo Streamlit (main)
├── data_generator.py       ← Generación de datos sintéticos + stub datos reales
├── requirements.txt        ← Dependencias Python
├── README.md               ← Este archivo
│
└── notebooks/
    └── analisis_exploratorio.ipynb   ← (opcional) EDA en Jupyter
```

---

## 🚀 Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/pfaval-regresion.git
cd pfaval-regresion
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 3. Lanzar la aplicación
```bash
streamlit run app.py
```
La app abrirá en `http://localhost:8501`

---

## ☁️ Despliegue en Streamlit Community Cloud

1. Haz **fork** de este repositorio o súbelo a tu GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Conecta tu GitHub, selecciona el repo y el archivo `app.py`
4. Haz clic en **Deploy** — lista en ~2 minutos

> El `requirements.txt` en la raíz del repo es suficiente; Streamlit Cloud lo instala automáticamente.

---

## 📊 Contenido del Dashboard

| Pestaña | Contenido |
|---------|-----------|
| **Serie Temporal** | Precio PFAVAL con eventos clave, media móvil configurable, estadísticas descriptivas |
| **Correlaciones** | Heatmap de correlación, barras de Pearson r, significancia estadística |
| **Variables Regresoras** | Análisis individual por variable: serie dual, scatterplot, banda de confianza |
| **Modelo de Regresión** | R², R² ajustado, RMSE, MAPE, ecuación OLS, coeficientes β, residuos, Q-Q plot |
| **Datos** | Tabla filtrable por año y fecha, fuentes de datos |

---

## 🧮 Modelo Estadístico

```
PFAVAL = β₀ + β₁·TRM + β₂·WTI + β₃·TPM + β₄·COLCAP + β₅·VIX + ε
```

Estimado por **Mínimos Cuadrados Ordinarios (OLS)**.

### Métricas reportadas
- **R²** y **R² ajustado** — proporción de varianza explicada
- **RMSE** — error cuadrático medio en COP
- **MAPE** — error porcentual absoluto medio
- **Test de Shapiro-Wilk** — normalidad de residuos
- **p-valores** de correlaciones de Pearson

---

## 📅 Período de Análisis

**2 enero 2020 — 16 mayo 2025** · ~1,400 días hábiles

Incluye eventos macro clave:
- COVID-19 crash (marzo 2020)
- Recuperación económica (2021)
- Ciclo alcista de tasas BanRep (2022–2023)
- Elecciones presidenciales Colombia (mayo 2022)
- Pico inflación / TPM 13.25% (julio 2023)
- Ciclo bajista BanRep — recortes de tasas (2024)

---

## 🔄 Datos Reales vs Simulados

El código soporta ambos modos:

**Simulado** (por defecto, sin acceso a internet):
```python
from data_generator import generate_data
df = generate_data(seed=7)  # Datos calibrados con parámetros reales
```

**Real** (requiere internet + yfinance):
```python
from data_generator import fetch_real_data
df = fetch_real_data()  # Descarga de Yahoo Finance + BanRep
```

Para activar datos reales, cambia `_REAL = False` a `_REAL = True` en `app.py`.

---

## 🛠 Stack Tecnológico

| Librería | Uso |
|---------|-----|
| `streamlit` | Dashboard web interactivo |
| `plotly` | Visualizaciones interactivas |
| `pandas` / `numpy` | Manipulación de datos |
| `scipy` | Correlaciones, tests estadísticos, regresión |
| `scikit-learn` | LinearRegression, métricas |
| `yfinance` | Descarga de datos financieros |

---

## 👤 Autor

Desarrollado como proyecto académico para el análisis de regresión lineal con datos de la Bolsa de Valores de Colombia.

- Materia: Análisis Cuantitativo / Econometría Financiera
- Herramientas: Python · Streamlit · yfinance

---

## 📄 Licencia

MIT License — libre para uso académico y personal.