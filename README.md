# Yfinance
# 📈 Dashboard PFAVAL — Análisis de Regresión Lineal

## Acción analizada
**Preferencial Grupo Aval (PFAVAL.CL)** — Bolsa de Valores de Colombia

## Variables regresoras
| Variable | Descripción | Fuente |
|----------|------------|--------|
| **TRM (USD/COP)** | Tasa de cambio representativa del mercado | Banco de la República / BVC |
| **TPM BanRep (%)** | Tasa de política monetaria del Banco de la República | Banco de la República |

## Estructura del proyecto
```
pfaval_dashboard/
├── app.py              ← Dashboard principal de Streamlit
├── requirements.txt    ← Dependencias del proyecto
└── README.md           ← Este archivo
```

## Instalación local

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

La app abrirá automáticamente en `http://localhost:8501`

## Despliegue en Streamlit Community Cloud

1. Sube la carpeta a un repositorio de GitHub (público)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Haz clic en **"New app"**
4. Selecciona tu repositorio, rama `main` y el archivo `app.py`
5. Clic en **"Deploy!"**

> ⚠️ El archivo `requirements.txt` debe estar en la raíz del repositorio.

## Nota sobre los datos

Actualmente el dashboard usa **datos simulados** con parámetros calibrados
al comportamiento real de PFAVAL (precio ~1000–1200 COP, volatilidad ~14% anual).

Para usar datos reales, en `app.py` reemplaza la función `load_data()` con:

```python
import yfinance as yf

@st.cache_data
def load_data():
    pfaval = yf.download("PFAVAL.CL", period="1y", interval="1d")["Close"]
    # También descargar TRM y TPM de fuentes oficiales
    ...
```

## Contenido del dashboard

- **KPIs** — Precio actual, máximo, mínimo, volatilidad anual, TRM, TPM
- **Serie de tiempo** — Precio PFAVAL con media móvil configurable
- **Gráfico TRM vs PFAVAL** — Doble eje + dispersión con línea de regresión
- **Gráfico TPM vs PFAVAL** — Doble eje + dispersión con línea de regresión
- **Modelo de regresión múltiple** — R², coeficientes β, predicho vs real, residuos
- **Ecuación del modelo** — Expresada explícitamente
- **Tabla de datos** — Expandible con todos los registros

## Correlaciones típicas observadas

| Par | r de Pearson | Interpretación |
|-----|-------------|----------------|
| PFAVAL ↔ TRM | ~−0.55 | Negativa moderada: dólar fuerte presiona la acción |
| PFAVAL ↔ TPM | ~+0.55 | Positiva moderada: tasas altas benefician al sector financiero |

---
*Trabajo académico — Análisis de Regresión Lineal — Bolsa de Valores de Colombia*
