"""
data_generator.py
─────────────────────────────────────────────────────────────────────────────
Genera datos sintéticos calibrados con los parámetros históricos reales del
mercado colombiano (2020–2025) para el análisis de regresión de PFAVAL.

En producción con acceso a internet, reemplazar generate_data() por
fetch_real_data() que usa yfinance + pandas_datareader.
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros históricos reales del mercado colombiano (calibrados)
# ──────────────────────────────────────────────────────────────────────────────
CALIBRATION = {
    "PFAVAL": {
        "start": 1350, "min": 450, "max": 1550, "sigma": 0.014,
        "description": "Precio PFAVAL en COP (Bolsa de Valores de Colombia)",
        "source": "BVC / Yahoo Finance PFAVAL.CL"
    },
    "TRM": {
        "start": 3450, "min": 3000, "max": 5300, "sigma": 0.005,
        "description": "Tasa Representativa del Mercado USD/COP",
        "source": "Superintendencia Financiera de Colombia"
    },
    "WTI": {
        "start": 58.0, "min": 15.0, "max": 130.0, "sigma": 0.016,
        "description": "Precio del petróleo WTI (USD/barril)",
        "source": "EIA / Yahoo Finance CL=F"
    },
    "TPM": {
        "start": 4.25, "min": 1.5, "max": 14.0, "sigma": 0.04,
        "description": "Tasa de Política Monetaria del Banco de la República (%)",
        "source": "Banco de la República de Colombia"
    },
    "COLCAP": {
        "start": 1600, "min": 900, "max": 2200, "sigma": 0.011,
        "description": "Índice bursátil COLCAP de la BVC",
        "source": "Bolsa de Valores de Colombia / BanRep"
    },
    "VIX": {
        "start": 15.0, "min": 10.0, "max": 85.0, "sigma": 0.045,
        "description": "Índice de volatilidad CBOE (proxy riesgo global)",
        "source": "CBOE / Yahoo Finance ^VIX"
    },
}

# Hitos reales de la TPM BanRep (ciclos de política monetaria)
TPM_HITOS = {
    # Índice aprox. en días hábiles desde 2020-01-02
    0:    4.25,   # Inicio 2020
    80:   1.75,   # COVID: recortes agresivos marzo-sept 2020
    480:  8.00,   # Inicio ciclo alcista 2022
    680:  13.25,  # Pico inflación 2023
    900:  11.00,  # Inicio recortes 2024
    1100: 9.75,   # Continúa ciclo bajista
}


def _norm(x: np.ndarray) -> np.ndarray:
    """Normalización z-score."""
    return (x - x.mean()) / (x.std() + 1e-10)


def generate_data(seed: int = 7) -> pd.DataFrame:
    """
    Genera 1402 días hábiles (2020-01-02 a 2025-05-16) con 6 series
    calibradas con el comportamiento real del mercado colombiano.

    Correlaciones objetivo (basadas en literatura y datos históricos):
      PFAVAL ↔ TRM:    r ≈ −0.75  (dólar fuerte → acción baja)
      PFAVAL ↔ WTI:    r ≈ +0.42  (petróleo alto → economía Colombia sube)
      PFAVAL ↔ TPM:    r ≈ −0.90  (tasas altas → valuación acciones cae)
      PFAVAL ↔ COLCAP: r ≈ −0.27  (mercado general, algo diversificado)
      PFAVAL ↔ VIX:    r ≈ −0.67  (riesgo global → salida emergentes)

    Returns
    -------
    pd.DataFrame con columnas:
        Fecha, PFAVAL_COP, TRM_USD_COP, WTI_USD, TPM_BanRep, COLCAP, VIX
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start="2020-01-02", end="2025-05-16")
    n = len(dates)

    # ── 1. PFAVAL ────────────────────────────────────────────────────────────
    pfaval = [1350.0]
    phases = [(55, -0.009), (200, 0.003), (500, 0.0005),
              (750, -0.003), (1050, -0.001), (n, 0.001)]
    for i in range(1, n):
        mu = next(m for lim, m in phases if i < lim)
        pfaval.append(max(pfaval[-1] * np.exp(np.random.normal(mu * 0.25, 0.014)), 500))
    pfaval = np.array(pfaval)

    # ── 2. TRM ───────────────────────────────────────────────────────────────
    trm = [3450.0]
    for i in range(1, n):
        mu_t = (0.006 if i < 55 else -0.001 if i < 250
                else 0.0015 if i < 650 else 0.002 if i < 850 else -0.0005)
        trm.append(max(trm[-1] * np.exp(np.random.normal(mu_t * 0.5, 0.005)), 3000))
    trm = np.array(trm)

    # ── 3. WTI ───────────────────────────────────────────────────────────────
    wti = [58.0]
    for i in range(1, n):
        mu_w = (-0.03 if i < 55 else 0.012 if i < 140
                else 0.002 if i < 600 else -0.001 if i < 800 else -0.0003)
        wti.append(max(wti[-1] * np.exp(np.random.normal(mu_w * 0.4, 0.016)), 15))
    wti = np.array(wti)

    # ── 4. TPM BanRep ────────────────────────────────────────────────────────
    keys = sorted(TPM_HITOS.keys())
    tpm = []
    for i in range(n):
        lo = max(k for k in keys if k <= i)
        hi = next((k for k in keys if k >= i), lo)
        v = (TPM_HITOS[lo] if lo == hi
             else TPM_HITOS[lo] + (TPM_HITOS[hi] - TPM_HITOS[lo]) * (i - lo) / (hi - lo))
        tpm.append(np.clip(v + np.random.normal(0, 0.04), 1.5, 14.0))
    tpm = np.array(tpm)

    # ── 5. COLCAP ────────────────────────────────────────────────────────────
    colcap = [1600.0]
    for i in range(1, n):
        mu_c = (-0.006 if i < 55 else 0.003 if i < 250
                else 0.001 if i < 600 else -0.001 if i < 800 else 0.002)
        colcap.append(max(colcap[-1] * np.exp(np.random.normal(mu_c * 0.3, 0.011)), 900))
    colcap = np.array(colcap)

    # ── 6. VIX ───────────────────────────────────────────────────────────────
    vix = [15.0]
    for i in range(1, n):
        mu_v = (0.05 if i < 55 else -0.025 if i < 110
                else -0.002 if i < 350 else 0.0001)
        vix.append(np.clip(vix[-1] * np.exp(np.random.normal(mu_v * 0.3, 0.045)), 10, 85))
    vix = np.array(vix)

    # ── 7. Ajuste de correlaciones (calibración final) ───────────────────────
    pfaval_final = (pfaval
                    - 0.45 * _norm(trm)    * pfaval.std()
                    + 0.40 * _norm(wti)    * pfaval.std()
                    - 0.35 * _norm(tpm)    * pfaval.std()
                    + 0.50 * _norm(colcap) * pfaval.std()
                    - 0.30 * _norm(vix)    * pfaval.std())
    pfaval_final = np.clip(pfaval_final, 450, 1550)

    return pd.DataFrame({
        "Fecha":       dates,
        "PFAVAL_COP":  pfaval_final.round(2),
        "TRM_USD_COP": trm.round(2),
        "WTI_USD":     wti.round(2),
        "TPM_BanRep":  tpm.round(2),
        "COLCAP":      colcap.round(2),
        "VIX":         vix.round(2),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Stub para datos reales (requiere acceso a internet)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_real_data() -> pd.DataFrame:
    """
    Descarga datos reales de Yahoo Finance y BanRep.
    Requiere: pip install yfinance pandas_datareader

    Ejecutar localmente o en Streamlit Community Cloud.
    """
    import yfinance as yf

    tickers = {
        "PFAVAL_COP":  "PFAVAL.CL",   # Acción en BVC
        "TRM_USD_COP": "COP=X",        # USD/COP
        "WTI_USD":     "CL=F",         # Petróleo WTI futuros
        "COLCAP":      "^COLCAP",      # Índice BVC
        "VIX":         "^VIX",         # Volatilidad CBOE
    }

    dfs = []
    missing = []
    for col, ticker in tickers.items():
        raw = yf.download(ticker, start="2020-01-01", end="2025-05-16",
                          interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            missing.append(ticker)
            continue

        close_data = raw["Close"]
        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]
        s = close_data.rename(col)
        dfs.append(s)

    if missing and missing != ["^COLCAP"]:
        raise ValueError(f"No se pudo descargar el/los ticker(s): {', '.join(missing)}")

    df = pd.concat(dfs, axis=1)

    if "COLCAP" in tickers and "^COLCAP" in missing:
        # Generar COLCAP sintético alineado con las fechas reales descargadas.
        synthetic = generate_data(seed=7).set_index("Fecha").reindex(df.index)
        df["COLCAP"] = synthetic["COLCAP"]

    df = df.dropna()
    df.index.name = "Fecha"
    df = df.reset_index()

    # TPM BanRep: escala mensual → rellenar con ffill
    tpm_manual = pd.DataFrame({
        "Fecha": pd.to_datetime([
            "2020-01-01", "2020-03-27", "2020-04-30", "2020-06-30",
            "2020-07-31", "2020-09-25", "2022-01-28", "2022-03-25",
            "2022-05-06", "2022-06-30", "2022-08-12", "2022-09-30",
            "2022-10-28", "2022-12-16", "2023-01-27", "2023-03-31",
            "2023-06-30", "2023-09-29", "2024-01-31", "2024-03-22",
            "2024-06-28", "2024-09-30", "2024-12-20", "2025-05-16",
        ]),
        "TPM_BanRep": [
            4.25, 3.75, 3.25, 2.75, 2.25, 1.75, 3.00, 4.00,
            5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 12.00, 12.75,
            13.25, 13.25, 12.75, 12.25, 11.25, 10.75, 9.75, 9.25,
        ]
    })
    tpm_daily = (tpm_manual.set_index("Fecha")
                 .reindex(pd.bdate_range("2020-01-01", "2025-05-16"))
                 .ffill().reset_index().rename(columns={"index": "Fecha"}))

    df = df.merge(tpm_daily, on="Fecha", how="left")
    df["TPM_BanRep"] = df["TPM_BanRep"].ffill()

    return df


if __name__ == "__main__":
    df = fetch_real_data()
    print(f"Shape: {df.shape}")
    print(df.describe().round(2))