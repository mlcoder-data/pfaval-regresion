"""
app.py — Dashboard PFAVAL | Análisis de Regresión Lineal Múltiple
Bolsa de Valores de Colombia | 2020–2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ── Intentar datos reales, fallback a sintéticos ──────────────────────────────
try:
    from data_generator import fetch_real_data, generate_data
    _REAL = False
except ImportError:
    from data_generator import generate_data
    _REAL = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PFAVAL Analytics | BVC Colombia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg":       "#080d1a",
    "surface":  "#0d1b2e",
    "border":   "#1e3a5f",
    "accent1":  "#00c8ff",
    "accent2":  "#f97316",
    "accent3":  "#a78bfa",
    "accent4":  "#22c55e",
    "accent5":  "#f43f5e",
    "text":     "#e2f0ff",
    "muted":    "#6a8faa",
}

VAR_META = {
    "TRM_USD_COP": {
        "label": "TRM (USD/COP)",
        "color": "#f97316",
        "icon":  "💵",
        "unit":  "COP",
        "desc":  "Tasa Representativa del Mercado. Precio del dólar en pesos colombianos.",
        "teoria": (
            "Cuando el dólar se encarece, el peso colombiano se devalúa. "
            "Esto genera incertidumbre macroeconómica, aumenta el costo de financiamiento "
            "externo de los bancos del Grupo Aval y reduce el apetito de los inversionistas "
            "por activos en COP. **Correlación esperada: negativa.**"
        ),
    },
    "WTI_USD": {
        "label": "WTI (USD/barril)",
        "color": "#f43f5e",
        "icon":  "🛢️",
        "unit":  "USD",
        "desc":  "Precio del petróleo crudo West Texas Intermediate.",
        "teoria": (
            "Colombia es un país exportador de petróleo. Un WTI alto mejora la "
            "balanza comercial, fortalece el COP, reduce la TRM y aumenta la actividad "
            "crediticia —el negocio principal de Grupo Aval. "
            "**Correlación esperada: positiva.**"
        ),
    },
    "TPM_BanRep": {
        "label": "TPM BanRep (%)",
        "color": "#a78bfa",
        "icon":  "🏦",
        "unit":  "%",
        "desc":  "Tasa de Política Monetaria del Banco de la República de Colombia.",
        "teoria": (
            "El ciclo de tasas afecta a las acciones financieras de dos formas: "
            "(1) tasas muy altas comprimen la valuación por descuento de flujos, "
            "(2) tasas en descenso señalan relajamiento monetario y revalorización. "
            "En el período 2022-2023, el alza brutal (1.75% → 13.25%) coincidió con "
            "caídas pronunciadas del precio. **Correlación esperada: negativa.**"
        ),
    },
    "COLCAP": {
        "label": "COLCAP",
        "color": "#22c55e",
        "icon":  "📈",
        "unit":  "pts",
        "desc":  "Índice bursátil principal de la Bolsa de Valores de Colombia.",
        "teoria": (
            "COLCAP refleja el sentimiento general del mercado colombiano. "
            "PFAVAL forma parte del índice, por lo que existe correlación por "
            "construcción, aunque puede divergir por factores idiosincráticos de "
            "Grupo Aval. **Correlación esperada: positiva.**"
        ),
    },
    "VIX": {
        "label": "VIX (CBOE)",
        "color": "#00c8ff",
        "icon":  "⚡",
        "unit":  "pts",
        "desc":  "Índice de volatilidad implícita del S&P 500 ('índice del miedo').",
        "teoria": (
            "El VIX mide la volatilidad implícita global. Cuando sube (eventos "
            "de riesgo como COVID, guerra Ucrania), los capitales huyen de mercados "
            "emergentes como Colombia. PFAVAL cae por salida de flujos. "
            "**Correlación esperada: negativa.**"
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
* {{ font-family: 'DM Sans', sans-serif; box-sizing: border-box; }}
.stApp {{ background-color: {PALETTE['bg']}; }}
div[data-testid="stSidebarContent"] {{
    background: linear-gradient(180deg, #04091a 0%, #080d1a 100%);
    border-right: 1px solid {PALETTE['border']};
}}

/* Títulos */
.hero {{ margin-bottom: 0.5rem; }}
.hero-badge {{
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.9rem;
    color: {PALETTE['accent1']};
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 700;
    color: {PALETTE['text']};
    line-height: 1.15;
    margin: 0;
}}
.hero-title span {{ color: {PALETTE['accent1']}; }}
.hero-sub {{ color: {PALETTE['muted']}; font-size: 0.95rem; margin-top: 0.3rem; }}

/* KPI cards */
.kpi {{
    background: linear-gradient(135deg, {PALETTE['surface']}, #112240);
    border: 1px solid {PALETTE['border']};
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}}
.kpi-val {{
    font-family: 'DM Mono', monospace;
    font-size: 1.55rem;
    font-weight: 500;
    color: {PALETTE['accent1']};
    line-height: 1.1;
}}
.kpi-lbl {{ font-size: 0.72rem; color: {PALETTE['muted']}; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.2rem; }}
.kpi-delta-p {{ color: {PALETTE['accent4']}; font-size: 0.82rem; font-weight: 600; }}
.kpi-delta-n {{ color: {PALETTE['accent5']}; font-size: 0.82rem; font-weight: 600; }}

/* Secciones */
.sec-hdr {{
    font-size: 1.25rem;
    font-weight: 600;
    color: {PALETTE['text']};
    border-left: 3px solid {PALETTE['accent1']};
    padding-left: 0.7rem;
    margin: 1.6rem 0 0.7rem 0;
}}
.sec-sub {{ color: {PALETTE['muted']}; font-size: 0.88rem; margin: -0.4rem 0 1rem 1rem; }}

/* Insight box */
.insight {{
    background: #0a1628;
    border: 1px solid {PALETTE['border']};
    border-top: 3px solid {PALETTE['accent1']};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #c5ddf0;
    font-size: 0.88rem;
    line-height: 1.65;
}}
.insight b {{ color: {PALETTE['text']}; }}

/* Correlation badge */
.cb-pos {{ display:inline-block; background:rgba(34,197,94,0.12); border:1px solid #22c55e;
           border-radius:20px; padding:0.2rem 0.7rem; color:#22c55e; font-weight:600; font-size:0.95rem; }}
.cb-neg {{ display:inline-block; background:rgba(244,63,94,0.12); border:1px solid #f43f5e;
           border-radius:20px; padding:0.2rem 0.7rem; color:#f43f5e; font-weight:600; font-size:0.95rem; }}
.cb-mid {{ display:inline-block; background:rgba(249,115,22,0.12); border:1px solid #f97316;
           border-radius:20px; padding:0.2rem 0.7rem; color:#f97316; font-weight:600; font-size:0.95rem; }}

/* Ecuación */
.eq-box {{
    background: #050c1a;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.3rem 2rem;
    text-align: center;
    font-family: 'DM Mono', monospace;
    color: {PALETTE['accent1']};
    font-size: 1.05rem;
    line-height: 2;
    margin: 0.5rem 0;
}}
.eq-box small {{ color: {PALETTE['muted']}; font-family: 'DM Sans', sans-serif; font-size: 0.8rem; }}

/* Data note */
.data-note {{
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    color: #b8a8e0;
    font-size: 0.82rem;
    margin-bottom: 0.5rem;
}}
hr {{ border-color: {PALETTE['border']}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{PALETTE['accent1']};font-weight:700;font-size:1.1rem;'>⚙️ Panel de Control</div>", unsafe_allow_html=True)
    st.markdown("---")

    seed = st.number_input("Semilla aleatoria", 1, 999, 7, 1,
                           help="Reproducibilidad de la simulación")

    use_real = st.checkbox(
        "Usar datos reales (requiere internet)", value=False,
        help="Descargar datos reales de Yahoo Finance y BanRep. Si no hay conexión, se usará la simulación."
    )

    st.markdown("**Período de análisis**")
    periodo = st.selectbox("Rango", ["2020–2025 (completo)", "2022–2025", "2023–2025"], index=0)

    st.markdown("**Variables regresoras activas**")
    vars_activas = {}
    for var, meta in VAR_META.items():
        vars_activas[var] = st.checkbox(
            f"{meta['icon']} {meta['label']}", value=True
        )

    st.markdown("---")
    show_ma   = st.checkbox("Media Móvil en serie temporal", value=True)
    ma_win    = st.slider("Ventana MA (días)", 10, 60, 20, 5, disabled=not show_ma)
    show_reg  = st.checkbox("Línea de regresión en scatterplots", value=True)
    show_conf = st.checkbox("Banda de confianza 95%", value=True)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.78rem; color:{PALETTE['muted']}; line-height:1.8;'>
    📌 <b style='color:{PALETTE['text']};'>Acción:</b> PFAVAL.CL<br>
    🏦 <b style='color:{PALETTE['text']};'>Emisor:</b> Grupo Aval Acciones y Valores<br>
    📅 <b style='color:{PALETTE['text']};'>Período:</b> Ene 2020 – May 2025<br>
    💰 <b style='color:{PALETTE['text']};'>Moneda:</b> COP (Pesos Colombianos)<br>
    🔢 <b style='color:{PALETTE['text']};'>Frecuencia:</b> Diaria (días hábiles)<br>
    📊 <b style='color:{PALETTE['text']};'>Bolsa:</b> BVC — Bolsa de Valores de Colombia
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(seed_val, real_data):
    if real_data:
        try:
            return fetch_real_data()
        except Exception as exc:
            st.warning(
                "No fue posible descargar datos reales; se cargará la simulación en su lugar. "
                f"Error: {exc}"
            )
    return generate_data(seed=seed_val)

with st.spinner("Cargando datos..."):
    df_full = get_data(seed, use_real)

# Filtro de período
if periodo == "2022–2025":
    df = df_full[df_full["Fecha"] >= "2022-01-01"].copy().reset_index(drop=True)
elif periodo == "2023–2025":
    df = df_full[df_full["Fecha"] >= "2023-01-01"].copy().reset_index(drop=True)
else:
    df = df_full.copy()

n = len(df)
vars_sel = [v for v, active in vars_activas.items() if active]

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
precio_ini  = df["PFAVAL_COP"].iloc[0]
precio_fin  = df["PFAVAL_COP"].iloc[-1]
ret_total   = (precio_fin / precio_ini - 1) * 100
ret_diario  = df["PFAVAL_COP"].pct_change().mean() * 100
vol_anual   = df["PFAVAL_COP"].pct_change().std() * np.sqrt(252) * 100
precio_max  = df["PFAVAL_COP"].max()
precio_min  = df["PFAVAL_COP"].min()
trm_actual  = df["TRM_USD_COP"].iloc[-1]
tpm_actual  = df["TPM_BanRep"].iloc[-1]
wti_actual  = df["WTI_USD"].iloc[-1]

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
col_h, col_p = st.columns([3, 1])
with col_h:
    st.markdown(f"""
    <div class="hero">
        <div class="hero-badge">📊 Análisis de Regresión Lineal — BVC Colombia</div>
        <div class="hero-title">Preferencial <span>Grupo Aval</span></div>
        <div class="hero-sub">
            PFAVAL · {df["Fecha"].iloc[0].strftime("%d %b %Y")} – {df["Fecha"].iloc[-1].strftime("%d %b %Y")} · 
            {n:,} observaciones diarias · 5 variables regresoras
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_p:
    delta_cls = "kpi-delta-p" if ret_total >= 0 else "kpi-delta-n"
    arrow = "▲" if ret_total >= 0 else "▼"
    st.markdown(f"""
    <div class="kpi" style="margin-top:0.3rem;">
        <div class="kpi-val">$ {precio_fin:,.1f}</div>
        <div class="kpi-lbl">Precio PFAVAL (COP)</div>
        <div class="{delta_cls}">{arrow} {ret_total:+.2f}% período</div>
    </div>
    """, unsafe_allow_html=True)

if not _REAL:
    st.markdown("""
    <div class="data-note">
    ⚠️ <b>Modo simulación:</b> Los datos son sintéticos, calibrados con parámetros históricos reales del mercado colombiano.
    En Streamlit Community Cloud con acceso a internet, la función <code>fetch_real_data()</code> descarga los datos reales de Yahoo Finance y BanRep.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (f"$ {precio_max:,.0f}", "Máx. período"),
    (f"$ {precio_min:,.0f}", "Mín. período"),
    (f"{vol_anual:.1f}%",    "Volatilidad anual"),
    (f"$ {trm_actual:,.0f}", "TRM (COP/USD)"),
    (f"{tpm_actual:.2f}%",   "TPM BanRep"),
    (f"$ {wti_actual:.1f}",  "WTI (USD/bbl)"),
]
for col, (val, lbl) in zip([k1, k2, k3, k4, k5, k6], kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Serie Temporal",
    "🔗  Correlaciones",
    "📉  Variables Regresoras",
    "🧮  Modelo de Regresión",
    "📋  Datos",
])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — SERIE TEMPORAL                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="sec-hdr">Evolución del Precio PFAVAL (COP)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Precio de cierre diario · Grupos de eventos clave marcados</div>', unsafe_allow_html=True)

    fig_ts = go.Figure()

    # Área rellena
    fig_ts.add_trace(go.Scatter(
        x=df["Fecha"], y=df["PFAVAL_COP"], name="PFAVAL",
        line=dict(color=PALETTE["accent1"], width=1.8),
        fill="tozeroy", fillcolor="rgba(0,200,255,0.05)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>$ %{y:,.2f} COP<extra></extra>"
    ))

    # Media móvil
    if show_ma:
        ma = df["PFAVAL_COP"].rolling(ma_win, min_periods=1).mean()
        fig_ts.add_trace(go.Scatter(
            x=df["Fecha"], y=ma, name=f"MA {ma_win}d",
            line=dict(color=PALETTE["accent2"], width=1.5, dash="dot"), opacity=0.85
        ))

    # Eventos clave
    events = [
        ("2020-03-06", "COVID-19\nCrash", PALETTE["accent5"]),
        ("2022-01-28", "Inicio\nalza TPM", PALETTE["accent3"]),
        ("2022-05-29", "Elecciones\npresidenciales", PALETTE["accent2"]),
        ("2023-07-28", "Pico\nTPM 13.25%", PALETTE["accent5"]),
        ("2024-01-31", "Inicio\nrecortes BanRep", PALETTE["accent4"]),
    ]
    for ev_date, ev_text, ev_color in events:
        ev_dt = pd.Timestamp(ev_date)
        if df["Fecha"].min() <= ev_dt <= df["Fecha"].max():
            closest_idx = (df["Fecha"] - ev_dt).abs().idxmin()
            ev_price = df.loc[closest_idx, "PFAVAL_COP"]
            fig_ts.add_vline(x=ev_dt, line_dash="dot", line_color=ev_color, line_width=1, opacity=0.6)
            fig_ts.add_annotation(
                x=ev_dt, y=ev_price * 1.05,
                text=ev_text.replace("\n", "<br>"),
                showarrow=True, arrowhead=2, arrowcolor=ev_color,
                font=dict(size=9, color=ev_color), bgcolor="rgba(8,13,26,0.8)",
                bordercolor=ev_color, borderwidth=1, arrowwidth=1
            )

    fig_ts.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.5)", height=420,
        margin=dict(l=5, r=5, t=15, b=10),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
        yaxis=dict(title="COP", gridcolor="#192d4d", tickformat=",.0f", zeroline=False),
        xaxis=dict(gridcolor="#192d4d", zeroline=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Estadísticas descriptivas
    st.markdown('<div class="sec-hdr" style="font-size:1rem;">Estadísticas Descriptivas</div>', unsafe_allow_html=True)
    cols_stat = ["PFAVAL_COP", "TRM_USD_COP", "WTI_USD", "TPM_BanRep", "COLCAP", "VIX"]
    desc = df[cols_stat].describe().round(2)
    desc.index = ["Count", "Media", "Desv. Estándar", "Mínimo", "Q1 (25%)", "Mediana", "Q3 (75%)", "Máximo"]
    st.dataframe(desc.style.format("{:,.2f}").background_gradient(cmap="Blues", axis=1),
                 use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — CORRELACIONES                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="sec-hdr">Análisis de Correlación</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Coeficiente de Pearson entre PFAVAL y cada variable regresora</div>', unsafe_allow_html=True)

    all_vars = list(VAR_META.keys())
    corr_results = {}
    for var in all_vars:
        r, p = stats.pearsonr(df["PFAVAL_COP"], df[var])
        corr_results[var] = {"r": r, "p": p, "r2": r**2}

    # Heatmap de correlación
    cols_corr = ["PFAVAL_COP"] + all_vars
    corr_matrix = df[cols_corr].corr()
    labels = ["PFAVAL", "TRM", "WTI", "TPM", "COLCAP", "VIX"]

    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr_matrix.values.round(3),
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.4f}<extra></extra>",
        colorbar=dict(title="r", thickness=14),
    ))
    fig_heat.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=400,
        margin=dict(l=5, r=5, t=20, b=5),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )

    col_h1, col_h2 = st.columns([1.3, 1])
    with col_h1:
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_h2:
        st.markdown("**Coeficientes de correlación con PFAVAL**")
        for var, res in corr_results.items():
            r, p, r2 = res["r"], res["p"], res["r2"]
            meta = VAR_META[var]
            badge = ("cb-pos" if r > 0.3 else "cb-neg" if r < -0.3 else "cb-mid")
            sig = "✅ significativa" if p < 0.05 else "⚠️ no significativa"
            st.markdown(f"""
            <div style="margin: 0.4rem 0; padding: 0.6rem 0.8rem;
                        background:#0d1b2e; border-radius:8px; border:1px solid #1e3a5f;">
                <b style="color:{meta['color']};">{meta['icon']} {meta['label']}</b><br>
                <span class="{badge}">r = {r:+.4f}</span>
                &nbsp; R² = {r2:.4f} &nbsp; {sig}<br>
                <small style="color:#6a8faa;">p-valor: {p:.2e}</small>
            </div>
            """, unsafe_allow_html=True)

    # Gráfico de barras de correlaciones
    st.markdown('<div class="sec-hdr" style="font-size:1rem; margin-top:1rem;">Visualización de Correlaciones</div>', unsafe_allow_html=True)

    fig_bar = go.Figure(go.Bar(
        x=[VAR_META[v]["label"] for v in all_vars],
        y=[corr_results[v]["r"] for v in all_vars],
        marker_color=[PALETTE["accent4"] if corr_results[v]["r"] > 0 else PALETTE["accent5"]
                      for v in all_vars],
        text=[f"r = {corr_results[v]['r']:+.4f}" for v in all_vars],
        textposition="outside",
        hovertemplate="%{x}<br>r = %{y:.4f}<extra></extra>",
    ))
    fig_bar.add_hline(y=0, line_color="#4a7a99", line_width=1)
    fig_bar.add_hline(y=0.5, line_dash="dot", line_color=PALETTE["accent4"], opacity=0.4, annotation_text="corr. fuerte", annotation_font_size=10)
    fig_bar.add_hline(y=-0.5, line_dash="dot", line_color=PALETTE["accent5"], opacity=0.4)
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,46,0.5)", height=320,
        margin=dict(l=5, r=5, t=20, b=5),
        yaxis=dict(title="Coef. de Pearson (r)", range=[-1.1, 1.1], gridcolor="#192d4d", zeroline=False),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — VARIABLES REGRESORAS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="sec-hdr">Análisis Detallado por Variable</div>', unsafe_allow_html=True)

    var_selected = st.selectbox(
        "Seleccionar variable regresora",
        options=all_vars,
        format_func=lambda v: f"{VAR_META[v]['icon']}  {VAR_META[v]['label']}"
    )
    meta = VAR_META[var_selected]
    r, p = stats.pearsonr(df["PFAVAL_COP"], df[var_selected])

    # Descripción teórica
    st.markdown(f"""
    <div class="insight">
        <b>{meta['icon']} {meta['label']}</b><br>
        <small style="color:{PALETTE['muted']}">{meta['desc']}</small><br><br>
        {meta['teoria']}<br><br>
        <b>Correlación observada:</b>
        <span class="{'cb-pos' if r > 0 else 'cb-neg'}">r = {r:+.4f}</span>
        &nbsp;&nbsp; <b>p-valor:</b> {p:.2e}
        &nbsp;&nbsp; <b>R²:</b> {r**2:.4f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col_l, col_r = st.columns(2)

    with col_l:
        # Doble eje: serie temporal PFAVAL + variable
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Scatter(
            x=df["Fecha"], y=df["PFAVAL_COP"], name="PFAVAL (COP)",
            line=dict(color=PALETTE["accent1"], width=1.5),
            hovertemplate="PFAVAL: $%{y:,.2f}<extra></extra>"
        ), secondary_y=False)
        fig_dual.add_trace(go.Scatter(
            x=df["Fecha"], y=df[var_selected], name=meta["label"],
            line=dict(color=meta["color"], width=1.5),
            hovertemplate=f"{meta['label']}: %{{y:.2f}}<extra></extra>"
        ), secondary_y=True)
        fig_dual.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,46,0.5)", height=340,
            margin=dict(l=5, r=5, t=30, b=5),
            title=dict(text="Evolución temporal comparada", font=dict(size=12, color=PALETTE["muted"])),
            legend=dict(orientation="h", x=0, y=1.12),
            hovermode="x unified",
        )
        fig_dual.update_yaxes(title_text="PFAVAL (COP)", secondary_y=False,
                              gridcolor="#192d4d", tickformat=",.0f")
        fig_dual.update_yaxes(title_text=meta["label"], secondary_y=True,
                              gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dual, use_container_width=True)

    with col_r:
        # Scatterplot con regresión
        fig_sc = go.Figure()

        # Coloreado por año
        df_sc = df.copy()
        df_sc["Año"] = df_sc["Fecha"].dt.year

        for year, grp in df_sc.groupby("Año"):
            fig_sc.add_trace(go.Scatter(
                x=grp[var_selected], y=grp["PFAVAL_COP"],
                mode="markers", name=str(year),
                marker=dict(size=4, opacity=0.6),
                hovertemplate=f"{year}<br>{meta['label']}: %{{x:.2f}}<br>PFAVAL: $%{{y:,.2f}}<extra></extra>"
            ))

        # Línea de regresión
        if show_reg:
            m_lr, b_lr, *_ = stats.linregress(df[var_selected], df["PFAVAL_COP"])
            x_line = np.linspace(df[var_selected].min(), df[var_selected].max(), 200)
            fig_sc.add_trace(go.Scatter(
                x=x_line, y=m_lr * x_line + b_lr, name="Regresión OLS",
                mode="lines", line=dict(color="white", width=1.8, dash="dash")
            ))
            if show_conf:
                # Banda de confianza 95%
                X_lr = df[var_selected].values.reshape(-1, 1)
                y_lr = df["PFAVAL_COP"].values
                n_lr = len(y_lr)
                y_hat = m_lr * X_lr.ravel() + b_lr
                sse = np.sum((y_lr - y_hat) ** 2)
                s2 = sse / (n_lr - 2)
                x_mean = X_lr.mean()
                sxx = np.sum((X_lr - x_mean) ** 2)
                x_new = x_line.reshape(-1, 1)
                se_pred = np.sqrt(s2 * (1/n_lr + (x_new.ravel()-x_mean)**2/sxx))
                t_crit = stats.t.ppf(0.975, n_lr - 2)
                y_upper = m_lr*x_line+b_lr + t_crit*se_pred
                y_lower = m_lr*x_line+b_lr - t_crit*se_pred
                fig_sc.add_trace(go.Scatter(
                    x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself", fillcolor="rgba(255,255,255,0.06)",
                    line=dict(color="rgba(0,0,0,0)"), name="IC 95%"
                ))

        fig_sc.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,46,0.5)", height=340,
            margin=dict(l=5, r=5, t=30, b=5),
            title=dict(text=f"Dispersión: {meta['label']} vs PFAVAL", font=dict(size=12, color=PALETTE["muted"])),
            legend=dict(orientation="h", x=0, y=1.12, font=dict(size=9)),
            xaxis=dict(title=meta["label"], gridcolor="#192d4d"),
            yaxis=dict(title="PFAVAL (COP)", gridcolor="#192d4d", tickformat=",.0f"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Resumen estadístico de la variable
    c1, c2, c3, c4 = st.columns(4)
    ser = df[var_selected]
    for col, (val, lbl) in zip([c1, c2, c3, c4], [
        (f"{ser.mean():,.2f}", f"Media {meta['unit']}"),
        (f"{ser.std():,.2f}",  "Desv. estándar"),
        (f"{ser.min():,.2f}",  f"Mínimo"),
        (f"{ser.max():,.2f}",  f"Máximo"),
    ]):
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-val" style="font-size:1.2rem;">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — MODELO DE REGRESIÓN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab4:
    st.markdown('<div class="sec-hdr">Modelo de Regresión Lineal Múltiple</div>', unsafe_allow_html=True)

    if len(vars_sel) == 0:
        st.warning("Selecciona al menos una variable regresora en el panel lateral.")
    else:
        X = df[vars_sel].values
        y = df["PFAVAL_COP"].values

        # Modelo completo
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        residuals = y - y_pred

        r2   = r2_score(y, y_pred)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - len(vars_sel) - 1)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae  = mean_absolute_error(y, y_pred)
        mape = np.mean(np.abs(residuals / y)) * 100

        # ── KPIs del modelo ──────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        for col, (val, lbl, note) in zip([m1, m2, m3, m4], [
            (f"{r2:.4f}",    "R²",                f"{r2*100:.1f}% varianza explicada"),
            (f"{r2_adj:.4f}","R² ajustado",       f"{len(vars_sel)} vars. | {n} obs."),
            (f"$ {rmse:.2f}","RMSE (COP)",        "Error cuadrático medio"),
            (f"{mape:.2f}%", "MAPE",              "Error porcentual absoluto medio"),
        ]):
            with col:
                st.markdown(f"""
                <div class="kpi">
                    <div class="kpi-val">{val}</div>
                    <div class="kpi-lbl">{lbl}</div>
                    <div style="color:#6a8faa;font-size:0.78rem;margin-top:0.2rem;">{note}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Ecuación del modelo ───────────────────────────────────────────────
        eq_terms = " ".join([
            f"{'+ ' if reg.coef_[i] >= 0 else '− '}{abs(reg.coef_[i]):.4f} × {VAR_META[v]['label']}"
            for i, v in enumerate(vars_sel)
        ])
        st.markdown(f"""
        <div class="eq-box">
            PFAVAL = {reg.intercept_:,.2f} {eq_terms}<br>
            <small>R² = {r2:.4f} | R² adj. = {r2_adj:.4f} | n = {n:,} observaciones</small>
        </div>
        """, unsafe_allow_html=True)

        # ── Coeficientes ─────────────────────────────────────────────────────
        st.markdown('<div class="sec-hdr" style="font-size:1rem; margin-top:1.2rem;">Coeficientes del Modelo</div>', unsafe_allow_html=True)

        coef_df = pd.DataFrame({
            "Variable":    [VAR_META[v]["label"] for v in vars_sel],
            "Coeficiente": reg.coef_,
            "Signo":       ["📈 Positivo" if c > 0 else "📉 Negativo" for c in reg.coef_],
            "Interpretación": [
                f"Por cada +1 {VAR_META[v]['unit']}, PFAVAL {'sube' if reg.coef_[i]>0 else 'baja'} $ {abs(reg.coef_[i]):.4f} COP"
                for i, v in enumerate(vars_sel)
            ]
        })

        fig_coef = go.Figure(go.Bar(
            x=[VAR_META[v]["label"] for v in vars_sel],
            y=reg.coef_,
            marker_color=[PALETTE["accent4"] if c > 0 else PALETTE["accent5"] for c in reg.coef_],
            text=[f"{c:+.4f}" for c in reg.coef_],
            textposition="outside",
            hovertemplate="%{x}<br>β = %{y:.6f}<extra></extra>",
        ))
        fig_coef.add_hline(y=0, line_color="#4a7a99", line_width=1)
        fig_coef.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,46,0.5)", height=280,
            margin=dict(l=5, r=5, t=20, b=5),
            yaxis=dict(title="Coeficiente β", gridcolor="#192d4d", zeroline=False),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.dataframe(
            coef_df.style.format({
                "Coeficiente": "{:+.4f}",
            }),
            use_container_width=True,
        )

        # ── Predicho vs Real + Residuos ────────────────────────────────────
        col_pv, col_res = st.columns(2)
        with col_pv:
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(
                x=df["Fecha"], y=y, name="Real",
                line=dict(color=PALETTE["accent1"], width=1.3)
            ))
            fig_pv.add_trace(go.Scatter(
                x=df["Fecha"], y=y_pred, name="Predicho (OLS)",
                line=dict(color=PALETTE["accent2"], width=1.3, dash="dot")
            ))
            fig_pv.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,27,46,0.5)", height=280,
                margin=dict(l=5, r=5, t=30, b=5),
                title=dict(text="Real vs Predicho", font=dict(size=12, color=PALETTE["muted"])),
                legend=dict(orientation="h", x=0, y=1.15),
                yaxis=dict(gridcolor="#192d4d", tickformat=",.0f"),
                xaxis=dict(gridcolor="#192d4d"),
            )
            st.plotly_chart(fig_pv, use_container_width=True)

        with col_res:
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(
                x=df["Fecha"], y=residuals,
                marker_color=np.where(residuals >= 0, PALETTE["accent4"], PALETTE["accent5"]),
                opacity=0.65, name="Residuo",
                hovertemplate="%{x|%d %b %Y}<br>Residuo: %{y:+,.2f} COP<extra></extra>"
            ))
            fig_res.add_hline(y=0, line_color="#4a7a99", line_dash="dot", line_width=1)
            fig_res.add_hrect(y0=-rmse, y1=rmse, fillcolor="rgba(255,255,255,0.03)",
                              line_width=0, annotation_text="±RMSE")
            fig_res.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,27,46,0.5)", height=280,
                margin=dict(l=5, r=5, t=30, b=5),
                title=dict(text="Residuos del Modelo", font=dict(size=12, color=PALETTE["muted"])),
                yaxis=dict(title="Error (COP)", gridcolor="#192d4d"),
                xaxis=dict(gridcolor="#192d4d"),
                showlegend=False,
            )
            st.plotly_chart(fig_res, use_container_width=True)

        # ── Q-Q Plot de residuos ───────────────────────────────────────────
        qq_x = stats.probplot(residuals, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=qq_x[0][0], y=qq_x[0][1], mode="markers",
            marker=dict(color=PALETTE["accent3"], size=3, opacity=0.6), name="Residuos"
        ))
        fig_qq.add_trace(go.Scatter(
            x=[qq_x[0][0][0], qq_x[0][0][-1]],
            y=[qq_x[1][1] + qq_x[1][0]*qq_x[0][0][0],
               qq_x[1][1] + qq_x[1][0]*qq_x[0][0][-1]],
            mode="lines", line=dict(color="white", dash="dash", width=1.2), name="Normal teórica"
        ))
        fig_qq.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,46,0.5)", height=280,
            margin=dict(l=5, r=5, t=30, b=5),
            title=dict(text="Q-Q Plot de Residuos (normalidad)", font=dict(size=12, color=PALETTE["muted"])),
            xaxis=dict(title="Cuantiles teóricos (Normal)", gridcolor="#192d4d"),
            yaxis=dict(title="Cuantiles de residuos", gridcolor="#192d4d"),
        )

        stat_sw, p_sw = stats.shapiro(residuals[:5000])  # Shapiro máx 5000
        col_qq, col_sw = st.columns([2, 1])
        with col_qq:
            st.plotly_chart(fig_qq, use_container_width=True)
        with col_sw:
            st.markdown(f"""
            <div class="insight" style="margin-top:1rem;">
                <b>Test de Shapiro-Wilk</b><br>
                (normalidad de residuos)<br><br>
                W = {stat_sw:.4f}<br>
                p-valor = {p_sw:.4e}<br><br>
                {"✅ Residuos aproximadamente normales (p > 0.05)" if p_sw > 0.05 else
                 "⚠️ Residuos no normales (p < 0.05). Con n={:,} obs., el teorema central del límite mitiga este problema.".format(n)}<br><br>
                <b>Interpretación del R²:</b><br>
                El modelo explica el <b>{r2*100:.1f}%</b> de la variabilidad del precio de PFAVAL con las variables seleccionadas.
            </div>
            """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 — DATOS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
with tab5:
    st.markdown('<div class="sec-hdr">Tabla de Datos Completa</div>', unsafe_allow_html=True)

    # Filtros rápidos
    colf1, colf2 = st.columns(2)
    with colf1:
        year_filter = st.multiselect("Filtrar por año", sorted(df["Fecha"].dt.year.unique()), default=[])
    with colf2:
        search_date = st.text_input("Buscar fecha (YYYY-MM)", "")

    df_show = df.copy()
    if year_filter:
        df_show = df_show[df_show["Fecha"].dt.year.isin(year_filter)]
    if search_date:
        df_show = df_show[df_show["Fecha"].astype(str).str.startswith(search_date)]

    df_display = df_show.copy()
    df_display["Fecha"] = df_display["Fecha"].dt.strftime("%Y-%m-%d")
    df_display = df_display.rename(columns={
        "PFAVAL_COP":  "PFAVAL (COP)",
        "TRM_USD_COP": "TRM (COP/USD)",
        "WTI_USD":     "WTI (USD/bbl)",
        "TPM_BanRep":  "TPM BanRep (%)",
        "COLCAP":      "COLCAP (pts)",
        "VIX":         "VIX",
    })

    st.dataframe(
        df_display.style.format({
            "PFAVAL (COP)":  "$ {:,.2f}",
            "TRM (COP/USD)": "$ {:,.2f}",
            "WTI (USD/bbl)": "$ {:,.2f}",
            "TPM BanRep (%)":"{:.2f}%",
            "COLCAP (pts)":  "{:,.2f}",
            "VIX":           "{:.2f}",
        }),
        use_container_width=True, height=400
    )

    st.markdown(f"**{len(df_show):,} registros mostrados de {n:,} totales**")

    # Fuentes
    st.markdown('<div class="sec-hdr" style="font-size:1rem; margin-top:1rem;">Fuentes de Datos</div>', unsafe_allow_html=True)
    st.markdown("""
    | Variable | Fuente | Disponibilidad |
    |----------|--------|---------------|
    | PFAVAL (COP) | Bolsa de Valores de Colombia / `yfinance PFAVAL.CL` | Diaria |
    | TRM (USD/COP) | Superintendencia Financiera Colombia / BanRep | Diaria |
    | WTI (USD/bbl) | EIA / `yfinance CL=F` | Diaria |
    | TPM BanRep | Banco de la República de Colombia | Por reunión JDBR |
    | COLCAP | BVC / BanRep / `yfinance ^COLCAP` | Diaria |
    | VIX | CBOE / `yfinance ^VIX` | Diaria |
    """)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center; color:{PALETTE['muted']}; font-size:0.78rem; padding:0.5rem; line-height:2;'>
    📊 <b style='color:{PALETTE['text']};'>Dashboard PFAVAL</b> — Análisis de Regresión Lineal Múltiple<br>
    Preferencial Grupo Aval · Bolsa de Valores de Colombia · Período 2020–2025<br>
    Variables: TRM · WTI · TPM BanRep · COLCAP · VIX &nbsp;|&nbsp;
    Modelo: OLS (Ordinary Least Squares)<br>
    <i>Datos simulados calibrados con parámetros históricos reales. 
    En producción, <code>fetch_real_data()</code> usa yfinance + BanRep.</i>
</div>
""", unsafe_allow_html=True)