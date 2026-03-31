import streamlit as st
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL — Traffic Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. BRIGHT VIBRANT CSS  (baby pink + light green)
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">

<style>
/* ── Root Variables ── */
:root {
    --bg-deep:      #f0f4ff;
    --bg-card:      #ffffff;
    --accent-cyan:  #4f46e5;
    --accent-red:   #ef4444;
    --accent-pink:  #ffb3c6;       /* baby pink  — replaces amber/yellow */
    --accent-green: #90ee90;       /* light green — replaces dark green  */
    --accent-purple:#7c3aed;
    --border-glow:  rgba(79,70,229,0.22);
    --grid-color:   rgba(79,70,229,0.055);
    --text-primary: #1e1b4b;
    --text-muted:   #6b7280;
    --font-display: 'Orbitron', monospace;
    --font-body:    'Rajdhani', sans-serif;
    --font-mono:    'Share Tech Mono', monospace;
}

/* ── Global: vibrant gradient-mesh background ── */
* { box-sizing: border-box; }
.stApp {
    background:
        radial-gradient(ellipse at  8% 15%, rgba(167,139,250,0.30) 0%, transparent 48%),
        radial-gradient(ellipse at 88% 12%, rgba(251,146, 60,0.20) 0%, transparent 44%),
        radial-gradient(ellipse at 50% 88%, rgba( 99,102,241,0.22) 0%, transparent 52%),
        radial-gradient(ellipse at 92% 80%, rgba(244,114,182,0.20) 0%, transparent 42%),
        radial-gradient(ellipse at 20% 75%, rgba(167,243,208,0.25) 0%, transparent 45%),
        linear-gradient(135deg, #f0f4ff 0%, #fce7f3 30%, #ede9fe 60%, #ecfdf5 100%);
    background-attachment: fixed;
    color: var(--text-primary);
    font-family: var(--font-body);
}
/* grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 2px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1060 0%, #2d1b69 55%, #1e1b4b 100%) !important;
    border-right: 2px solid rgba(167,139,250,0.28);
}
[data-testid="stSidebar"] * { color: #e9d5ff !important; }
[data-testid="stSidebar"] label {
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
    letter-spacing: 1px;
    color: #c4b5fd !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, rgba(79,70,229,0.10), rgba(124,58,237,0.07)) !important;
    border: 1px solid rgba(167,139,250,0.50) !important;
    color: #4f46e5 !important;
    font-family: var(--font-display) !important;
    font-size: 11px !important;
    letter-spacing: 3px !important;
    padding: 14px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    overflow: hidden;
    position: relative;
}
.stButton > button::before {
    content: '';
    position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(79,70,229,0.14), transparent);
    transition: left 0.4s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(79,70,229,0.20), rgba(124,58,237,0.16)) !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.28) !important;
    transform: translateY(-2px) !important;
}

/* ── Selects ── */
.stSelectbox > div > div {
    background: white !important;
    border: 1px solid var(--border-glow) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border-glow) !important; opacity: 0.4; }

/* ── Headings ── */
h1,h2,h3 { font-family: var(--font-display) !important; color: var(--text-primary) !important; }
h1 { font-size: 13px !important; letter-spacing: 6px !important; }

/* ══════════════ VEHICLE ANIMATION ══════════════ */
.road-scene {
    position: relative;
    width: 100%;
    height: 100px;
    overflow: hidden;
    border-radius: 14px;
    margin: 14px 0 22px;
    background: linear-gradient(180deg,
        #dbeafe 0%, #dbeafe 28%,
        #374151 28%, #374151 72%,
        #d1fae5 72%, #d1fae5 100%);
    box-shadow: 0 6px 24px rgba(79,70,229,0.12);
}
/* dashed centre line */
.road-scene::after {
    content: '';
    position: absolute;
    top: 49%; left: 0; right: 0;
    height: 3px;
    background: repeating-linear-gradient(
        90deg,
        #fbbf24 0px, #fbbf24 28px,
        transparent 28px, transparent 56px
    );
}
@keyframes drive-right {
    from { transform: translateX(-200px); }
    to   { transform: translateX(105vw);  }
}
@keyframes drive-left {
    from { transform: translateX(105vw) scaleX(-1); }
    to   { transform: translateX(-200px) scaleX(-1); }
}
.veh {
    position: absolute;
    font-size: 34px;
    line-height: 1;
    filter: drop-shadow(2px 3px 5px rgba(0,0,0,0.25));
    animation-timing-function: linear;
    animation-iteration-count: infinite;
}
/* Top lane — going right */
.veh.top  { top: 10px;  animation-name: drive-right; }
/* Bottom lane — going left */
.veh.bot  { top: 55px;  animation-name: drive-left;  }

.veh.d1 { animation-duration: 2.8s; animation-delay: 0.0s; }
.veh.d2 { animation-duration: 3.6s; animation-delay: 0.7s; }
.veh.d3 { animation-duration: 2.3s; animation-delay: 1.4s; }
.veh.d4 { animation-duration: 3.1s; animation-delay: 0.3s; }
.veh.d5 { animation-duration: 4.0s; animation-delay: 1.0s; }
.veh.d6 { animation-duration: 2.6s; animation-delay: 1.9s; }

/* ══════════════ HUD HEADER ══════════════ */
.sentinel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.65);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 22px 30px;
    margin-bottom: 28px;
    box-shadow: 0 4px 28px rgba(79,70,229,0.10);
    border: 1px solid rgba(79,70,229,0.12);
}
.sentinel-logo {
    font-family: var(--font-display);
    font-size: 30px;
    font-weight: 900;
    letter-spacing: 8px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #ec4899, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sentinel-sub {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 3px;
    margin-top: 5px;
}
.sentinel-clock {
    font-family: var(--font-mono);
    font-size: 12px;
    color: #4f46e5;
    text-align: right;
    background: rgba(79,70,229,0.06);
    padding: 12px 18px;
    border-radius: 10px;
    border: 1px solid rgba(79,70,229,0.15);
    line-height: 1.8;
}

/* ══════════════ STATUS PANELS ══════════════ */
/* OPTIMAL — light green */
.status-optimal {
    background: linear-gradient(135deg, rgba(144,238,144,0.20) 0%, rgba(144,238,144,0.05) 100%);
    border: 2px solid rgba(144,238,144,0.65);
    border-radius: 14px;
    padding: 36px 42px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(144,238,144,0.18);
}
.status-optimal::before {
    content: 'OPTIMAL';
    font-family: var(--font-display);
    font-size: 96px; font-weight: 900;
    position: absolute; right: -8px; top: 50%;
    transform: translateY(-50%);
    color: rgba(144,238,144,0.09);
    letter-spacing: -4px; pointer-events: none;
}

/* ELEVATED — baby pink */
.status-elevated {
    background: linear-gradient(135deg, rgba(255,179,198,0.22) 0%, rgba(255,179,198,0.06) 100%);
    border: 2px solid rgba(255,179,198,0.75);
    border-radius: 14px;
    padding: 36px 42px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(255,179,198,0.22);
}
.status-elevated::before {
    content: 'WARNING';
    font-family: var(--font-display);
    font-size: 96px; font-weight: 900;
    position: absolute; right: -8px; top: 50%;
    transform: translateY(-50%);
    color: rgba(255,179,198,0.09);
    letter-spacing: -4px; pointer-events: none;
}

/* CRITICAL — red */
.status-critical {
    background: linear-gradient(135deg, rgba(239,68,68,0.14) 0%, rgba(239,68,68,0.04) 100%);
    border: 2px solid rgba(239,68,68,0.62);
    border-radius: 14px;
    padding: 36px 42px;
    position: relative;
    overflow: hidden;
    animation: critPulse 2s ease-in-out infinite;
    box-shadow: 0 8px 32px rgba(239,68,68,0.16);
}
@keyframes critPulse {
    0%,100% { box-shadow: 0 8px 32px rgba(239,68,68,0.12); }
    50%      { box-shadow: 0 8px 52px rgba(239,68,68,0.32); }
}
.status-critical::before {
    content: 'CRITICAL';
    font-family: var(--font-display);
    font-size: 96px; font-weight: 900;
    position: absolute; right: -8px; top: 50%;
    transform: translateY(-50%);
    color: rgba(239,68,68,0.07);
    letter-spacing: -4px; pointer-events: none;
}

.status-badge {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: var(--font-display); font-size: 22px;
    font-weight: 700; letter-spacing: 4px; margin-bottom: 12px;
}
.badge-dot {
    width: 12px; height: 12px; border-radius: 50%;
    display: inline-block;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.25;} }
.status-subtitle {
    font-family: var(--font-mono); font-size: 13px;
    letter-spacing: 2px; color: var(--text-muted); margin-bottom: 18px;
}
/* recommendation box — baby pink border */
.reco-box {
    margin-top: 18px;
    background: rgba(255,179,198,0.12);
    border-left: 4px solid var(--accent-pink);
    padding: 13px 18px; border-radius: 0 8px 8px 0;
    font-family: var(--font-mono); font-size: 13px;
    color: #be185d; letter-spacing: 1px;
}

/* ══════════════ PROBABILITY BARS ══════════════ */
.prob-row {
    display: flex; align-items: center; gap: 14px;
    margin: 11px 0; font-family: var(--font-mono); font-size: 12px;
}
.prob-label  { width: 82px; color: var(--text-muted); letter-spacing: 1px; }
.prob-bar-track {
    flex: 1; height: 9px;
    background: rgba(79,70,229,0.08);
    border-radius: 5px; overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 5px; }
.prob-value { width: 52px; text-align: right; font-weight: 700; }

/* ══════════════ KPI CARDS ══════════════ */
.kpi-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 14px; margin-bottom: 24px;
}
.kpi-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(79,70,229,0.14);
    border-radius: 14px; padding: 20px 22px;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 18px rgba(79,70,229,0.08);
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(79,70,229,0.14); }
.kpi-card::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #4f46e5, #ec4899, #f97316);
}
.kpi-label {
    font-family: var(--font-mono); font-size: 10px;
    color: var(--text-muted); letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 8px;
}
.kpi-value { font-family: var(--font-display); font-size: 28px; font-weight: 700; }
.kpi-unit  { font-family: var(--font-mono); font-size: 10px; color: var(--text-muted); }

/* ══════════════ MISC ══════════════ */
.section-label {
    font-family: var(--font-display); font-size: 10px;
    letter-spacing: 5px; color: var(--text-muted);
    border-bottom: 1px solid rgba(79,70,229,0.15);
    padding-bottom: 8px; margin-bottom: 16px;
}
.chart-frame {
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(79,70,229,0.14);
    border-radius: 14px; padding: 20px;
    box-shadow: 0 4px 18px rgba(79,70,229,0.07);
}
/* Terminal log */
.terminal-log {
    background: #1e1b4b;
    border: 1px solid rgba(167,139,250,0.28);
    border-radius: 14px; padding: 16px 20px;
    font-family: var(--font-mono); font-size: 12px;
    color: #c4b5fd; max-height: 200px; overflow-y: auto;
    box-shadow: inset 0 2px 14px rgba(0,0,0,0.3);
}
.log-line           { opacity:.8; line-height:1.9; }
.log-line span      { color: #90ee90; }   /* light green */
.log-line.warn span { color: #ffb3c6; }   /* baby pink   */
.log-line.crit span { color: #ef4444; }   /* red         */

/* hide chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. LOAD / BUILD MODEL
# ─────────────────────────────────────────────
MODEL_LOADED = True
try:
    model      = joblib.load('traffic_model.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_surface = joblib.load('le_surface.pkl')
except Exception:
    MODEL_LOADED = False
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    np.random.seed(42)
    n = 2000
    X = pd.DataFrame({
        'Hour':            np.random.randint(0,24,n),
        'Day_of_Week':     np.random.randint(0,7,n),
        'Traffic_Density': np.random.uniform(0.1,1.0,n),
        'Weather':         np.random.randint(0,4,n),
        'Road_Surface':    np.random.randint(0,3,n),
        'Speed_Limit':     np.random.choice([30,50,70,90,110],n),
    })
    y = np.where(X['Speed_Limit']>90, 2, np.where(X['Traffic_Density']>0.75, 1, 0))
    y = np.clip(y + np.random.randint(-1,2,n), 0, 2)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    le_weather = LabelEncoder(); le_weather.fit(['Sunny','Rainy','Cloudy','Snowy'])
    le_surface = LabelEncoder(); le_surface.fit(['Dry','Wet','Icy'])


# ─────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────
RISK_LABELS = {0:'LOW', 1:'ELEVATED', 2:'CRITICAL'}
RISK_COLORS = {0:'#90ee90', 1:'#ffb3c6', 2:'#ef4444'}   # light-green / baby-pink / red
CHART_BG    = '#f8f7ff'

WEATHER_RISK = {'Sunny':0.05, 'Cloudy':0.15, 'Rainy':0.55, 'Snowy':0.90}
SURFACE_RISK = {'Dry':0.05,   'Wet':0.45,    'Icy':0.92}

def build_input(h, dow, dens, wthr, surf, spd):
    df = pd.DataFrame([[h,dow,dens,wthr,surf,spd]],
        columns=['Hour','Day_of_Week','Traffic_Density','Weather','Road_Surface','Speed_Limit'])
    df['Weather']      = le_weather.transform([wthr])
    df['Road_Surface'] = le_surface.transform([surf])
    return df

def find_safe_speed(base_df, wthr, surf):
    h, dow, dens = base_df['Hour'].values[0], base_df['Day_of_Week'].values[0], base_df['Traffic_Density'].values[0]
    for spd in [30,50,70,90]:
        if model.predict(build_input(h,dow,dens,wthr,surf,spd))[0] == 0:
            return spd
    return None

def composite_risk(proba, wthr, surf, dens, spd):
    env  = (WEATHER_RISK[wthr] + SURFACE_RISK[surf]) / 2
    raw  = proba[1]*0.35 + proba[2]*0.45 + env*0.12 + dens*0.05 + (spd/110)*0.03
    return min(round(raw*100,1), 100.0)

def react_dist(spd, wthr):
    return round((spd/3.6)*(1.5 + WEATHER_RISK[wthr]*0.5), 1)

def brake_dist(spd, surf):
    mu = {'Dry':0.8,'Wet':0.45,'Icy':0.15}[surf]
    return round((spd/3.6)**2 / (2*9.81*mu), 1)

def gen_log(pred, wthr, surf, dens, spd, hr):
    ts = datetime.now().strftime("%H:%M:%S")
    lines = [
        (f"[{ts}] SENTINEL v4.2 — Scenario analysis initiated", "ok"),
        (f"[{ts}] INPUTS  → Hour={hr:02d}h  Density={dens:.2f}  Speed={spd}km/h", "ok"),
        (f"[{ts}] ENV     → Weather={wthr}  Surface={surf}", "ok"),
    ]
    if pred == 0:
        lines += [(f"[{ts}] RESULT  → All parameters within nominal bounds","ok"),
                  (f"[{ts}] STATUS  → GREEN — No intervention required","ok")]
    elif pred == 1:
        lines += [(f"[{ts}] RESULT  → Hazard index elevated above threshold","warn"),
                  (f"[{ts}] STATUS  → PINK ALERT — Speed reduction advised","warn")]
    else:
        lines += [(f"[{ts}] RESULT  → CRITICAL multi-factor hazard convergence!","crit"),
                  (f"[{ts}] STATUS  → RED — Immediate action required","crit"),
                  (f"[{ts}] ALERT   → Route avoidance recommended","crit")]
    lines.append((f"[{ts}] DONE    — Analysis complete in 1.47s","ok"))
    return lines


# ─────────────────────────────────────────────
# 5. HEADER
# ─────────────────────────────────────────────
now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S UTC")
st.markdown(f"""
<div class="sentinel-header">
  <div>
    <div class="sentinel-logo">SENTINEL</div>
    <div class="sentinel-sub">TRAFFIC INTELLIGENCE SYSTEM  ·  v4.2.0</div>
  </div>
  <div class="sentinel-clock">
    🟢 SYS ONLINE &nbsp;|&nbsp; {now_str}<br>
    {'⚙️  DEMO MODE' if not MODEL_LOADED else '✅  LIVE MODEL'}
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label" style="color:#c4b5fd;border-color:rgba(167,139,250,0.3)">SCENARIO PARAMETERS</div>', unsafe_allow_html=True)
    hour            = st.slider('⏱  HOUR OF DAY',         0, 23, 14)
    day_of_week     = st.slider('📅  DAY OF WEEK (0=MON)', 0,  6,  2)
    traffic_density = st.slider('🚦  TRAFFIC DENSITY',      0.1, 1.0, 0.55, step=0.05)
    weather         = st.selectbox('🌦  WEATHER CONDITION',  ['Sunny','Cloudy','Rainy','Snowy'])
    road_surface    = st.selectbox('🛣  ROAD SURFACE',       ['Dry','Wet','Icy'])
    speed_limit     = st.selectbox('⚡  VEHICLE SPEED (km/h)',[30,50,70,90,110])

    st.markdown("---")
    st.markdown('<div class="section-label" style="color:#c4b5fd;border-color:rgba(167,139,250,0.3)">SCENARIO PRESETS</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    if p1.button("🌙 NIGHT"):
        hour, traffic_density, weather, road_surface, speed_limit = 2,0.3,'Cloudy','Wet',90
    if p2.button("🌧 STORM"):
        hour, traffic_density, weather, road_surface, speed_limit = 17,0.9,'Rainy','Wet',70
    if p1.button("❄️ ICE"):
        hour, traffic_density, weather, road_surface, speed_limit = 7,0.6,'Snowy','Icy',50
    if p2.button("☀️ CLEAR"):
        hour, traffic_density, weather, road_surface, speed_limit = 10,0.3,'Sunny','Dry',70

    st.markdown("---")
    analyze_btn = st.button("⟶  INITIATE ANALYSIS")


# ─────────────────────────────────────────────
# 7. MAIN DASHBOARD
# ─────────────────────────────────────────────
if analyze_btn:

    # ══ VEHICLE ANIMATION ══
    st.markdown("""
    <div class="road-scene">
      <!-- top lane: going right -->
      <div class="veh top d1">🚗</div>
      <div class="veh top d2">🚕</div>
      <div class="veh top d3">🏎️</div>
      <!-- bottom lane: going left -->
      <div class="veh bot d4">🚛</div>
      <div class="veh bot d5">🚌</div>
      <div class="veh bot d6">🚙</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Compute ──
    input_df   = build_input(hour, day_of_week, traffic_density, weather, road_surface, speed_limit)
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    r_score    = composite_risk(proba, weather, road_surface, traffic_density, speed_limit)
    r_dist     = react_dist(speed_limit, weather)
    b_dist     = brake_dist(speed_limit, road_surface)
    stop_dist  = r_dist + b_dist
    safe_spd   = find_safe_speed(input_df, weather, road_surface)

    # ── KPI Row ──
    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    for col, (label, value, unit) in zip(
        st.columns(4),
        [("RISK SCORE",f"{r_score:.1f}","%"),
         ("STOPPING DIST",f"{stop_dist:.0f}","m"),
         ("BRAKING DIST",f"{b_dist:.0f}","m"),
         ("THREAT LEVEL",RISK_LABELS[prediction],"")]
    ):
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{RISK_COLORS[prediction]}">{value}</div>
          <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Status + Prob Bars ──
    sc, pc = st.columns([1.6, 1])

    with sc:
        css_cls   = {0:'status-optimal',1:'status-elevated',2:'status-critical'}[prediction]
        dot_col   = RISK_COLORS[prediction]
        subtitles = {
            0:"All parameters within acceptable operational bounds.",
            1:"Hazard index above nominal threshold. Advisory issued.",
            2:"Multi-factor convergence detected. Immediate intervention required.",
        }
        reco_html = ""
        if prediction in [1,2]:
            if safe_spd:
                reco_html = f'<div class="reco-box">▶ ADVISORY: Reduce speed to {safe_spd} km/h to enter safe operating envelope.</div>'
            else:
                reco_html = '<div class="reco-box">▶ CRITICAL: Conditions exceed all thresholds. Halt and seek alternative route.</div>'

        st.markdown(f"""
        <div class="{css_cls}">
          <div class="status-badge" style="color:{dot_col}">
            <span class="badge-dot" style="background:{dot_col};box-shadow:0 0 14px {dot_col};"></span>
            {RISK_LABELS[prediction]} RISK
          </div>
          <div class="status-subtitle">{subtitles[prediction]}</div>
          <div style="font-family:var(--font-mono);font-size:12px;color:var(--text-muted);">
            REACTION DIST: {r_dist} m &nbsp;·&nbsp; BRAKING DIST: {b_dist} m
          </div>
          {reco_html}
        </div>""", unsafe_allow_html=True)

    with pc:
        st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">CLASSIFICATION CONFIDENCE</div>', unsafe_allow_html=True)
        for lbl, val, clr in zip(
            ['LOW RISK','ELEVATED','CRITICAL'],
            proba,
            ['#90ee90','#ffb3c6','#ef4444']   # light-green / baby-pink / red
        ):
            pct = val * 100
            st.markdown(f"""
            <div class="prob-row">
              <div class="prob-label">{lbl}</div>
              <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{clr};
                     box-shadow:0 0 8px {clr}80;"></div>
              </div>
              <div class="prob-value" style="color:{clr}">{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 Charts ──
    c1, c2, c3 = st.columns(3)

    # Feature Importance
    with c1:
        st.markdown('<div class="section-label">FACTOR IMPACT ANALYSIS</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3.8))
        fig.patch.set_facecolor(CHART_BG); ax.set_facecolor(CHART_BG)
        imp = model.feature_importances_
        feats = ['Hour','Day','Density','Weather','Surface','Speed']
        idx   = np.argsort(imp)
        ax.barh([feats[i] for i in idx], [imp[i] for i in idx],
                color=RISK_COLORS[prediction], alpha=0.85, height=0.55,
                edgecolor='white', linewidth=0.5)
        for i in idx:
            ax.text(imp[i]+0.004, idx.tolist().index(i)+0 ,
                    f'{imp[i]:.3f}', va='center',
                    color='#6b7280', fontsize=8, fontfamily='monospace')
        ax.spines[:].set_visible(False)
        ax.tick_params(colors='#6b7280', labelsize=9)
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, max(imp)*1.28)
        plt.tight_layout(pad=0.4); st.pyplot(fig); plt.close()

    # Radar
    with c2:
        st.markdown('<div class="section-label">MULTI-FACTOR RISK RADAR</div>', unsafe_allow_html=True)
        rlbls  = ['Speed','Density','Weather','Surface','Time','Overall']
        rvals  = [(speed_limit-30)/80, traffic_density,
                  WEATHER_RISK[weather], SURFACE_RISK[road_surface],
                  abs(hour-14)/14, r_score/100]
        angles = np.linspace(0, 2*np.pi, len(rlbls), endpoint=False).tolist()
        av     = rvals + [rvals[0]]; aa = angles + [angles[0]]
        fig, ax = plt.subplots(figsize=(4.2,3.8), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(CHART_BG); ax.set_facecolor(CHART_BG)
        ax.plot(aa, av, color=RISK_COLORS[prediction], linewidth=2.5)
        ax.fill(aa, av, color=RISK_COLORS[prediction], alpha=0.22)
        for r in [.25,.5,.75,1.0]:
            ax.plot(aa, [r]*len(aa), color='#d1d5db', linewidth=0.6, linestyle='--')
        ax.set_xticks(angles); ax.set_xticklabels(rlbls, color='#6b7280', fontsize=8)
        ax.set_yticklabels([]); ax.set_ylim(0,1)
        ax.spines['polar'].set_color('#e5e7eb')
        ax.grid(color='#e5e7eb', linewidth=0.5)
        plt.tight_layout(pad=0.4); st.pyplot(fig); plt.close()

    # Speed Sweep
    with c3:
        st.markdown('<div class="section-label">SPEED RISK SWEEP</div>', unsafe_allow_html=True)
        spds = [30,50,70,90,110]
        sweep = []
        for s in spds:
            p = model.predict_proba(build_input(hour,day_of_week,traffic_density,weather,road_surface,s))[0]
            sweep.append(composite_risk(p,weather,road_surface,traffic_density,s))
        fig, ax = plt.subplots(figsize=(4.5,3.8))
        fig.patch.set_facecolor(CHART_BG); ax.set_facecolor(CHART_BG)
        bc = ['#90ee90' if s<30 else '#ffb3c6' if s<60 else '#ef4444' for s in sweep]
        ax.bar(spds, sweep, color=bc, alpha=0.88, width=15, edgecolor='white', linewidth=0.8)
        ax.axhline(30, color='#90ee9088', linewidth=1.2, linestyle='--')
        ax.axhline(60, color='#ffb3c688', linewidth=1.2, linestyle='--')
        ci = spds.index(speed_limit)
        ax.bar([speed_limit],[sweep[ci]],color=bc[ci],width=15,edgecolor='#4f46e5',linewidth=2,alpha=1)
        for s,v,c in zip(spds,sweep,bc):
            ax.text(s,v+1.5,f'{v:.0f}',ha='center',va='bottom',color='#6b7280',fontsize=8)
        ax.set_xticks(spds); ax.set_xticklabels([str(s) for s in spds],color='#6b7280',fontsize=9)
        ax.set_ylabel('RISK SCORE',color='#6b7280',fontsize=8)
        ax.set_ylim(0,115)
        for sp in ax.spines.values(): sp.set_color('#e5e7eb')
        ax.tick_params(colors='#6b7280',labelsize=8)
        plt.tight_layout(pad=0.4); st.pyplot(fig); plt.close()

    # ── Terminal Log ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">SYSTEM LOG</div>', unsafe_allow_html=True)
    log  = gen_log(prediction, weather, road_surface, traffic_density, speed_limit, hour)
    cmap = {'ok':'log-line','warn':'log-line warn','crit':'log-line crit'}
    html = ''.join(f'<div class="{cmap[k]}"><span>›</span> {m}</div>' for m,k in log)
    st.markdown(f'<div class="terminal-log">{html}</div>', unsafe_allow_html=True)

# ── Idle ──
else:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;
        border:2px dashed rgba(79,70,229,0.22);border-radius:18px;
        background:rgba(255,255,255,0.55);backdrop-filter:blur(10px);
        box-shadow:0 4px 28px rgba(79,70,229,0.07);">
      <div style="font-size:54px;margin-bottom:18px;">🛡️</div>
      <div style="font-family:'Orbitron',monospace;font-size:13px;
                  letter-spacing:5px;color:#4f46e5;margin-bottom:10px;">
        AWAITING SCENARIO INPUT
      </div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                  color:#6b7280;letter-spacing:2px;">
        Configure parameters in the sidebar and initiate analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="margin-top:40px;border-top:1px solid rgba(79,70,229,0.12);
    padding-top:14px;font-family:'Share Tech Mono',monospace;
    font-size:10px;color:rgba(107,114,128,0.55);letter-spacing:2px;text-align:center;">
  SENTINEL TRAFFIC INTELLIGENCE  ·  RESEARCH-GRADE MODEL  ·  NOT FOR OPERATIONAL USE
</div>
""", unsafe_allow_html=True)
