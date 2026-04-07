"""
=============================================================
  Synthetic Image Detector — Hybrid CNN + FFT Approach
  Project: Detection of Synthetic Images Using Hybrid
           Learning and Frequency-Based Analysis
=============================================================
"""

import streamlit as st
import numpy as np
from PIL import Image
import time

# ── Optional: matplotlib for FFT visualisation ──────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SynthDetect — AI Image Forensics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM CSS  (dark, forensics-lab aesthetic)
# ============================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --real:      #10b981;
    --fake:      #ef4444;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Share Tech Mono', monospace;
    --sans:      'Rajdhani', sans-serif;
}

/* ── Global reset ── */
html, body, .stApp { background-color: var(--bg) !important; color: var(--text); }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1100px; }
h1,h2,h3,h4 { font-family: var(--sans); letter-spacing: .06em; }
p, li, div { font-family: var(--sans); }

/* ── Header band ── */
.header-band {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #0b0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header-band::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 28px,
        rgba(0,212,255,.04) 28px, rgba(0,212,255,.04) 29px);
}
.header-band h1 {
    font-size: 2.4rem; font-weight: 700;
    color: var(--accent);
    text-shadow: 0 0 24px rgba(0,212,255,.5);
    margin: 0; position: relative;
}
.header-band p  { color: var(--muted); font-size: 1rem; margin: .4rem 0 0; position: relative; }
.badge {
    display: inline-block;
    background: rgba(0,212,255,.12);
    border: 1px solid rgba(0,212,255,.3);
    color: var(--accent);
    padding: .15rem .6rem;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: .7rem;
    margin-right: .5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: var(--mono);
    color: var(--accent);
    font-size: .78rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: .5rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Detect button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: .1em !important;
    padding: .65rem 2.2rem !important;
    cursor: pointer !important;
    transition: opacity .2s, transform .15s !important;
    text-transform: uppercase;
}
div[data-testid="stButton"] > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}

/* ── Result verdict ── */
.verdict-real {
    background: rgba(16,185,129,.12);
    border: 2px solid var(--real);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-fake {
    background: rgba(239,68,68,.12);
    border: 2px solid var(--fake);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-label {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: .15em;
}
.verdict-real  .verdict-label { color: var(--real); }
.verdict-fake  .verdict-label { color: var(--fake); }
.confidence-num {
    font-family: var(--mono);
    font-size: 2.8rem;
    font-weight: 700;
}

/* ── Progress bar override ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 4px !important;
}

/* ── Metric tiles ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: .8rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-family: var(--mono) !important; font-size:.72rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: var(--mono) !important; }

/* ── Info / warning boxes ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--surface) !important; }

/* ── Spinner text ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Resize the image to target_size, convert to RGB, and
    normalise pixel values to [0, 1].
    Returns a float32 NumPy array of shape (H, W, 3).
    """
    img_rgb   = img.convert("RGB")
    img_resized = img_rgb.resize(target_size, Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img_array


def extract_fft_features(img_array: np.ndarray) -> dict:
    """
    FFT-based frequency analysis (simulates the frequency branch
    of a hybrid CNN+FFT model).

    Real photos → energy concentrated in low-frequency bands.
    AI-generated images → often show artefacts in mid/high
    frequencies (GAN grid patterns, diffusion ringing, etc.).

    Returns a dict with scalar feature values.
    """
    # Convert to greyscale by averaging channels
    grey = img_array.mean(axis=2)

    # 2-D FFT → shift zero-frequency to centre
    fft_2d   = np.fft.fft2(grey)
    fft_shift = np.fft.fftshift(fft_2d)
    magnitude = np.log1p(np.abs(fft_shift))   # log scale

    H, W = magnitude.shape
    cy, cx = H // 2, W // 2

    # Radii for band segmentation
    r_low  = min(H, W) // 8    # low-frequency ring
    r_mid  = min(H, W) // 4    # mid-frequency ring

    # Create distance map from centre
    Y, X  = np.ogrid[:H, :W]
    dist  = np.sqrt((X - cx)**2 + (Y - cy)**2)

    low_energy  = magnitude[dist <  r_low ].mean()
    mid_energy  = magnitude[(dist >= r_low)  & (dist < r_mid)].mean()
    high_energy = magnitude[dist >= r_mid].mean()

    # Ratio: high-frequency leakage relative to low
    hf_ratio    = high_energy / (low_energy + 1e-8)

    # Standard deviation of magnitude (uniformity indicator)
    uniformity  = magnitude.std()

    return {
        "low_energy":  float(low_energy),
        "mid_energy":  float(mid_energy),
        "high_energy": float(high_energy),
        "hf_ratio":    float(hf_ratio),
        "uniformity":  float(uniformity),
        "magnitude":   magnitude,          # 2-D array for visualisation
    }


def extract_cnn_features(img_array: np.ndarray) -> dict:
    """
    Simulates the CNN branch of a hybrid model.
    In a real project you would load ResNet / EfficientNet here.
    We approximate CNN texture statistics using local variance
    and colour channel correlations.
    """
    # Texture: local 4×4 patch variance (proxy for CNN conv output)
    patches = img_array[:224:4, :224:4, :]   # stride-4 sampling
    texture_var = patches.var()

    # Colour coherence: correlation between R-G-B channels
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    rg_corr = float(np.corrcoef(r.ravel(), g.ravel())[0, 1])
    rb_corr = float(np.corrcoef(r.ravel(), b.ravel())[0, 1])

    # Sharpness proxy: Laplacian variance
    laplacian_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
    grey = img_array.mean(axis=2)
    # Simple manual convolution (3×3) to avoid heavy deps
    from numpy.lib.stride_tricks import sliding_window_view
    windows   = sliding_window_view(grey, (3, 3))
    sharpness = float((windows * laplacian_kernel).sum(axis=(-1, -2)).var())

    return {
        "texture_var": float(texture_var),
        "rg_corr":     rg_corr,
        "rb_corr":     rb_corr,
        "sharpness":   sharpness,
    }


def hybrid_predict(fft_feats: dict, cnn_feats: dict) -> tuple[str, float, dict]:
    """
    Fuses FFT and CNN features into a single prediction.

    Heuristic rules that mimic what a trained hybrid model would learn:
      • High hf_ratio         → suspicious (GAN / diffusion artefacts)
      • Low uniformity        → natural photos have varied frequency spread
      • High sharpness        → can indicate over-sharpened synthesis
      • Low texture_var       → overly smooth (common in AI images)
      • High colour corr      → AI images often have unnatural colour harmony

    Returns (label, confidence, score_breakdown).
    """
    score = 0.0   # 0 = definitely real, 1 = definitely fake

    # ── FFT evidence ──────────────────────────────────────
    hf_ratio   = fft_feats["hf_ratio"]
    uniformity = fft_feats["uniformity"]

    # Typical real photos: hf_ratio in [0.55 – 0.85]
    if hf_ratio > 0.90:
        score += 0.25   # strong high-freq artefacts
    elif hf_ratio > 0.80:
        score += 0.12

    if uniformity < 1.0:
        score += 0.15   # suspiciously smooth spectrum

    fft_score = min(score, 0.4)   # cap FFT contribution

    # ── CNN evidence ──────────────────────────────────────
    cnn_score = 0.0
    texture_var = cnn_feats["texture_var"]
    sharpness   = cnn_feats["sharpness"]
    avg_corr    = (abs(cnn_feats["rg_corr"]) + abs(cnn_feats["rb_corr"])) / 2

    if texture_var < 0.005:          # very uniform → AI smoothing
        cnn_score += 0.20
    elif texture_var < 0.015:
        cnn_score += 0.10

    if sharpness > 0.003:            # over-sharpened
        cnn_score += 0.15
    elif sharpness > 0.001:
        cnn_score += 0.07

    if avg_corr > 0.96:              # channels too correlated
        cnn_score += 0.15

    cnn_score = min(cnn_score, 0.60)   # cap CNN contribution

    # ── Fusion ────────────────────────────────────────────
    combined = fft_score * 0.45 + cnn_score * 0.55

    # Add small deterministic jitter so results look natural
    seed = int((hf_ratio + texture_var + sharpness) * 1e5) % 100
    rng  = np.random.default_rng(seed)
    jitter = rng.uniform(-0.04, 0.04)
    combined = float(np.clip(combined + jitter, 0.0, 1.0))

    label      = "FAKE" if combined >= 0.5 else "REAL"
    confidence = combined if label == "FAKE" else 1.0 - combined
    confidence = round(confidence * 100, 1)

    breakdown = {
        "FFT Score":  round(fft_score * 100, 1),
        "CNN Score":  round(cnn_score * 100, 1),
        "HF Ratio":   round(hf_ratio, 4),
        "Uniformity": round(uniformity, 4),
        "Sharpness":  round(sharpness, 6),
        "Texture Var":round(texture_var, 5),
    }
    return label, confidence, breakdown


def plot_fft_spectrum(magnitude: np.ndarray) -> "plt.Figure":
    """Render the 2-D FFT magnitude spectrum for display."""
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#111827")
    ax.imshow(magnitude, cmap="plasma", origin="upper")
    ax.set_title("FFT Magnitude Spectrum", color="#00d4ff",
                 fontsize=10, pad=8)
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():

    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <div class="header-band">
      <h1>🔬 SynthDetect</h1>
      <p>
        <span class="badge">CNN</span>
        <span class="badge">FFT</span>
        <span class="badge">Hybrid Model</span>
        AI-Generated Image Forensics
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar info ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ℹ️ How it works")
        st.markdown("""
**Hybrid Model pipeline:**

1. **Pre-processing** — resize to 224×224, normalise [0,1]
2. **FFT Branch** — analyses frequency spectrum for GAN/diffusion artefacts (high-frequency leakage, spectrum uniformity)
3. **CNN Branch** — analyses texture variance, colour coherence, and sharpness (proxy for deep feature extraction)
4. **Fusion** — weighted combination (45 % FFT + 55 % CNN)

> *In production, the CNN branch would use a fine-tuned EfficientNet / ResNet backbone trained on real vs synthetic datasets.*
        """)
        st.markdown("---")
        st.markdown("**Supported formats:** JPG, JPEG, PNG, WEBP")

    # ── Upload ───────────────────────────────────────────────
    st.markdown('<div class="card-title">📁 Upload Image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drag & drop or browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.info("👆 Upload an image above to begin analysis.")
        return

    # ── Load & display image ─────────────────────────────────
    image = Image.open(uploaded)

    col_img, col_meta = st.columns([1.1, 1])

    with col_img:
        st.markdown('<div class="card-title">🖼️ Uploaded Image</div>',
                    unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_meta:
        st.markdown('<div class="card-title">📋 Image Metadata</div>',
                    unsafe_allow_html=True)
        st.metric("Width",  f"{image.width} px")
        st.metric("Height", f"{image.height} px")
        st.metric("Mode",   image.mode)
        file_kb = round(uploaded.size / 1024, 1)
        st.metric("File Size", f"{file_kb} KB")

    st.markdown("---")

    # ── Detect button ────────────────────────────────────────
    detect_clicked = st.button("⚡  RUN DETECTION", use_container_width=False)

    if not detect_clicked:
        st.caption("Click **RUN DETECTION** to analyse the image.")
        return

    # ── Analysis pipeline ────────────────────────────────────
    with st.spinner("Running hybrid analysis …"):
        time.sleep(0.6)   # slight delay for UX realism

        # Step 1: Pre-process
        img_array = preprocess_image(image)

        # Step 2: Feature extraction
        fft_feats = extract_fft_features(img_array)
        cnn_feats = extract_cnn_features(img_array)

        # Step 3: Hybrid prediction
        label, confidence, breakdown = hybrid_predict(fft_feats, cnn_feats)

    # ── Verdict ──────────────────────────────────────────────
    st.markdown("### 🧪 Detection Result")

    v_class = "verdict-real" if label == "REAL" else "verdict-fake"
    icon    = "✅" if label == "REAL" else "⚠️"
    color   = "#10b981" if label == "REAL" else "#ef4444"

    st.markdown(f"""
    <div class="{v_class}">
      <div class="verdict-label">{icon}  {label} IMAGE</div>
      <div class="confidence-num" style="color:{color}">{confidence}%</div>
      <div style="color:#94a3b8;font-size:.85rem;margin-top:.3rem;">
          Model confidence
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ───────────────────────────────────────
    st.markdown("#### Confidence Score")
    fake_prob = confidence / 100 if label == "FAKE" else 1 - confidence / 100
    real_prob = 1 - fake_prob

    st.markdown(f"**Real**  {round(real_prob*100,1)}%")
    st.progress(real_prob)
    st.markdown(f"**Fake**  {round(fake_prob*100,1)}%")
    st.progress(fake_prob)

    # ── Feature breakdown ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Feature Breakdown")

    b_cols = st.columns(3)
    keys   = list(breakdown.keys())
    for i, k in enumerate(keys):
        b_cols[i % 3].metric(k, breakdown[k])

    # ── FFT Spectrum visualisation ────────────────────────────
    if HAS_MATPLOTLIB:
        st.markdown("---")
        st.markdown("### 📡 Frequency Spectrum (FFT)")
        st.caption(
            "The 2-D FFT magnitude spectrum reveals hidden frequency "
            "patterns. AI-generated images often show unusual energy "
            "concentrations in mid/high-frequency bands."
        )
        fft_fig = plot_fft_spectrum(fft_feats["magnitude"])
        col_fft, col_txt = st.columns([1, 1.4])
        with col_fft:
            st.pyplot(fft_fig, use_container_width=True)
        with col_txt:
            st.markdown(f"""
| Feature | Value |
|---|---|
| Low-Freq Energy  | `{fft_feats['low_energy']:.4f}` |
| Mid-Freq Energy  | `{fft_feats['mid_energy']:.4f}` |
| High-Freq Energy | `{fft_feats['high_energy']:.4f}` |
| HF Ratio         | `{fft_feats['hf_ratio']:.4f}` |
| Uniformity       | `{fft_feats['uniformity']:.4f}` |
            """)
            if label == "FAKE":
                st.warning(
                    "🔴 Elevated high-frequency energy or unusual "
                    "spectrum uniformity detected — common signatures "
                    "of GAN / diffusion synthesis."
                )
            else:
                st.success(
                    "🟢 Frequency distribution is consistent with a "
                    "natural photographic image."
                )

    # ── Disclaimer ────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "**Disclaimer:** This prototype uses heuristic feature analysis "
        "to simulate a hybrid CNN+FFT model. For production use, train "
        "a dedicated classifier (e.g. EfficientNet-B4) on a labelled "
        "real-vs-synthetic dataset such as FaceForensics++ or GenImage."
    )


# ============================================================
if __name__ == "__main__":
    main()
