import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io
import os

from data.ela import compute_ela
from gradcam import GradCAM, overlay_heatmap, parse_tamper_zone, get_gradcam_target_layer
from inference import preprocess
from model_factory import load_model_from_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT  = "models/imgforge_swin_tiny_ela_384.pth"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="ImgForge — Image Forgery Detector",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1 { color: #4A90E2; font-size: 2.4rem; font-weight: 800; }
    h3 { color: #E2E8F0; }
    .metric-card {
        background: #1E2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2D3748;
    }
    .forged-badge {
        background: #C53030;
        color: white;
        padding: 10px 28px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    .authentic-badge {
        background: #276749;
        color: white;
        padding: 10px 28px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    .zone-tag {
        background: #2D3748;
        color: #90CDF4;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .info-box {
        background: #1A1F2E;
        border-left: 4px solid #4A90E2;
        padding: 14px 18px;
        border-radius: 6px;
        color: #A0AEC0;
        font-size: 0.9rem;
    }
    div[data-testid="stFileUploader"] {
        background: #1E2130;
        border: 2px dashed #4A90E2;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = load_model_from_checkpoint(ckpt, DEVICE)
    model.eval()
    cfg = {**ckpt.get("cfg", {}), "decision_threshold": ckpt.get("decision_threshold", 0.5)}
    return model, ckpt["val_f1"], ckpt["epoch"], cfg


def preprocess_pil(uploaded, tmp_path: str, cfg=None):
    """Write the upload unchanged (re-encoding it would rewrite the history ELA measures)."""
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getvalue())
    cfg = cfg or {}
    tensor, rgb_resized = preprocess(tmp_path, DEVICE, cfg.get("image_size", 224),
                                     cfg.get("input_mode", "rgb_ela"))
    return tensor, rgb_resized, compute_ela(tmp_path, use_cache=False)


def run_inference(model, tensor, threshold=0.5):
    tensor.requires_grad_(True)
    target_layer = get_gradcam_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    with torch.enable_grad():
        output = model(tensor)

    probs = torch.softmax(output, dim=1)[0]
    pred_class = int(probs[1].item() >= threshold)
    confidence = probs[pred_class].item()
    forged_prob = probs[1].item()

    cam = gradcam.generate(tensor, class_idx=1)
    zone_info = parse_tamper_zone(cam)

    return pred_class, confidence, forged_prob, cam, zone_info


def make_ela_display(ela: np.ndarray) -> np.ndarray:
    ela_bright = np.clip(ela * 3.0, 0, 1)
    return (ela_bright * 255).astype(np.uint8)


def make_heatmap(rgb_vis: np.ndarray, cam: np.ndarray) -> np.ndarray:
    original = (rgb_vis * 255).astype(np.uint8)
    return overlay_heatmap(original, cam, alpha=0.55)


def zone_bar_chart(zone_scores: dict):
    zones = list(zone_scores.keys())
    scores = [zone_scores[z] for z in zones]
    top_zone = max(zone_scores, key=zone_scores.get)
    colors = ["#E53E3E" if z == top_zone else "#4A90E2" for z in zones]

    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="#1E2130")
    ax.set_facecolor("#1E2130")
    bars = ax.barh(zones, scores, color=colors, height=0.55)
    ax.set_xlim(0, max(scores) * 1.3 if max(scores) > 0 else 1)
    ax.tick_params(colors="#A0AEC0", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel("Activation Score", color="#A0AEC0", fontsize=9)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", color="#E2E8F0", fontsize=8)

    plt.tight_layout()
    return fig


def main():
    st.markdown("# 🔍 ImgForge")
    st.markdown("#### Image Forgery Detection & Tamper Localization")
    st.markdown("---")

    try:
        model, best_f1, best_epoch, cfg = load_model()
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""<div class='metric-card'>
                <div style='color:#68D391;font-size:0.8rem;'>MODEL STATUS</div>
                <div style='color:#E2E8F0;font-size:1.1rem;font-weight:700;'>✅ Loaded</div>
            </div>""", unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""<div class='metric-card'>
                <div style='color:#68D391;font-size:0.8rem;'>BEST VAL F1</div>
                <div style='color:#E2E8F0;font-size:1.1rem;font-weight:700;'>{best_f1:.4f}</div>
            </div>""", unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""<div class='metric-card'>
                <div style='color:#68D391;font-size:0.8rem;'>DEVICE</div>
                <div style='color:#E2E8F0;font-size:1.1rem;font-weight:700;'>{DEVICE.upper()}</div>
            </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    st.markdown("---")
    st.markdown("### 📂 Upload Image")
    st.markdown("""<div class='info-box'>
        Upload any image to check for digital forgery — copy-move, splicing, or inpainting artifacts.
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    uploaded = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded is None:
        st.markdown("""<div style='text-align:center;color:#4A5568;
            padding:60px;font-size:1rem;'>
            ↑ Upload an image to begin analysis
        </div>""", unsafe_allow_html=True)
        return

    pil_image = Image.open(uploaded).convert("RGB")
    os.makedirs("outputs", exist_ok=True)
    tmp_path = f"outputs/_tmp_{uploaded.name}"

    with st.spinner("Running ImgForge analysis..."):
        try:
            tensor, rgb_vis, ela = preprocess_pil(uploaded, tmp_path, cfg)
            pred_class, confidence, forged_prob, cam, zone_info = run_inference(
                model, tensor, cfg["decision_threshold"])
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    label = "FORGED" if pred_class == 1 else "AUTHENTIC"
    badge_class = "forged-badge" if pred_class == 1 else "authentic-badge"
    verdict_icon = "🚨" if pred_class == 1 else "✅"

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#A0AEC0;font-size:0.75rem;margin-bottom:8px;'>VERDICT</div>
            <span class='{badge_class}'>{verdict_icon} {label}</span>
        </div>""", unsafe_allow_html=True)

    with r2:
        bar_color = "#E53E3E" if pred_class == 1 else "#38A169"
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#A0AEC0;font-size:0.75rem;margin-bottom:8px;'>CONFIDENCE</div>
            <div style='color:#E2E8F0;font-size:2rem;font-weight:800;'>{confidence*100:.1f}%</div>
            <div style='background:#2D3748;border-radius:4px;height:6px;margin-top:8px;'>
                <div style='background:{bar_color};width:{confidence*100:.0f}%;
                    height:6px;border-radius:4px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#A0AEC0;font-size:0.75rem;margin-bottom:8px;'>FORGED PROBABILITY</div>
            <div style='color:#FC8181;font-size:2rem;font-weight:800;'>{forged_prob*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with r4:
        st.markdown(f"""<div class='metric-card'>
            <div style='color:#A0AEC0;font-size:0.75rem;margin-bottom:8px;'>TAMPER ZONE</div>
            <div style='margin-top:6px;'>
                <span class='zone-tag'>📍 {zone_info['top_zone'].replace('_', ' ').upper()}</span>
            </div>
            <div style='color:#718096;font-size:0.75rem;margin-top:8px;'>
                score: {zone_info['zone_confidence']:.3f}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 🖼️ Visual Analysis")
    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown("**Original Image**")
        st.image(pil_image, use_container_width=True)

    with v2:
        st.markdown("**ELA Map** — Compression artifact analysis")
        ela_display = make_ela_display(ela)
        ela_pil = Image.fromarray(
            cv2.resize(ela_display, (pil_image.width, pil_image.height))
        )
        st.image(ela_pil, use_container_width=True)

    with v3:
        st.markdown("**Grad-CAM Heatmap** — Red = Suspicious region")
        heatmap = make_heatmap(rgb_vis, cam)
        heatmap_pil = Image.fromarray(
            cv2.resize(heatmap, (pil_image.width, pil_image.height))
        )
        st.image(heatmap_pil, use_container_width=True)

    st.markdown("### 📍 Tamper Zone Breakdown")
    z1, z2 = st.columns([1.2, 1])

    with z1:
        zone_fig = zone_bar_chart(zone_info["zone_scores"])
        st.pyplot(zone_fig, use_container_width=True)
        plt.close()

    with z2:
        st.markdown("""<div class='info-box' style='margin-top:20px;'>
            <b style='color:#90CDF4;'>How zone detection works:</b><br><br>
            The image is divided into 6 regions. Grad-CAM activation
            scores indicate which region the model focused on when making
            its forgery decision. Higher scores = more suspicious activity
            in that zone.
        </div>""", unsafe_allow_html=True)

        if pred_class == 1:
            st.markdown(f"""<div class='info-box' style='margin-top:12px;
                border-left-color:#E53E3E;'>
                🚨 <b style='color:#FC8181;'>Forgery detected</b> in the
                <b style='color:#FEB2B2;'>
                {zone_info['top_zone'].replace('_', ' ').upper()}</b> region
                with activation score
                <b style='color:#FEB2B2;'>{zone_info['zone_confidence']:.3f}</b>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class='info-box' style='margin-top:12px;
                border-left-color:#38A169;'>
                ✅ <b style='color:#68D391;'>No significant forgery indicators
                detected.</b>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style='text-align:center;color:#4A5568;font-size:0.8rem;'>
        ImgForge — EfficientNet-B0 + ELA Fusion + Grad-CAM Localization<br>
        Built by Mayur Das · MIT Manipal, MAHE · github.com/MayurDas24
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()