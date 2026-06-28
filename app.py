import os
import json

import streamlit as st 
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import io

from tensorflow.keras.applications.densenet     import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet_v2    import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.vgg16        import preprocess_input as vgg_preprocess

PREPROCESS_FUNCTIONS = {
    'DenseNet121':    densenet_preprocess,
    'ResNet50V2':     resnet_preprocess,
    'MobileNetV2':    mobilenet_preprocess,
    'EfficientNetB3': efficientnet_preprocess,
    'InceptionV3':    inception_preprocess,
    'VGG16':          vgg_preprocess,
}

st.set_page_config(
    page_title="SkinScan",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: #F8FAFC;}
[data-testid="stSidebar"]          {display: none;}
header[data-testid="stHeader"]     {background: transparent;}
.block-container {
    padding: 2rem 4rem 4rem 4rem;
    max-width: 1100px;
    margin: 0 auto;
}

body, p, div, span, label {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #1E293B;
}

.sk-header {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
}
.sk-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #1E40AF;
    margin: 0;
    letter-spacing: -0.5px;
}
.sk-header .sk-icon {
    font-size: 2.2rem;
    vertical-align: middle;
    margin-right: 0.4rem;
}
.sk-divider {
    border: none;
    border-top: 2px solid #DBEAFE;
    margin: 1rem auto 2rem auto;
    width: 60%;
}

.sk-upload-card {
    background: #FFFFFF;
    border: 2px dashed #93C5FD;
    border-radius: 14px;
    padding: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    box-shadow: 0 2px 12px rgba(37,99,235,0.07);
    margin-bottom: 1.5rem;
}
.sk-upload-left {text-align: center; flex: 1;}
.sk-upload-left .up-icon {font-size: 2.5rem; color: #2563EB;}
.sk-upload-left p {color: #2563EB; font-weight: 600; font-size: 1rem; margin: 0.3rem 0 0 0;}
.sk-upload-left span {font-size: 0.8rem; color: #94A3B8;}
.sk-upload-right {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: #F1F5F9;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    min-width: 220px;
}
.sk-upload-right img {
    width: 52px;
    height: 52px;
    border-radius: 6px;
    object-fit: cover;
}
.sk-upload-right .fname {font-weight: 600; font-size: 0.9rem; color: #1E293B;}
.sk-upload-right .fsize {font-size: 0.75rem; color: #64748B;}
.sk-upload-right .check {color: #10B981; font-size: 1.2rem;}

div[data-testid="stButton"] > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 4px 14px rgba(37,99,235,0.3) !important;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.4) !important;
}

.sk-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 16px rgba(15,23,42,0.07);
    margin-bottom: 1.2rem;
    height: 100%;
}
.sk-card-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.9rem;
    letter-spacing: 0.2px;
}
.sk-card-title .ct-icon {color: #2563EB; font-size: 1rem;}

.sk-img-card img {
    width: 100%;
    border-radius: 10px;
    display: block;
}
.sk-img-caption {
    font-size: 0.72rem;
    color: #94A3B8;
    margin-top: 0.7rem;
    text-align: center;
    font-style: italic;
}

.sk-pred-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
}
.sk-pred-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.sk-pred-badge {
    width: 54px; height: 54px;
    border-radius: 50%;
    background: #DBEAFE;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    flex-shrink: 0;
}
.sk-pred-label {font-size: 0.8rem; font-weight: 600; color: #2563EB; letter-spacing: 0.3px;}
.sk-pred-name  {font-size: 2rem; font-weight: 800; color: #1E293B; margin: 0; line-height: 1.1;}
.sk-conf-section {}
.sk-conf-label {font-size: 0.8rem; font-weight: 600; color: #2563EB; margin-bottom: 0.2rem;}
.sk-conf-value {font-size: 2rem; font-weight: 800; color: #10B981; margin: 0; line-height: 1.1;}

.sk-progress-wrap {
    background: #E2E8F0;
    border-radius: 999px;
    height: 8px;
    width: 220px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.sk-progress-fill {
    height: 8px;
    border-radius: 999px;
    background: #2563EB;
    transition: width 0.6s ease;
}
.sk-progress-pct {
    font-size: 0.72rem;
    color: #94A3B8;
    text-align: right;
    width: 220px;
    margin-top: 0.15rem;
}

.sk-info-text {font-size: 0.9rem; color: #475569; line-height: 1.65;}
.sk-top3-list {list-style: none; padding: 0; margin: 0;}
.sk-top3-list li {
    display: flex;
    justify-content: space-between;
    padding: 0.55rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.88rem;
}
.sk-top3-list li:last-child {border-bottom: none;}
.sk-top3-rank {color: #94A3B8; font-weight: 600; min-width: 22px;}
.sk-top3-name {flex: 1; padding: 0 0.5rem; color: #1E293B;}
.sk-top3-pct  {font-weight: 600; color: #2563EB;}

.sk-ood-banner {
    background: #FEF3C7;
    border: 1.5px solid #FCD34D;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
}
.sk-ood-banner .ood-icon {font-size: 2rem;}
.sk-ood-banner h3 {color: #92400E; font-size: 1.1rem; margin: 0.4rem 0 0.2rem 0;}
.sk-ood-banner p  {color: #78350F; font-size: 0.85rem; margin: 0;}


[data-testid="stFileUploaderLabel"] {display: none;}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}


#MainMenu                       {visibility: hidden; height: 0%;}
[data-testid="stToolbar"]       {visibility: hidden; height: 0%;}
[data-testid="stDecoration"]    {visibility: hidden; height: 0%;}
[data-testid="stStatusWidget"]  {visibility: hidden; height: 0%;}
header                          {visibility: hidden; height: 0%;}
footer                          {visibility: hidden; height: 0%;}
</style>
""", unsafe_allow_html=True)

DISEASE_INFO = {
    "Actinic Keratosis":           "A precancerous lesion caused by prolonged ultraviolet exposure. Without treatment it may progress to squamous cell carcinoma.",
    "Atopic Dermatitis":           "A chronic inflammatory skin condition causing itchy, red, and dry patches. Often linked to allergies and a compromised skin barrier.",
    "Benign Keratosis":            "A common, non-cancerous skin growth that appears waxy or scaly. Harmless but can sometimes be mistaken for other conditions.",
    "Dermatofibroma":              "A benign, firm nodule that develops in the skin, often on the legs. Usually painless and not a health concern.",
    "Melanocytic Nevus":           "A common mole formed by a cluster of pigment-producing cells. Usually benign, but changes in size, shape, or color should be monitored.",
    "Melanoma":                    "An aggressive form of skin cancer that develops from pigment-producing cells. Early detection is important because it can spread to other parts of the body.",
    "Squamous Cell Carcinoma":     "A common form of skin cancer arising from squamous cells. Usually treatable when caught early, but can spread if ignored.",
    "Tinea Ringworm Candidiasis":  "A fungal infection causing circular, scaly patches or yeast-related irritation. Spreads through contact and responds well to antifungal treatment.",
    "Vascular Lesion":             "A benign growth made up of blood vessels, such as a birthmark or hemangioma. Typically not a health risk.",
}

MODEL_DIR     = "model"
MODEL_PATH    = os.path.join(MODEL_DIR, "skin_disease_model.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")


@st.cache_resource(show_spinner=False)
def load_model_bundle():
   
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    model = keras.models.load_model(MODEL_PATH)
    return model, metadata


model, metadata = load_model_bundle()

if model is None:
    st.error(
        "Model files not found. Download `skin_disease_model.keras` and "
        "`metadata.json` from the Google Drive folder the notebook saved "
        "to (Section 15), and place them in a `model/` folder next to this script."
    )
    st.stop()

CLASS_NAMES   = metadata["class_names"]
IMG_SIZE      = metadata["img_size"]
OOD_THRESHOLD = metadata.get("ood_threshold", 0.6)
PREPROCESS_FN = PREPROCESS_FUNCTIONS[metadata["model_name"]]


def predict_image(pil_image: Image.Image) -> dict:
  
    img = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    img = PREPROCESS_FN(img)
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    top3  = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: -x[1])[:3]

    return {
        "prediction": CLASS_NAMES[idx] if conf >= OOD_THRESHOLD else "Unknown / OOD",
        "confidence": conf,
        "is_ood":     conf < OOD_THRESHOLD,
        "top_3":      top3,
    }


def _get_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def generate_gradcam(pil_image: Image.Image) -> Image.Image:
   
    img_resized = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    img_batch   = np.expand_dims(PREPROCESS_FN(img_resized.copy()), axis=0)

    last_conv_name = _get_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(tf.cast(img_batch, tf.float32))
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads   = tape.gradient(class_channel, conv_outputs)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colormap        = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap_rgb    = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized.astype(np.uint8), 0.55, colormap_rgb, 0.45, 0)
    return Image.fromarray(overlay)


def pil_to_b64_thumb(pil_img: Image.Image, size=(52, 52)) -> str:
    import base64
    thumb = pil_img.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def file_size_str(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024**2:
        return f"{n_bytes/1024:.0f} KB"
    else:
        return f"{n_bytes/1024**2:.1f} MB"


for key in ("result", "gradcam_img", "orig_img"):
    if key not in st.session_state:
        st.session_state[key] = None


st.markdown("""
<div class="sk-header">
    <h1><span class="sk-icon">🛡️</span> <span style="color:#2563EB;">Skin</span><span style="color:#1E293B;">Scan</span></h1>
</div>
<hr class="sk-divider">
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    label="upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file:
    pil_img   = Image.open(uploaded_file).convert("RGB")
    thumb_b64 = pil_to_b64_thumb(pil_img)
    fsize     = file_size_str(uploaded_file.size)
    fname     = uploaded_file.name
    st.session_state.orig_img = pil_img


    if st.session_state.result is not None:
      
        pass

    st.markdown(f"""
    <div class="sk-upload-card">
        <div class="sk-upload-left">
            <div class="up-icon">☁️</div>
            <p>Upload Skin Image</p>
            <span>JPG, JPEG, PNG</span>
        </div>
        <div class="sk-upload-right">
            <img src="data:image/png;base64,{thumb_b64}" alt="thumb"/>
            <div>
                <div class="fname">{fname}</div>
                <div class="fsize">{fsize} &nbsp;<span class="check">✅</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="sk-upload-card">
        <div class="sk-upload-left">
            <div class="up-icon">☁️</div>
            <p>Upload Skin Image</p>
            <span>JPG, JPEG, PNG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    analyze_clicked = st.button("🔍  Analyze", use_container_width=True, disabled=(uploaded_file is None))

if analyze_clicked and uploaded_file is not None:
    with st.spinner("Running AI analysis…"):
        pil_img = st.session_state.orig_img
        result  = predict_image(pil_img)
        gradcam = generate_gradcam(pil_img)
        st.session_state.result     = result
        st.session_state.gradcam_img = gradcam

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


if st.session_state.result is not None:
    result     = st.session_state.result
    gradcam    = st.session_state.gradcam_img
    orig_img   = st.session_state.orig_img

    col_orig, col_cam = st.columns(2, gap="medium")

    with col_orig:
        st.markdown("""
        <div class="sk-card sk-img-card">
            <div class="sk-card-title"><span class="ct-icon">🖼️</span> Original Image</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(orig_img, use_container_width=True)

    with col_cam:
        st.markdown("""
        <div class="sk-card sk-img-card">
            <div class="sk-card-title"><span class="ct-icon">✨</span> Grad-CAM Heatmap</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(gradcam, use_container_width=True)
        st.markdown("""
        <p style="font-size:0.72rem;color:#94A3B8;text-align:center;font-style:italic;margin-top:0.4rem;">
            Highlighted regions indicate areas that most influenced the prediction.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    if result["is_ood"]:
        st.markdown("""
        <div class="sk-ood-banner">
            <div class="ood-icon">⚠️</div>
            <h3>Unknown / Uncertain Image</h3>
            <p>The model is not sufficiently confident about this image.<br>
               Please upload a clear, close-up skin photograph.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        pred_name = result["prediction"]
        conf_pct  = result["confidence"] * 100
        bar_width = int(conf_pct)

        st.markdown(f"""
        <div class="sk-card">
            <div class="sk-pred-card">
                <div class="sk-pred-left">
                    <div class="sk-pred-badge">🎯</div>
                    <div>
                        <div class="sk-pred-label">PREDICTION</div>
                        <div class="sk-pred-name">{pred_name}</div>
                    </div>
                </div>
                <div class="sk-conf-section">
                    <div class="sk-conf-label">CONFIDENCE SCORE</div>
                    <div class="sk-conf-value">{conf_pct:.1f}%</div>
                    <div class="sk-progress-wrap">
                        <div class="sk-progress-fill" style="width:{bar_width}%"></div>
                    </div>
                    <div class="sk-progress-pct">{conf_pct:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_info, col_top3 = st.columns(2, gap="medium")

        with col_info:
            desc = DISEASE_INFO.get(pred_name, "No description available for this condition.")
            st.markdown(f"""
            <div class="sk-card">
                <div class="sk-card-title"><span class="ct-icon">📖</span> About This Condition</div>
                <p class="sk-info-text">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_top3:
            top3_html = ""
            for rank, (name, prob) in enumerate(result["top_3"], 1):
                pct = prob * 100
                top3_html += f"""<li>
                    <span class="sk-top3-rank">{rank}.</span>
                    <span class="sk-top3-name">{name}</span>
                    <span class="sk-top3-pct">{pct:.1f}%</span>
                </li>"""
            st.markdown(f"""
            <div class="sk-card">
                <div class="sk-card-title"><span class="ct-icon">📊</span> Top 3 Predictions</div>
                <ul class="sk-top3-list">
                    {top3_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)


