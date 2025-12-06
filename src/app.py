from pathlib import Path
import streamlit as st
from PIL import Image
from helper import load_efficientnet_model, predict_freshness, denormalize_image

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="FreshHarvest AI Inspector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

# ============================================
# Custom CSS for Better UI
# ============================================
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #2E7D32;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }

    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2em;
    }

    /* Result card styling */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    /* Fresh result */
    .fresh-result {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
    }

    /* Spoiled result */
    .spoiled-result {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
    }

    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 20px 0;
    }

    /* Upload area styling */
    .upload-area {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background: #F1F8F4;
        margin: 20px 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fruit grid styling */
    .fruit-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 20px 0;
    }

    .fruit-item {
        text-align: center;
        padding: 10px;
        background: #f5f5f5;
        border-radius: 8px;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# Load Model (Cache it so it only loads once)
# ============================================
@st.cache_resource
def load_model():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / 'notebooks' / 'best_efficientnet_b0_transfer.pth'
    model = load_efficientnet_model(str(model_path), num_classes=2)
    return model


model = load_model()

# ============================================
# Sidebar - Hamburger Menu
# ============================================
with st.sidebar:
    st.title("📱 Menu")
    st.markdown("---")

    st.subheader("📊 Model Information")
    st.markdown("""
    **Architecture:** EfficientNet-B0  
    **Training Method:** Transfer Learning  
    **Dataset:** 8 fruit types  
    **Accuracy:** 98%+  
    """)

    st.markdown("---")

    st.subheader("🍎 Supported Fruits")
    fruits = {
        "🍌": "Banana",
        "🍋": "Lemon",
        "🥭": "Mango",
        "🍊": "Orange",
        "🍓": "Strawberry",
        "🍅": "Tomato",
        "🫐": "Tamarillo",
        "🟢": "Lulo"
    }

    for emoji, name in fruits.items():
        st.markdown(f"{emoji} **{name}**")

    st.markdown("---")

    st.subheader("ℹ️ About")
    st.markdown("""
    FreshHarvest AI uses deep learning to automatically 
    classify fruits as fresh or spoiled, helping improve 
    quality control and reduce waste.

    **Technology Stack:**
    - PyTorch
    - EfficientNet-B0
    - Streamlit
    - Computer Vision
    """)

    st.markdown("---")
    st.caption("© 2025 FreshHarvest Logistics")

# ============================================
# Main Content
# ============================================

# Header
st.markdown('<p class="main-title">🍎 FreshHarvest AI Quality Inspector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered freshness detection for fruits and vegetables</p>',
            unsafe_allow_html=True)

# Instructions
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    ### 📸 How to Use:
    1. Click **Browse files** below to upload a fruit image
    2. Wait for AI analysis (takes 1-2 seconds)
    3. View the freshness prediction and confidence score
    """)

st.markdown("---")

# File uploader in center with custom styling
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Upload Fruit Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

# ============================================
# Prediction Section
# ============================================
if uploaded_file is not None:
    st.markdown("---")

    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    # Create two columns for image and result
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True, caption="Original Image")

    # Predict
    with st.spinner('🔍 Analyzing fruit quality...'):
        prediction, confidence = predict_freshness(image, model)

    # Display result in second column
    with col2:
        st.subheader("🎯 Analysis Result")

        # Create result card
        if prediction == "Fresh":
            st.markdown(f"""
            <div class="result-card fresh-result">
                <h2 style="color: #2E7D32; margin: 0;">✅ FRESH</h2>
                <h3 style="color: #1B5E20; margin-top: 10px;">Confidence: {confidence:.2f}%</h3>
                <p style="margin-top: 15px; color: #33691E;">
                    This fruit is in good condition and safe for distribution.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(confidence / 100)

        else:
            st.markdown(f"""
            <div class="result-card spoiled-result">
                <h2 style="color: #C62828; margin: 0;">❌ SPOILED</h2>
                <h3 style="color: #B71C1C; margin-top: 10px;">Confidence: {confidence:.2f}%</h3>
                <p style="margin-top: 15px; color: #D84315;">
                    This fruit shows signs of spoilage. Do not distribute.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Progress bar
            st.progress(confidence / 100)

    # Additional Details Section
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Prediction",
            value=prediction,
            delta="Good" if prediction == "Fresh" else "Bad",
            delta_color="normal" if prediction == "Fresh" else "inverse"
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence:.1f}%",
            delta=f"{confidence - 50:.1f}% above baseline"
        )

    with col3:
        st.metric(
            label="Model",
            value="EfficientNet-B0"
        )

    with col4:
        st.metric(
            label="Processing Time",
            value="< 2s"
        )

    # Technical Details (Expandable)
    with st.expander("🔬 Technical Details"):
        st.markdown("""
        **Model Architecture:** EfficientNet-B0 with Transfer Learning  
        **Input Size:** 224x224 pixels  
        **Preprocessing:** ImageNet normalization  
        **Framework:** PyTorch  
        **Training Dataset:** 8 fruit types (Fresh & Spoiled)  
        **Validation Accuracy:** 98%+  

        **How it works:**
        1. Image is resized to 224x224 pixels
        2. Normalized using ImageNet statistics
        3. Fed through EfficientNet-B0 backbone
        4. Custom classifier head produces binary prediction
        5. Softmax activation provides confidence scores
        """)

# ============================================
# Bottom Section - Features
# ============================================
else:
    st.markdown("---")
    st.markdown("### 🌟 Key Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #E8F5E9; border-radius: 10px;">
            <h2 style="color: #2E7D32;">⚡</h2>
            <h4 style="color: #000; margin: 10px 0;">Fast</h4>
            <p style="font-size: 0.9em; color: #333;">Results in under 2 seconds</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #E3F2FD; border-radius: 10px;">
            <h2 style="color: #1976D2;">🎯</h2>
            <h4 style="color: #000; margin: 10px 0;">Accurate</h4>
            <p style="font-size: 0.9em; color: #333;">98%+ accuracy rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #FFF3E0; border-radius: 10px;">
            <h2 style="color: #F57C00;">🤖</h2>
            <h4 style="color: #000; margin: 10px 0;">AI-Powered</h4>
            <p style="font-size: 0.9em; color: #333;">Deep learning technology</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #F3E5F5; border-radius: 10px;">
            <h2 style="color: #7B1FA2;">📱</h2>
            <h4 style="color: #000; margin: 10px 0;">Easy to Use</h4>
            <p style="font-size: 0.9em; color: #333;">Simple drag & drop</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Supported fruits grid
    st.markdown("### 🍎 Supported Fruits")

    col1, col2, col3, col4 = st.columns(4)
    fruits_list = [
        ("🍌", "Banana"),
        ("🍋", "Lemon"),
        ("🥭", "Mango"),
        ("🍊", "Orange"),
        ("🍓", "Strawberry"),
        ("🍅", "Tomato"),
        ("🫐", "Tamarillo"),
        ("🟢", "Lulo")
    ]

    for i, (emoji, name) in enumerate(fruits_list):
        with [col1, col2, col3, col4][i % 4]:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px; margin: 5px 0;">
                <h2 style="margin: 0;">{emoji}</h2>
                <p style="margin: 5px 0; font-weight: bold;">{name}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>FreshHarvest Logistics AI Quality Inspector</strong></p>
    <p>Powered by EfficientNet-B0 Deep Learning | Built with PyTorch & Streamlit</p>
    <p style="font-size: 0.9em;">© 2025 FreshHarvest Logistics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)