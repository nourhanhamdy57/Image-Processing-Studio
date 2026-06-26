import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="Image Processing Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .main-title {
        font-size: 24px; font-weight: bold; color: #ffffff;
        margin-bottom: 20px; border-bottom: 1px solid #21262d; padding-bottom: 10px;
    }
    .image-panel {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 6px; padding: 15px; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌌 Image Processing  </div>', unsafe_allow_html=True)

# 2. Sidebar Layout
st.sidebar.markdown("### OPERATIONS")
uploaded_file = st.sidebar.file_uploader("Upload Primary Image", type=["png", "jpg", "jpeg"], key="main_img")

st.sidebar.markdown("### SELECT OPERATION CATEGORY")
category = st.sidebar.radio(
    "Categories:",
    ["View Original", "Geometric Ops", "Histogram Ops", "Arithmetic Ops", "Filtering", "Detection"]
)

# Initialize operation variables
geom_op = "None"
hist_op = "None"
arith_op = "None"
filter_op = "None"
detect_op = "None"

if category == "Geometric Ops":
    geom_op = st.sidebar.selectbox("Geometric:", ["Translation", "Rotation", "Scaling"])
elif category == "Histogram Ops":
    hist_op = st.sidebar.selectbox("Histogram:", ["Histogram Graph", "Equalization", "Thresholding"])
elif category == "Arithmetic Ops":
    arith_op = st.sidebar.selectbox("Arithmetic:", ["Addition", "Subtraction", "Multiplication", "Division"])
elif category == "Filtering":
    filter_op = st.sidebar.selectbox("Filtering:", ["Median Filter"])
elif category == "Detection":
    detect_op = st.sidebar.selectbox("Detection:", ["Canny Edge", "Shape Detection"])

# 3. Main Logic Execution
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, channels = img.shape
    
    processed_rgb = img_rgb.copy()
    current_operation = "Original Image"
    show_histogram = False

    # --- GEOMETRIC ---
    if geom_op == "Translation":
        current_operation = "Dynamic Translation"
        st.sidebar.markdown("---")
        st.sidebar.markdown("##### Translation Settings")
        tx = st.sidebar.slider("Shift X (Horizontal)", min_value=-cols, max_value=cols, value=50, step=5)
        ty = st.sidebar.slider("Shift Y (Vertical)", min_value=-rows, max_value=rows, value=50, step=5)
        
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        processed_rgb = cv2.warpAffine(img_rgb, M, (cols, rows))
        
    elif geom_op == "Rotation":
        current_operation = "Dynamic Rotation"
        st.sidebar.markdown("---")
        st.sidebar.markdown("##### Rotation Settings")
        angle = st.sidebar.slider("Angle (Degrees)", min_value=0, max_value=360, value=45, step=1)
        scale_rot = st.sidebar.slider("Scale inside Rotation", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale_rot)
        processed_rgb = cv2.warpAffine(img_rgb, M, (cols, rows))
        
    elif geom_op == "Scaling":
        current_operation = "Dynamic Scaling"
        st.sidebar.markdown("---")
        st.sidebar.markdown("##### Scaling Settings")
        scale_factor = st.sidebar.slider("Scale Factor (fx & fy)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        processed_rgb = cv2.resize(img_rgb, None, fx=scale_factor, fy=scale_factor)

    # --- HISTOGRAM ---
    elif hist_op == "Histogram Graph":
        current_operation = "Histogram Graph"
        show_histogram = True
    elif hist_op == "Equalization":
        current_operation = "Histogram Equalization"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        processed_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    elif hist_op == "Thresholding":
        current_operation = "Thresholding"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # --- FILTERING ---
    elif filter_op == "Median Filter":
        current_operation = "Median Filter"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.medianBlur(gray, 5)
        processed_rgb = cv2.cvtColor(median, cv2.COLOR_GRAY2RGB)

    # --- DETECTION ---
    elif detect_op == "Canny Edge":
        current_operation = "Canny Edge Detection"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    elif detect_op == "Shape Detection":
        current_operation = "Shape Detection"
        img_contours = img.copy()
        gray = cv2.cvtColor(img_contours, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            shape = "Triangle" if len(approx) == 3 else "Rectangle" if len(approx) == 4 else "Circle" if len(approx) > 6 else "Unknown"
            cv2.putText(img_contours, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        processed_rgb = cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)

    # --- ARITHMETIC ---
    elif arith_op != "None":
        st.sidebar.markdown("---")
        st.sidebar.markdown("##### Upload Second Image for Arithmetic")
        uploaded_file2 = st.sidebar.file_uploader("Choose Second Image", type=["png", "jpg", "jpeg"], key="sec_img")
        
        if uploaded_file2 is not None:
            file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
            img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
            img2 = cv2.resize(img2, (cols, rows))
            
            if arith_op == "Addition":
                current_operation = "Arithmetic: Addition"
                res = cv2.add(img, img2)
            elif arith_op == "Subtraction":
                current_operation = "Arithmetic: Subtraction"
                res = cv2.subtract(img, img2)
            elif arith_op == "Multiplication":
                current_operation = "Arithmetic: Multiplication"
                res = cv2.multiply(img2, 1.5)
            elif arith_op == "Division":
                current_operation = "Arithmetic: Division"
                res = cv2.divide(img2, 2)
                
            processed_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        else:
            current_operation = "Waiting for Second Image..."

    # --- UI Columns Rendering ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-panel">🔹 <b>Original Image</b></div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
        st.markdown(f"""
        <p style="text-align:center; color:#8b949e; font-size:13px;">
        Dimensions: {cols} &times; {rows} px &nbsp;|&nbsp; Channels: {channels} &nbsp;|&nbsp; Mean Intensity: {round(np.mean(img_rgb), 1)}
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="image-panel">🟢 <b>Processed Image</b> <span style="float:right; color:#58a6ff; font-size:12px;">{current_operation}</span></div>', unsafe_allow_html=True)
        
        if show_histogram:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            fig.patch.set_facecolor('#161b22')
            ax.set_facecolor('#161b22')
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ax.hist(gray_img.ravel(), bins=256, range=[0,256], color='#58a6ff')
            ax.title.set_color('#c9d1d9')
            ax.tick_params(colors='#c9d1d9')
            st.pyplot(fig)
        elif arith_op != "None" and uploaded_file2 is None:
            st.warning("Please upload the second image from the sidebar to perform arithmetic operation.")
        else:
            st.image(processed_rgb, use_container_width=True)
            
        st.markdown("<p style='text-align:center; color:#58a6ff; font-size:13px;'>Status: Active</p>", unsafe_allow_html=True)

else:
    st.info("Please upload a primary image from the sidebar to begin.")