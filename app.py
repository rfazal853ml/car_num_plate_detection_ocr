import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from paddleocr import PaddleOCR
import tempfile
import time

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="AI Number Plate Recognition",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Load CSS from external file
# -----------------------------------------------------
def load_css(file_path):
    try:
        with open(file_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")

load_css('styles.css')

# -----------------------------------------------------
# Load models
# -----------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("model/num_plate_det_v8n.pt")

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(["en"])

@st.cache_resource
def load_paddleocr():
    return PaddleOCR(use_textline_orientation=True, lang='en')

# -----------------------------------------------------
# OCR functions
# -----------------------------------------------------
def run_easyocr(roi):
    reader = load_easyocr()
    results = reader.readtext(roi)
    return " ".join([res[1] for res in results])

def run_paddleocr(roi):
    ocr = load_paddleocr()
    results = ocr.predict(roi)
    
    # Check if results exist and if 'rec_texts' has at least one item
    if results and \
       len(results) > 0 and \
       'rec_texts' in results[0] and \
       len(results[0]['rec_texts']) > 0:
        
        return results[0]['rec_texts'][0]
        
    return ""

# -----------------------------------------------------
# YOLO detection + OCR
# -----------------------------------------------------
def detect_and_ocr(image, ocr_type):
    model = load_yolo()
    results = model(image, conf=0.75)[0]

    processed_frame = image.copy()
    cropped_images = []
    detected_texts = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        roi = image[y1:y2, x1:x2]
        cropped_images.append(roi)

        if ocr_type == "EasyOCR":
            text = run_easyocr(roi)
        else:
            text = run_paddleocr(roi)

        detected_texts.append(text)

    return processed_frame, cropped_images, detected_texts

# -----------------------------------------------------
# Compact Header
# -----------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>AI Number Plate Recognition</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------
st.sidebar.markdown("### Settings")
mode = st.sidebar.radio("Input Type", ["Image", "Video"])
ocr_type = st.sidebar.selectbox("OCR Engine", ["EasyOCR", "PaddleOCR"])

# -----------------------------------------------------
# IMAGE MODE
# -----------------------------------------------------
if mode == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2, col3 = st.columns([1, 1, 1])

        # LEFT - Original Image
        with col1:
            st.markdown('<div class="card-title">Original</div>', unsafe_allow_html=True)
            display_img = cv2.resize(img, (320, 320))
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Process
        processed_img, cropped_plates, detected_texts = detect_and_ocr(img, ocr_type)

        # MIDDLE - Detected Plates
        with col2:
            st.markdown('<div class="card-title">Detected Plates</div>', unsafe_allow_html=True)
            
            if len(cropped_plates) == 0:
                st.warning("No plates detected")
            else:
                st.markdown(f'<div class="status-badge status-success">Found {len(cropped_plates)}</div>', unsafe_allow_html=True)
                for i, plate in enumerate(cropped_plates):
                    h, w = plate.shape[:2]
                    target_height = 30
                    target_width = 100
                    resized_plate = cv2.resize(plate, (target_width, target_height))
                    st.image(cv2.cvtColor(resized_plate, cv2.COLOR_BGR2RGB), caption=f"Plate {i+1}", use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # RIGHT - OCR Results
        with col3:
            st.markdown('<div class="card-title">OCR Results</div>', unsafe_allow_html=True)
            
            if len(detected_texts) == 0:
                st.warning("No text detected")
            else:
                for i, txt in enumerate(detected_texts):
                    st.markdown(f"""
                    <div class="ocr-result">
                        <div class="plate-label">Plate {i+1}</div>
                        <div class="plate-number">{txt if txt else 'N/A'}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Upload an image to start detection")

# -----------------------------------------------------
# VIDEO MODE
# -----------------------------------------------------
elif mode == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        video_col, results_col = st.columns([2, 1])
        
        with video_col:
            st.markdown('<div class="card-title">Live Feed</div>', unsafe_allow_html=True)
            stframe = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with results_col:
            st.markdown('<div class="card-title">Real-time OCR</div>', unsafe_allow_html=True)
            fps_display = st.empty()
            ocr_box = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        fps_counter = 0
        fps_time_start = time.time()
        instant_fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fps_counter += 1

            # If 1 second has passed → update instantaneous FPS
            if time.time() - fps_time_start >= 1.0:
                instant_fps = fps_counter
                fps_counter = 0
                fps_time_start = time.time()

            frame = cv2.resize(frame, (640, 480))
            processed_frame, _, detected_texts = detect_and_ocr(frame, ocr_type)

            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            fps_display.markdown(f"""
            <div class="fps-display">
                <span class="fps-label">Processing FPS:</span>
                <span class="fps-value">{instant_fps}</span>
            </div>
            """, unsafe_allow_html=True)

            if detected_texts:
                ocr_html = ""
                for i, txt in enumerate(detected_texts):
                    ocr_html += f"""
                    <div class="ocr-result">
                        <div class="plate-label">Plate {i+1}</div>
                        <div class="plate-number">{txt if txt else 'N/A'}</div>
                    </div>
                    """
                ocr_box.markdown(ocr_html, unsafe_allow_html=True)
            else:
                ocr_box.markdown('<div class="ocr-result"><div class="plate-number">⏳ Scanning...</div></div>', unsafe_allow_html=True)

        cap.release()
        st.success("Completed!")
        
    else:
        st.info("Upload a video to start detection")