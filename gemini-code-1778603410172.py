import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# --- 1. Configuration (ให้ตรงกับที่เทรนไว้) ---
class Config:
    MODEL_NAME = 'efficientnetv2_rw_s'
    NUM_CLASSES = 2
    IMG_SIZE = 224
    # ระบุชื่อไฟล์โมเดลที่บันทึกไว้
    MODEL_PATH = 'best_house_recognition_model.pth' 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Load Model Functions ---
@st.cache_resource # ใช้ cache เพื่อให้โหลดโมเดลครั้งเดียว
def load_trained_model():
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=False, # ไม่ต้องโหลดน้ำหนักจาก ImageNet ใหม่ เพราะเราจะใช้ไฟล์เรา
        num_classes=Config.NUM_CLASSES
    )
    # โหลด State Dict จากไฟล์ที่คุณเซฟไว้
    try:
        state_dict = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
        model = model.to(Config.DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"❌ ไม่พบไฟล์โมเดล '{Config.MODEL_PATH}' กรุณาตรวจสอบว่าชื่อไฟล์ถูกต้อง")
        return None

# --- 3. Image Preprocessing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # เพิ่มมิติ batch

# --- 4. UI Design ---
st.set_page_config(page_title="House Recognition AI", page_icon="🏠")

st.title("🏠 House Recognition AI")
st.markdown("""
ระบบจำแนกบ้านอัตโนมัติ สำหรับโปรเจกต์ **Super AI Engineer**
โดย AI ตัวนี้จะตรวจสอบว่าภาพที่อัปโหลดคือ **บ้านหมายเลข 0007** หรือไม่
""")

# โหลดโมเดล
with st.spinner("กำลังโหลดโมเดล..."):
    model = load_trained_model()

# ส่วนอัปโหลดรูปภาพ
uploaded_file = st.file_uploader("เลือกรูปภาพบ้านที่ต้องการตรวจสอบ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงรูปภาพที่อัปโหลด
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)
    
    if st.button("🔍 ตรวจสอบภาพนี้"):
        if model is not None:
            # เตรียมภาพ
            input_tensor = preprocess_image(image).to(Config.DEVICE)
            
            # ทำนายผล
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # แสดงผลลัพธ์
            label = "✅ นี่คือบ้าน ID 0007" if predicted.item() == 1 else "❌ ไม่ใช่บ้าน ID 0007"
            prob_percent = confidence.item() * 100
            
            st.divider()
            if predicted.item() == 1:
                st.success(f"### ผลลัพธ์: {label}")
            else:
                st.error(f"### ผลลัพธ์: {label}")
                
            st.info(f"**ความมั่นใจ (Confidence):** {prob_percent:.2f}%")
            
            # แสดงกราฟความมั่นใจแบบง่าย
            st.progress(confidence.item())
        else:
            st.warning("ไม่สามารถรัน AI ได้เนื่องจากไม่ได้ติดตั้งโมเดล")

st.sidebar.markdown("---")
st.sidebar.write("💻 **Model:** EfficientNet-V2-S")
st.sidebar.write("📊 **Task:** Binary Classification")