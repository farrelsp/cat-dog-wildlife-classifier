import streamlit as st
import math
import time
import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

st.set_page_config(page_title="Animal Face Classifier", page_icon="🐶")

# ----- Configuration -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Cat', 'Dog', 'Wild']

# ----- Load Model -----
@st.cache_resource
def load_model():
  model = models.efficientnet_b0(pretrained=False)
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
  model.load_state_dict(torch.load("models/effnet_best.pth", map_location=device))
  model.to(device)
  model.eval()
  return model

model = load_model()

# ----- Image Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
  
# ----- Streamlit UI -----
st.title("🐾 Animal Face Classifier")
st.write("Upload one or more images of a cat, dog, or wild animal face.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
  num_images = len(uploaded_files)
  num_cols = min(3, num_images)  # Up to 3 columns for aesthetic
  cols = st.columns(num_cols)
  
  progress = st.progress(0)
  
  for idx, uploaded_file in enumerate(uploaded_files):
    with st.spinner(f"Processing image {idx+1}/{len(uploaded_files)}..."):
      image = Image.open(uploaded_file).convert("RGB")
      input_tensor = transform(image).unsqueeze(0)
    
      # Inference
      start_time = time.time()
      with torch.no_grad():
          outputs = model(input_tensor)
          probs = torch.softmax(outputs, dim=1)
          conf, pred = torch.max(probs, 1)
      inference_time = time.time() - start_time
      
    # Select column and display result
    col = cols[idx % num_cols]  # Distribute across columns
    with col:
      st.image(image, caption=uploaded_file.name, use_container_width=True)
      st.markdown(f"**Prediction:** {class_names[pred.item()]}")
      st.markdown(f"**Confidence:** {conf.item():.2f}")
      st.markdown(f"**Inference Time:** {inference_time:.3f} s")
        
    progress.progress((idx + 1) / len(uploaded_files))
  
  st.success("✅ All images processed.")