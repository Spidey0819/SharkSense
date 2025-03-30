import streamlit as st
import os
import io
import zipfile
from PIL import Image, ImageDraw, ImageFont
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Shark Species Classifier",
    page_icon="🦈",
    layout="wide"
)

# Classes in the same order as training
classes = [
    'basking', 'blacktip', 'blue', 'bull', 'hammerhead',
    'lemon', 'mako', 'nurse', 'sand tiger', 'thresher',
    'tiger', 'whale', 'white', 'whitetip'
]

# Try to import torch-related dependencies
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from model import SharkClassifier
    
    # Define your image transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("⚠️ PyTorch dependencies could not be loaded. Running in demo mode.")

@st.cache_resource
def load_model():
    """Load the model once and cache it"""
    if not TORCH_AVAILABLE:
        return None, None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = SharkClassifier(num_classes=14)
    try:
        model.load_state_dict(torch.load('model_phase2_full.pth', map_location=device))
        logger.info("Model weights loaded successfully.")
    except FileNotFoundError:
        st.error("Model file 'model_phase2_full.pth' not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model weights: {str(e)}")
        return None, None
    
    model.to(device)
    model.eval()
    return model, device

def label_image(pil_image, text):
    """
    Draws the given text at the top-left corner of the image.
    """
    # Convert to RGBA for text overlay
    image_rgba = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(image_rgba)
    
    # Use default PIL font
    font = ImageFont.load_default()

    # Choose text position and color
    position = (20, 20)
    
    # Draw text with a darker background for visibility
    text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (100, 20)
    draw.rectangle(
        [position[0]-5, position[1]-5, position[0]+text_width+5, position[1]+text_height+5],
        fill=(0, 0, 0, 128)
    )
    
    # Draw text
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return image_rgba.convert("RGB")

def predict_image(image, model, device):
    """Make prediction for a single image"""
    if not TORCH_AVAILABLE or model is None:
        # Return mock prediction in demo mode
        import random
        predictions = []
        used_indices = set()
        
        # Get three random classes with decreasing confidence
        for i in range(3):
            idx = random.randint(0, len(classes)-1)
            while idx in used_indices:
                idx = random.randint(0, len(classes)-1)
            used_indices.add(idx)
            
            confidence = max(0.4, 0.9 - (i * 0.15) + (random.random() * 0.1))
            predictions.append((classes[idx], confidence))
        
        return predictions
    
    # Real prediction with model
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
    
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3)
    predictions = []
    
    for i in range(3):
        label = classes[top_indices[0][i].item()]
        confidence = top_probs[0][i].item()
        predictions.append((label, confidence))
    
    return predictions

def main():
    st.title("🦈 Shark Species Classifier")
    
    if not TORCH_AVAILABLE:
        st.warning("""
        **Running in Demo Mode**
        
        PyTorch dependencies could not be loaded. The app is running in demo mode with random predictions.
        For full functionality, update requirements.txt with compatible versions and redeploy.
        """)
    
    st.write("Upload shark images and get species predictions!")
    
    # Load model if possible
    model, device = load_model()
    
    # Sidebar for mode selection
    mode = st.sidebar.radio(
        "Choose mode:",
        ("Single Image", "Multiple Images")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About")
    st.sidebar.info(
        "This app uses a ResNet50 model trained to identify "
        "14 different species of sharks from images."
    )
    
    if mode == "Single Image":
        uploaded_file = st.file_uploader("Choose a shark image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner("Analyzing image..."):
                    predictions = predict_image(image, model, device)
                
                st.success("Analysis complete!")
                
                # Display top prediction
                st.markdown(f"### Top prediction: **{predictions[0][0].title()}** shark")
                st.progress(predictions[0][1])
                st.write(f"Confidence: {predictions[0][1]*100:.1f}%")
                
                # Display other predictions
                st.markdown("### Other possibilities:")
                for label, confidence in predictions[1:]:
                    st.write(f"{label.title()}: {confidence*100:.1f}%")
                
                # Add labeled image download option
                labeled_img = label_image(image, f"{predictions[0][0]} ({predictions[0][1]*100:.1f}%)")
                buf = io.BytesIO()
                labeled_img.save(buf, format="PNG")
                st.download_button(
                    label="Download Labeled Image",
                    data=buf.getvalue(),
                    file_name=f"labeled_{uploaded_file.name}",
                    mime="image/png"
                )
    
    else:  # Multiple Images mode
        uploaded_files = st.file_uploader("Choose shark images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            # Process button
            if st.button("Process All Images"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zf:
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
                        
                        # Read image
                        image = Image.open(uploaded_file)
                        
                        # Get prediction
                        predictions = predict_image(image, model, device)
                        top_label, top_confidence = predictions[0]
                        
                        # Label image
                        labeled_img = label_image(image, f"{top_label} ({top_confidence*100:.1f}%)")
                        
                        # Save to zip
                        img_byte_arr = io.BytesIO()
                        labeled_img.save(img_byte_arr, format='PNG')
                        img_byte_arr.seek(0)
                        
                        zf.writestr(f"labeled_{uploaded_file.name}", img_byte_arr.getvalue())
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Offer zip download
                zip_buffer.seek(0)
                st.success("All images processed!")
                st.download_button(
                    label="Download Zip of Labeled Images",
                    data=zip_buffer.getvalue(),
                    file_name="labeled_sharks.zip",
                    mime="application/zip"
                )

if __name__ == "__main__":
    main()