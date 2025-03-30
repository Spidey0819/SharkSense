import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import os
import zipfile
import random

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
    
    # Draw a semi-transparent background for text
    text_width, text_height = 200, 20  # Approximate size
    draw.rectangle(
        [position[0]-5, position[1]-5, position[0]+text_width+5, position[1]+text_height+5],
        fill=(0, 0, 0, 128)
    )
    
    # Draw text
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return image_rgba.convert("RGB")

def predict_image(image):
    """Make a demo prediction for an image"""
    # Return mock prediction in demo mode
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

def main():
    st.title("🦈 Shark Species Classifier")
    
    st.info("""
    **Demo Mode**
    
    This is a demonstration version of the Shark Classifier. 
    In this mode, the app generates random predictions for illustration purposes.
    """)
    
    st.write("Upload shark images and get species predictions!")
    
    # Sidebar for mode selection
    mode = st.sidebar.radio(
        "Choose mode:",
        ("Single Image", "Multiple Images")
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## About")
    st.sidebar.info(
        "This app demonstrates a shark classifier that would normally use "
        "a ResNet50 model trained to identify 14 different species of sharks from images."
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
                    predictions = predict_image(image)
                
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
                        predictions = predict_image(image)
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
                
    # Add information about how to implement the full model
    st.markdown("---")
    st.markdown("""
    ### Implementing the Full Model
    
    To implement the actual shark classification model:
    
    1. You'll need to install PyTorch 2.2.0+ compatible with your Python version
    2. Ensure the model file (`model_phase2_full.pth`) is accessible
    3. Add the necessary imports and model loading code
    
    You can use a platform like Hugging Face Spaces, Google Colab, or a custom server for better compatibility with PyTorch requirements.
    """)

if __name__ == "__main__":
    main()