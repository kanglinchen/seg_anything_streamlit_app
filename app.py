import cv2
import sys
import json
import torch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
import gc
gc.collect()
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
warnings.filterwarnings('ignore')
from PIL import Image

st.set_page_config(
    page_title="SEG-ANYTHING-STREAMLIT",
    page_icon="🚀",
    layout= "wide",
    )

@st.cache_data()
def mask_generate(image, input_point, input_label):
    #sam_checkpoint = "model/sam_vit_l_0b3195.pth"
    sam_checkpoint = "model/sam_vit_b_01ec64.pth"
    #model_type = "vit_l"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    #sam.to(device=device, dtype=torch.half, non_blocking=True)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    return masks, scores, logits


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


st.title("✨ Segment Anything 🏜")
st.info(' Let me help generate segments for any of your images. 😉')
col1, col2 = st.columns(2)
with col1:
    st.header("Select A Tool 🛠")
with col2:
    image_path = st.file_uploader("Upload Image 🚀", type=["png","jpg","bmp","jpeg"])
    if image_path is not None:
        with st.spinner("Working.. 💫"):
            image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            if width > 640:
                new_width = 640
                new_height = int(height * (new_width / width))
                image = cv2.resize(image, (new_width, new_height))

            input_point = np.array([[300, 250]])
            input_label = np.array([1])
            masks, scores, logits = mask_generate(image, input_point, input_label)

            col3, col4 = st.columns(2)
            with col3:
                st.image(image)
                st.success("Original Image")
            with col4:
                max_score_index = np.argmax(scores)
                max_mask = masks[max_score_index]
                max_score = scores[max_score_index]
                fig, ax = plt.subplots(figsize=(20,20))
                ax.imshow(image)
                show_mask(max_mask, ax)
                show_points(input_point, input_label, ax)
                plt.title(f"Mask {max_score_index + 1}, Score: {max_score:.3f}", fontsize=18)
                ax.axis('on')
                st.pyplot(fig)
                st.success("Output Image")
    else:
        st.warning('⚠ Please upload your Image! 😯')
