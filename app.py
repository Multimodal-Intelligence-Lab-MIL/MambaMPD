"""Streamlit demo app for MambaMPD marine-pollution segmentation.

Upload a SAR image and run the trained MambaMPD model to obtain a colourised
segmentation mask of the five M4D classes.

Run with:
    streamlit run app.py
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from PIL import Image

from models import build_mambampd
from utils.image_preprocessing import ImagePadder
from utils.logger import load_dict_from_json

CLASS_COLORS = ["#000000", "#00FFFF", "#FF0000", "#994C00", "#009900"]
CLASS_NAMES = ["sea_surface", "oil_spill", "oil_spill_look_alike", "ship", "land"]
LABEL_TO_COLOR = {
    0: np.array([0, 0, 0]),
    1: np.array([0, 255, 255]),
    2: np.array([255, 0, 0]),
    3: np.array([153, 76, 0]),
    4: np.array([0, 153, 0]),
}


def run_inference(image_array, file_weights, num_classes=5, file_stats_json="utils/image_stats.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mambampd(num_classes=num_classes, deep_supervision=False)
    model.to(device)
    model.load_state_dict(torch.load(file_weights, map_location=device))
    model.eval()

    try:
        dict_stats = load_dict_from_json(file_stats_json)
    except OSError:
        dir_json = os.path.dirname(os.path.realpath(__file__))
        dict_stats = load_dict_from_json(os.path.join(dir_json, file_stats_json))

    try:
        image_padder = ImagePadder("/data/images")
    except OSError:
        image_padder = ImagePadder("./sample_padding_image_for_inference")

    image_padded = image_padder.pad_image(image_array)
    image_pre = (image_padded / 255.0 - dict_stats["mean"]) / dict_stats["std"]
    image_pre = np.transpose(np.expand_dims(image_pre, axis=0), (0, 3, 1, 2))

    image_tensor = torch.tensor(image_pre).float().to(device)
    with torch.no_grad():
        pred_logits = model(image_tensor)
        pred_label = torch.argmax(F.softmax(pred_logits, dim=1), dim=1)

    pred_arr = np.squeeze(pred_label.detach().cpu().numpy()).astype(np.uint8)
    one_hot = np.eye(num_classes)[pred_arr]
    mask = np.zeros((pred_arr.shape[0], pred_arr.shape[1], 3))
    for sem_class in range(num_classes):
        layer = one_hot[:, :, sem_class].reshape(*pred_arr.shape, 1)
        mask += layer * LABEL_TO_COLOR[sem_class].reshape(1, 3)
    mask = mask.astype(np.uint8)

    h, w = pred_arr.shape
    return mask[11 : h - 11, 15 : w - 15]


def show_mask_interpretation():
    my_cmap = ListedColormap(CLASS_COLORS, name="my_cmap")
    fig = plt.figure(figsize=(20, 2))
    plt.title("Marine pollution mask interpretation")
    plt.xticks(ticks=np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    plt.yticks([])
    plt.imshow([[0, 1, 2, 3, 4]], cmap=my_cmap)
    st.pyplot(fig)


def infer():
    st.title("MambaMPD marine pollution detection")
    file_weights = st.sidebar.text_input("Model weights (.pt)", "mambampd/mambampd_best.pt")
    if not os.path.isfile(file_weights):
        st.write("Weights file not found.")

    image_file = st.sidebar.file_uploader("Select input SAR image", type=["jpg", "jpeg"])
    image_array = None
    if image_file is not None:
        image_array = np.array(Image.open(image_file))
        st.image(image_array, caption=f"Input image: {image_file.name}")
    else:
        st.write("Input image: not selected")

    if st.sidebar.button("Run inference") and image_array is not None:
        mask = run_inference(image_array, file_weights)
        st.image(mask, caption="Predicted segmentation mask")
        out = Image.fromarray(mask, "RGB")
        with io.BytesIO() as buf:
            out.save(buf, format="PNG")
            st.download_button("Download predicted mask", data=buf.getvalue(), file_name="pred_mask.png", mime="image/png")
        show_mask_interpretation()


def app_info():
    st.title("MambaMPD")
    st.markdown("_A Mamba-Driven Segmentation Framework for Marine Pollution Detection from Remote Sensing Imagery._")
    st.write("Upload a SAR image and run a trained MambaMPD model to segment marine pollution into 5 M4D classes.")


APP_MODES = {"App Info": app_info, "Inference": infer}


def main():
    mode = st.sidebar.selectbox("Select mode", list(APP_MODES.keys()))
    APP_MODES[mode]()


if __name__ == "__main__":
    main()
