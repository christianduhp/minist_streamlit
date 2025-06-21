import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import CVAE

device = torch.device("cpu")

# Load model
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location=device))
model.eval()

st.title("Digit Generator (0-9) - CVAE")

digit = st.number_input(
    "Choose a digit (0-9):", min_value=0, max_value=9, value=0, step=1
)

if st.button("Generate 5 Images"):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(5):
        z = torch.randn(1, 20)
        y = torch.tensor([digit])
        gen = model.decoder(z, y).detach().numpy().squeeze()
        axs[i].imshow(gen, cmap="gray")
        axs[i].axis("off")

    st.pyplot(fig)


st.markdown(
    """
    <hr style="margin-top: 40px;">
    <div style="text-align: center; color: gray;">
        Made by Christian Oliveira - AI Contest
    </div>
    """,
    unsafe_allow_html=True,
)
