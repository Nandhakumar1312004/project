# app.py

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import random

# --- PLACEHOLDER DUMMY MODELS ---
class DummyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 3 * 64 * 64)

    def forward(self, z, text_embedding):
        combined_input = torch.cat([z, text_embedding], dim=1) if text_embedding is not None else z
        out = self.linear(combined_input)
        return out.view(-1, 3, 64, 64).tanh()

class DummyTextEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(1000, embedding_dim)

    def forward(self, text):
        return torch.randn(text.shape[0], 128)

@st.cache_resource
def load_models():
    generator = DummyGenerator()
    text_encoder = DummyTextEncoder()
    generator.eval()
    text_encoder.eval()
    return generator, text_encoder

try:
    generator, text_encoder = load_models()
except Exception as e:
    st.error(f"Error loading placeholder models: {e}")
    generator = None
    text_encoder = None
# --- END OF PLACEHOLDER ---

def generate_image(text_prompt):
    if generator is None or text_encoder is None:
        st.warning("The picture-making machines haven't been loaded yet.")
        return None

    with torch.no_grad():
        dummy_text_input = torch.randint(0, 1000, (1, 10))
        text_embedding = text_encoder(dummy_text_input)
        noise = torch.randn(1, 100)
        generated_tensor = generator(noise, text_embedding)
        generated_tensor = (generated_tensor + 1) / 2.0
        img = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
        return img

st.title("Image generator!")
st.subheader("Type some words and see what picture I create.")

text_prompt = st.text_input("Enter your words here:", "A colorful flower")

if st.button("Make the Picture!"):
    if text_prompt:
        with st.spinner("Thinking hard and making the picture..."):
            try:
                generated_image = generate_image(text_prompt)
                if generated_image:
                    st.image(generated_image, caption=f"Picture for: '{text_prompt}'")
            except Exception as e:
                st.error(f"Oh no! Something went wrong during generation: {e}")
    else:
        st.warning("Please type some words before asking me to make a picture!")

st.markdown("---")
st.markdown("This is a super cool picture-making machine!")
