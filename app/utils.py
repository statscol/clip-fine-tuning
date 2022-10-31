import numpy as np
import clip
from tqdm import tqdm
import faiss
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torchvision.transforms as vision
from torchvision.transforms import Resize,ToTensor


##LOAD MODEL & IMAGES
BEST_MODEL_PATH="../best_model.pt"
model, preprocess = clip.load("RN50x4")
model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=torch.device('cpu')))


def preprocess_embeddings(embeddings: np.ndarray):
    embeddings = embeddings.astype("float32")
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    faiss.normalize_L2(embeddings)
    return embeddings

process_inference=vision.Compose([
        Resize((224,224),interpolation=vision.InterpolationMode.BICUBIC),
        ToTensor()
        ])

def search_images(text,faiss_index,k=10):
    with torch.no_grad():
        text_emb=model.encode_text(clip.tokenize(text,truncate=True)).detach().numpy()
    text_emb=preprocess_embeddings(text_emb)
    faiss_cos_sim, k_nearest_indexes = faiss_index.search(text_emb, k=k)
    return faiss_cos_sim,k_nearest_indexes