import streamlit as st
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from streamlit_imagegrid import streamlit_imagegrid
from utils import preprocess_embeddings,preprocess,model,search_images
import faiss
import numpy as np
import clip
from PIL import Image

st.set_page_config(page_title="Text-to-image Similarity",layout='wide',page_icon=":shark:")

BASE_PATH="/home/jhon.parra/Documents/clip-fine-tuning/flicker_data/flickr30k_images/"
IMG_EMBEDDINGS = np.load('../embeddings.npy')
IMG_WIDTH,IMG_HEIGHT=300,350 ## for resizing images
images=datasets.ImageFolder(BASE_PATH,transform=preprocess)


##LOAD IMAGE EMBEDDINGS
emb=preprocess_embeddings(IMG_EMBEDDINGS)

## TRAIN FAISS INDEX 
faiss_index = faiss.IndexFlatIP(emb.shape[1])
faiss_index = faiss.IndexIDMap(faiss_index)
faiss_index.train(emb)
faiss_index.add_with_ids(emb,np.arange(len(images)))


with st.sidebar:

    st.markdown(
        f"<h1 style='text-align: center;'> Text to image using a fine-tuned version of Open AI's CLIP </h1>",
        unsafe_allow_html=True,
    )
    form = st.form(key='search_form')
    search_string=form.text_input(label='Busca imágenes', value="Gatos en el campo")
    _,col2,_=st.columns([1,1,1])
    with col2:
        search_trigger=form.form_submit_button(label='Search')
    st.markdown("***")


if search_trigger and search_string!="":
    st.markdown("<h1 style='text-align: center; color: white;'>Imágenes similares encontradas </h1>", unsafe_allow_html=True)
    ##image grid taken from https://discuss.streamlit.io/t/grid-of-images-with-the-same-height/10668/6
    idx = 0 
    sim,indexes=search_images(text=search_string,faiss_index=faiss_index,k=15)
    sim=sim.squeeze().tolist()
    indexes=indexes.squeeze().tolist()
    for _ in range(len(indexes)):
        cols = st.columns(5) 
        if idx < len(indexes):
            cols[0].image(Image.open(images.imgs[indexes[idx]][0]).resize((IMG_WIDTH,IMG_HEIGHT)), use_column_width=True, caption=f"Similarity: {sim[idx]:.2%}")        
            idx+=1
        if idx < len(indexes):
            cols[1].image(Image.open(images.imgs[indexes[idx]][0]).resize((IMG_WIDTH,IMG_HEIGHT)), use_column_width=True, caption=f"Similarity: {sim[idx]:.2%}")        
            idx+=1
        if idx < len(indexes):
            cols[2].image(Image.open(images.imgs[indexes[idx]][0]).resize((IMG_WIDTH,IMG_HEIGHT)), use_column_width=True, caption=f"Similarity: {sim[idx]:.2%}")        
            idx+=1
        if idx < len(indexes):
            cols[3].image(Image.open(images.imgs[indexes[idx]][0]).resize((IMG_WIDTH,IMG_HEIGHT)), use_column_width=True, caption=f"Similarity: {sim[idx]:.2%}")        
            idx+=1
        if idx < len(indexes):
            cols[4].image(Image.open(images.imgs[indexes[idx]][0]).resize((IMG_WIDTH,IMG_HEIGHT)), use_column_width=True, caption=f"Similarity: {sim[idx]:.2%}")        
            idx+=1
        else:
            break


