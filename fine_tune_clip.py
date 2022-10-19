import numpy as np
import clip
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import argparse
import shutil
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import logging
import glob


logging.basicConfig(
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)

DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"
BASE_PATH="flicker_data/flickr30k_images/flickr30k_images"

#preprocess is just a sequential module
model, preprocess = clip.load("ViT-B/32") ## it can be modified to any model available in CLIP
model.to(DEVICE)

loss_img = nn.CrossEntropyLoss()
loss_caption = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

labels=pd.read_csv("flicker_data/flickr30k_images/results.csv",sep="|")
# keep the first caption per image and use only 20% of the data for training

## to avoid problems with mixed precision, taken from here https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float()


class FlickerDataset(Dataset):
    def __init__(self,captions,image_tensors):
        self.captions=captions
        self.images=image_tensors
        
    def __getitem__(self,idx):
        image_idx,_=self.images[idx]
        caption_idx=self.captions[idx]
        return {'image':image_idx,'caption':caption_idx}
        
    def __len__(self):
        return len(self.captions)


def train_model(n_epochs,train_dataloader,test_dataloader,checkpoint_path:str="./checkpoints"):
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    print(f"Using {DEVICE} for training")
    best_score=9e10
    for epoch in tqdm(range(n_epochs)):
        total_steps=0
        train_loss=0.0
        model.train()
        for step,data in enumerate(tqdm(train_dataloader),1):
            
            optimizer.zero_grad()
            
            img_batch=data['image'].to(DEVICE)
            captions_batch=clip.tokenize(data['caption'],truncate=True).to(DEVICE)
            with torch.cuda.amp.autocast():
                logits_image, logits_caption = model(img_batch, captions_batch)
            labels = torch.arange(len(img_batch)).to(DEVICE)  ## we are interested on predicting the right caption which is the caption position of every image
            img_loss=loss_img(logits_image,labels)
            caption_loss=loss_caption(logits_caption,labels)
            total_loss = (img_loss+caption_loss)/2
            total_loss.backward()
            train_loss+=total_loss.item()
            convert_models_to_fp32(model)
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_steps+=1
            
            if step%100==0:
                logger.info(f"Epoch {epoch} | step loss:{train_loss/total_steps:.4f}")
                wandb.log({'train_loss':train_loss/total_steps})
        
        val_metrics=validate(test_dataloader)
        wandb.log({'test_loss':val_metrics})
        logger.info(f"Epoch {epoch} end -> Train Loss: {train_loss/len(train_dataloader)} | Validation Loss: {val_metrics:.4f}")

        if val_metrics<best_score:
            print("Better score reached, saving checkpoint...")
            best_score=val_metrics
            if os.path.exists(Path(checkpoint_path)/"best_model.pt"):
                os.remove(Path(checkpoint_path)/"best_model.pt")
            torch.save(model.state_dict(), Path(checkpoint_path)/"best_model.pt")
            logger.info("Saving checkpoint to WANDB")
            wandb.save(checkpoint_path+"/best_model.pt")
            
        
        scheduler.step()
            
            

        
def validate(test_dl):
    model.eval()
    test_loss=0.0
    for data in tqdm(test_dl,desc="Evaluating in test"):
        img_batch=data['image'].to(DEVICE)
        captions_batch=clip.tokenize(data['caption'],truncate=True).to(DEVICE)
        with torch.no_grad():
            logits_image, logits_caption = model(img_batch, captions_batch)
        labels = torch.arange(len(img_batch)).to(DEVICE)  ## we are interested on predicting the right caption which is the caption position of every image
        total_loss = (loss_img(logits_image,labels) + loss_caption(logits_caption,labels))/2
        test_loss+=total_loss.item()
    
    test_total_loss=test_loss/len(test_dl)
    return test_total_loss
    
   



if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="Clip Trainer")
    parser.add_argument("-p", "--portion",type=float,default=0.2, help="fraction of the full data to use for train-test split", dest="prop_data")
    parser.add_argument("-tr", "--train_prop",type=float,default=0.7,help="train proportion of data", dest="train_prop")
    parser.add_argument("-e", "--epochs",type=int,default=5, help="Number of Epochs", dest="n_epochs")
    parser.add_argument("-bs", "--batch_size",type=int,default=45, help="Batch Size", dest="batch_size")
    parser.add_argument("-n", "--runname",type=str,default="clip-ft", help="Run name for training-wandb runname", dest="run_name")
    args = parser.parse_args()
    labels=labels[~labels.duplicated(subset="image_name",keep="first")].sample(frac=args.prop_data)
    labels['captions']=labels['translations'].map(lambda d: d.strip())
    train,test=train_test_split(labels,train_size=args.train_prop)
    ##make sure when using ImageFolder, idx positions match, ImageFolder will load files in ascending order by filename
    train.sort_values(by="image_name",ascending=True,inplace=True)
    test.sort_values(by="image_name",ascending=True,inplace=True)
    train.to_csv("train_labels.csv",sep=";",index=False)
    test.to_csv("test_labels.csv",sep=";",index=False)
    
    
    logger.info("1- Labels read...")
    
    if not os.path.isdir("flicker_data/train/images"):
        os.makedirs("flicker_data/train/images")
    else:
        logger.info("removing previous files in train/")
        for zippath in glob.iglob(os.path.join("flicker_data/train/images", '*.jpg')):
            os.remove(zippath)

    if not os.path.isdir("flicker_data/test/images"):
        os.makedirs("flicker_data/test/images")
    else:
        logger.info("removing previous files in test/")
        for zippath in glob.iglob(os.path.join("flicker_data/test/images", '*.jpg')):
            os.remove(zippath)

    for image in tqdm(train.image_name,desc="Copying train images"):
        shutil.copyfile(Path(BASE_PATH)/image,Path("flicker_data/train/images")/image)

    for image in tqdm(test.image_name,desc="Copying test images"):
        shutil.copyfile(Path(BASE_PATH)/image,Path("flicker_data/test/images")/image)

    ##read images and preprocess    
    train_images = datasets.ImageFolder("./flicker_data/train/",transform=preprocess)
    test_images = datasets.ImageFolder("./flicker_data/test/",transform=preprocess)
    
    logger.info("2- Reading Images... Done")
    
        
    train_dataset = FlickerDataset(train.captions.values.tolist(), train_images)
    test_dataset = FlickerDataset(test.captions.values.tolist(), test_images)
    logger.info(f"Train shape {len(train_dataset)}, Test Shape {len(test_dataset)}")
    
    
    ## create dataloaders
    tr_dl=DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size)
    ts_dl=DataLoader(test_dataset,shuffle=True,batch_size=args.batch_size)
    
    
    logger.info("3- DataLoaders... Done")
    
    ## before training
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    ##original paper
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(tr_dl)*args.n_epochs)
    torch.cuda.empty_cache()
    
    
    run=wandb.init(project="clip-fine-tuning",name=f"{args.run_name}_run-exp",config={"epochs": args.n_epochs,"batch_size":args.batch_size})
    
    logger.info("4. Starting Training...")
    train_model(args.n_epochs,tr_dl,ts_dl)
    
    run.finish()


