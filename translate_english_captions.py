### English to Spanish Captions

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
import pandas as pd
from transformers import pipeline
from datasets import Dataset as hfd
import logging


logging.basicConfig(
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger=logging.getLogger()
logger.setLevel(logging.INFO)

DEVICE_PIPE=0 if torch.cuda.is_available() else -1
OUT_PATH="flicker_data/flickr30k_images/results.csv"


##read original labels
labels=pd.read_csv("flicker_data/flickr30k_images/results.csv",sep="|")
labels=labels[~labels.duplicated(subset="image_name",keep="first")]


##download translator
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es",device=DEVICE_PIPE)


def translate_to_spanish(instance):
    instance['translation']=pipe(instance['caption'])
    return instance


if __name__=="__main__":
    captions=[{'caption':i.strip()} for i in labels[' comment'].values.tolist()]
    captions = hfd.from_list(captions)
    logger.info("Created HF-Dataset with captions\n Starting Translation using batches of size 10")
    captions=captions.map(translate_to_spanish,batched=True,batch_size=10)
    captions_df=captions.to_pandas()
    translations=[i['translation_text'].strip() for i in captions_df.translation.values.tolist()]
    labels['translations']=translations
    logger.info(f"Finished translation, saving csv file in {OUT_PATH}")
    labels.to_csv(OUT_PATH,sep="|",index=False)
