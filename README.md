# clip-fine-tuning
Fine-tuning Open AI's Clip for image encoding using Flicker Data, see Arxiv [Paper](https://arxiv.org/abs/2103.00020). This was made translating english captions to spanish using a transformer from the University of Helsinki available in [HuggingFace](https://huggingface.co/Helsinki-NLP/opus-mt-en-es).


![image.png](https://miro.medium.com/max/786/1*tg7akErlMSyCLQxrMtQIYw.png) (Image from OpenAI)


This training script for image-text representation but can be extended to any non-english language captioned images.


You will need a Kaggle Token in order to download data.


```bash

chmod +x download_data.sh
./download_data

```

### Translation from english to spanish

Now in order to get transcriptions, run the script with ` python3 translate_english_captions.py` this will append a column to the Flicker dataset with spanish translations from the original descriptions.

### Fine-tuning

Run the following to fine-tune for 30 epochs using 20% of the data and a 70-30% train-test partition.


```python
## see python3 fine_tune_clip --help for more details on the arguments
python3 fine_tune_clip -p 0.2 -e 30 
```

A step-by-step guide is also available in `clip-fine-tuning.ipynb`


## Demo

A small notebook for inference was created using Flickr30K dataset, see `inference.ipynb`

### Streamlit App

There is a small streamlit app available in `/app`. Run the following:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port=6060
```

