# clip-fine-tuning
Fine-tuning Open AI Clip for image encoding using Flicker Data. See Arxiv [Paper](https://arxiv.org/abs/2103.00020).

![image.png](https://miro.medium.com/max/786/1*tg7akErlMSyCLQxrMtQIYw.png)


This is an almost useless training script for image-text representation but can be extended to any non-english language captioned images.


You will need a Kaggle Token in order to download data.


```bash

chmod +x download_data.sh
./download_data

```

A small training script is available, run the following to fine-tune for 30 epochs using 20% of the data and a 70-30% train-test partition.


```python
## see python3 fine_tune_clip --help for more details on the arguments
python3 fine_tune_clip -e 30 
```


A step-by-step guide is also available in `training_model.ipynb`


## Demo

A small notebook for inference was created using Flickr30K dataset, see `inference.ipynb`

