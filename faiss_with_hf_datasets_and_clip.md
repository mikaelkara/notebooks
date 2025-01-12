# Embedding multimodal data for similarity search using 🤗 transformers, 🤗 datasets and FAISS

_Authored by: [Merve Noyan](https://huggingface.co/merve)_

Embeddings are semantically meaningful compressions of information. They can be used to do similarity search, zero-shot classification or simply train a new model. Use cases for similarity search include searching for similar products in e-commerce, content search in social media and more.
This notebook walks you through using 🤗transformers, 🤗datasets and FAISS to create and index embeddings from a feature extraction model to later use them for similarity search.
Let's install necessary libraries.


```python
!pip install -q datasets faiss-gpu transformers sentencepiece
```

For this tutorial, we will use [CLIP model](https://huggingface.co/openai/clip-vit-base-patch16) to extract the features. CLIP is a revolutionary model that introduced joint training of a text encoder and an image encoder to connect two modalities.


```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import faiss
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
```

Load the dataset. To keep this notebook light, we will use a small captioning dataset, [jmhessel/newyorker_caption_contest](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest).


```python
from datasets import load_dataset

ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
```

See an example.


```python
ds["train"][0]["image"]
```




    
![png](output_7_0.png)
    




```python
ds["train"][0]["image_description"]
```




    'Two women are looking out a window. There is snow outside, and there is a snowman with human arms.'



We don't have to write any function to embed examples or create an index. 🤗 datasets library's FAISS integration abstracts these processes. We can simply use `map` method of the dataset to create a new column with the embeddings for each example like below. Let's create one for text features on the prompt column.


```python
dataset = ds["train"]
ds_with_embeddings = dataset.map(lambda example:
                                {'embeddings': model.get_text_features(
                                    **tokenizer([example["image_description"]],
                                                truncation=True, return_tensors="pt")
                                    .to("cuda"))[0].detach().cpu().numpy()})

```

We can do the same and get the image embeddings.


```python
ds_with_embeddings = ds_with_embeddings.map(lambda example:
                                          {'image_embeddings': model.get_image_features(
                                              **processor([example["image"]], return_tensors="pt")
                                              .to("cuda"))[0].detach().cpu().numpy()})

```

Now, we create an index for each column.


```python
# create FAISS index for text embeddings
ds_with_embeddings.add_faiss_index(column='embeddings')
```


```python
# create FAISS index for image embeddings
ds_with_embeddings.add_faiss_index(column='image_embeddings')
```

## Querying the data with text prompts

We can now query the dataset with text or image to get similar items from it.


```python
prmt = "a snowy day"
prmt_embedding = model.get_text_features(**tokenizer([prmt], return_tensors="pt", truncation=True).to("cuda"))[0].detach().cpu().numpy()
scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', prmt_embedding, k=1)
```


```python
def downscale_images(image):
  width = 200
  ratio = (width / float(image.size[0]))
  height = int((float(image.size[1]) * float(ratio)))
  img = image.resize((width, height), Image.Resampling.LANCZOS)
  return img

images = [downscale_images(image) for image in retrieved_examples["image"]]
# see the closest text and image
print(retrieved_examples["image_description"])
display(images[0])


```

    ['A man is in the snow. A boy with a huge snow shovel is there too. They are outside a house.']
    


    
![png](output_19_1.png)
    


## Querying the data with image prompts

Image similarity inference is similar, where you just call `get_image_features`.


```python
import requests
# image of a beaver
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png"
image = Image.open(requests.get(url, stream=True).raw)
display(downscale_images(image))
```


    
![png](output_22_0.png)
    


Search for the similar image.


```python
img_embedding = model.get_image_features(**processor([image], return_tensors="pt", truncation=True).to("cuda"))[0].detach().cpu().numpy()
scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('image_embeddings', img_embedding, k=1)
```

Display the most similar image to the beaver image.


```python
images = [downscale_images(image) for image in retrieved_examples["image"]]
# see the closest text and image
print(retrieved_examples["image_description"])
display(images[0])
```

    ['Salmon swim upstream but they see a grizzly bear and are in shock. The bear has a smug look on his face when he sees the salmon.']
    


    
![png](output_26_1.png)
    


## Saving, pushing and loading the embeddings
We can save the dataset with embeddings with `save_faiss_index`.



```python
ds_with_embeddings.save_faiss_index('embeddings', 'embeddings/embeddings.faiss')
```


```python
ds_with_embeddings.save_faiss_index('image_embeddings', 'embeddings/image_embeddings.faiss')
```

It's a good practice to store the embeddings in a dataset repository, so we will create one and push our embeddings there to pull later.
We will login to Hugging Face Hub, create a dataset repository there and push our indexes there and load using `snapshot_download`.


```python
from huggingface_hub import HfApi, notebook_login, snapshot_download
notebook_login()
```


```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("merve/faiss_embeddings", repo_type="dataset")
api.upload_folder(
    folder_path="./embeddings",
    repo_id="merve/faiss_embeddings",
    repo_type="dataset",
)
```


```python
snapshot_download(repo_id="merve/faiss_embeddings", repo_type="dataset",
                  local_dir="downloaded_embeddings")
```

  We can load the embeddings to the dataset with no embeddings using `load_faiss_index`.


```python
ds = ds["train"]
ds.load_faiss_index('embeddings', './downloaded_embeddings/embeddings.faiss')
# infer again
prmt = "people under the rain"

```


```python
prmt_embedding = model.get_text_features(
                        **tokenizer([prmt], return_tensors="pt", truncation=True)
                        .to("cuda"))[0].detach().cpu().numpy()

scores, retrieved_examples = ds.get_nearest_examples('embeddings', prmt_embedding, k=1)
```


```python
display(retrieved_examples["image"][0])
```


    
![png](output_37_0.png)
    

