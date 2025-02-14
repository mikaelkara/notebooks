# Jina

You can check the list of available models from [here](https://jina.ai/embeddings/).

## Installation and setup

Install requirements


```python
pip install -U langchain-community
```

Import libraries


```python
import requests
from langchain_community.embeddings import JinaEmbeddings
from numpy import dot
from numpy.linalg import norm
from PIL import Image
```

## Embed text and queries with Jina embedding models through JinaAI API


```python
text_embeddings = JinaEmbeddings(
    jina_api_key="jina_*", model_name="jina-embeddings-v2-base-en"
)
```


```python
text = "This is a test document."
```


```python
query_result = text_embeddings.embed_query(text)
```


```python
print(query_result)
```


```python
doc_result = text_embeddings.embed_documents([text])
```


```python
print(doc_result)
```

## Embed images and queries with Jina CLIP through JinaAI API


```python
multimodal_embeddings = JinaEmbeddings(jina_api_key="jina_*", model_name="jina-clip-v1")
```


```python
image = "https://avatars.githubusercontent.com/u/126733545?v=4"

description = "Logo of a parrot and a chain on green background"

im = Image.open(requests.get(image, stream=True).raw)
print("Image:")
display(im)
```


```python
image_result = multimodal_embeddings.embed_images([image])
```


```python
print(image_result)
```


```python
description_result = multimodal_embeddings.embed_documents([description])
```


```python
print(description_result)
```


```python
cosine_similarity = dot(image_result[0], description_result[0]) / (
    norm(image_result[0]) * norm(description_result[0])
)
```


```python
print(cosine_similarity)
```
