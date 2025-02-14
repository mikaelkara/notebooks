# NLP Cloud

>[NLP Cloud](https://docs.nlpcloud.com/#introduction) is an artificial intelligence platform that allows you to use the most advanced AI engines, and even train your own engines with your own data. 

The [embeddings](https://docs.nlpcloud.com/#embeddings) endpoint offers the following model:

* `paraphrase-multilingual-mpnet-base-v2`: Paraphrase Multilingual MPNet Base V2 is a very fast model based on Sentence Transformers that is perfectly suited for embeddings extraction in more than 50 languages (see the full list here).


```python
%pip install --upgrade --quiet  nlpcloud
```


```python
from langchain_community.embeddings import NLPCloudEmbeddings
```


```python
import os

os.environ["NLPCLOUD_API_KEY"] = "xxx"
nlpcloud_embd = NLPCloudEmbeddings()
```


```python
text = "This is a test document."
```


```python
query_result = nlpcloud_embd.embed_query(text)
```


```python
doc_result = nlpcloud_embd.embed_documents([text])
```
