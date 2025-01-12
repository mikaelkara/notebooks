# ModelScope

>[ModelScope](https://www.modelscope.cn/home) is big repository of the models and datasets.

Let's load the ModelScope Embedding class.


```python
from langchain_community.embeddings import ModelScopeEmbeddings
```


```python
model_id = "damo/nlp_corom_sentence-embedding_english-base"
```


```python
embeddings = ModelScopeEmbeddings(model_id=model_id)
```


```python
text = "This is a test document."
```


```python
query_result = embeddings.embed_query(text)
```


```python
doc_results = embeddings.embed_documents(["foo"])
```
