# Instruct Embeddings on Hugging Face

>[Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers) is a Python framework for state-of-the-art sentence, text and image embeddings.
>One of the instruct embedding models is used in the `HuggingFaceInstructEmbeddings` class.



```python
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
```


```python
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the query for retrieval: "
)
```

    load INSTRUCTOR_Transformer
    max_seq_length  512
    


```python
text = "This is a test document."
```


```python
query_result = embeddings.embed_query(text)
```


```python

```
