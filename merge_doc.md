# Merge Documents Loader

Merge the documents returned from a set of specified data loaders.


```python
from langchain_community.document_loaders import WebBaseLoader

loader_web = WebBaseLoader(
    "https://github.com/basecamp/handbook/blob/master/37signals-is-you.md"
)
```


```python
from langchain_community.document_loaders import PyPDFLoader

loader_pdf = PyPDFLoader("../MachineLearning-Lecture01.pdf")
```


```python
from langchain_community.document_loaders.merge import MergedDataLoader

loader_all = MergedDataLoader(loaders=[loader_web, loader_pdf])
```


```python
docs_all = loader_all.load()
```


```python
len(docs_all)
```




    23


