# Concurrent Loader

Works just like the GenericLoader but concurrently for those who choose to optimize their workflow.



```python
from langchain_community.document_loaders import ConcurrentLoader
```


```python
loader = ConcurrentLoader.from_filesystem("example_data/", glob="**/*.txt")
```


```python
files = loader.load()
```


```python
len(files)
```




    2




```python

```
