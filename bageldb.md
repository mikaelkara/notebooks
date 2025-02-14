# BagelDB

> [BagelDB](https://www.bageldb.ai/) (`Open Vector Database for AI`), is like GitHub for AI data.
It is a collaborative platform where users can create,
share, and manage vector datasets. It can support private projects for independent developers,
internal collaborations for enterprises, and public contributions for data DAOs.

### Installation and Setup

```bash
pip install betabageldb langchain-community
```



## Create VectorStore from texts


```python
from langchain_community.vectorstores import Bagel

texts = ["hello bagel", "hello langchain", "I love salad", "my car", "a dog"]
# create cluster and add texts
cluster = Bagel.from_texts(cluster_name="testing", texts=texts)
```


```python
# similarity search
cluster.similarity_search("bagel", k=3)
```




    [Document(page_content='hello bagel', metadata={}),
     Document(page_content='my car', metadata={}),
     Document(page_content='I love salad', metadata={})]




```python
# the score is a distance metric, so lower is better
cluster.similarity_search_with_score("bagel", k=3)
```




    [(Document(page_content='hello bagel', metadata={}), 0.27392977476119995),
     (Document(page_content='my car', metadata={}), 1.4783176183700562),
     (Document(page_content='I love salad', metadata={}), 1.5342965126037598)]




```python
# delete the cluster
cluster.delete_cluster()
```

## Create VectorStore from docs


```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)[:10]
```


```python
# create cluster with docs
cluster = Bagel.from_documents(cluster_name="testing_with_docs", documents=docs)
```


```python
# similarity search
query = "What did the president say about Ketanji Brown Jackson"
docs = cluster.similarity_search(query)
print(docs[0].page_content[:102])
```

    Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the 
    

## Get all text/doc from Cluster


```python
texts = ["hello bagel", "this is langchain"]
cluster = Bagel.from_texts(cluster_name="testing", texts=texts)
cluster_data = cluster.get()
```


```python
# all keys
cluster_data.keys()
```




    dict_keys(['ids', 'embeddings', 'metadatas', 'documents'])




```python
# all values and keys
cluster_data
```




    {'ids': ['578c6d24-3763-11ee-a8ab-b7b7b34f99ba',
      '578c6d25-3763-11ee-a8ab-b7b7b34f99ba',
      'fb2fc7d8-3762-11ee-a8ab-b7b7b34f99ba',
      'fb2fc7d9-3762-11ee-a8ab-b7b7b34f99ba',
      '6b40881a-3762-11ee-a8ab-b7b7b34f99ba',
      '6b40881b-3762-11ee-a8ab-b7b7b34f99ba',
      '581e691e-3762-11ee-a8ab-b7b7b34f99ba',
      '581e691f-3762-11ee-a8ab-b7b7b34f99ba'],
     'embeddings': None,
     'metadatas': [{}, {}, {}, {}, {}, {}, {}, {}],
     'documents': ['hello bagel',
      'this is langchain',
      'hello bagel',
      'this is langchain',
      'hello bagel',
      'this is langchain',
      'hello bagel',
      'this is langchain']}




```python
cluster.delete_cluster()
```

## Create cluster with metadata & filter using metadata


```python
texts = ["hello bagel", "this is langchain"]
metadatas = [{"source": "notion"}, {"source": "google"}]

cluster = Bagel.from_texts(cluster_name="testing", texts=texts, metadatas=metadatas)
cluster.similarity_search_with_score("hello bagel", where={"source": "notion"})
```




    [(Document(page_content='hello bagel', metadata={'source': 'notion'}), 0.0)]




```python
# delete the cluster
cluster.delete_cluster()
```
