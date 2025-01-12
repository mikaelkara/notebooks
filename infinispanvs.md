# Infinispan

Infinispan is an open-source key-value data grid, it can work as single node as well as distributed.

Vector search is supported since release 15.x
For more: [Infinispan Home](https://infinispan.org)


```python
# Ensure that all we need is installed
# You may want to skip this
%pip install sentence-transformers
%pip install langchain
%pip install langchain_core
%pip install langchain_community
```

# Setup

To run this demo we need a running Infinispan instance without authentication and a data file.
In the next three cells we're going to:
- download the data file
- create the configuration
- run Infinispan in docker


```bash
%%bash
#get an archive of news
wget https://raw.githubusercontent.com/rigazilla/infinispan-vector/main/bbc_news.csv.gz
```


```bash
%%bash
#create infinispan configuration file
echo 'infinispan:
  cache-container: 
    name: default
    transport: 
      cluster: cluster 
      stack: tcp 
  server:
    interfaces:
      interface:
        name: public
        inet-address:
          value: 0.0.0.0 
    socket-bindings:
      default-interface: public
      port-offset: 0        
      socket-binding:
        name: default
        port: 11222
    endpoints:
      endpoint:
        socket-binding: default
        rest-connector:
' > infinispan-noauth.yaml
```


```python
!docker rm --force infinispanvs-demo
!docker run -d --name infinispanvs-demo -v $(pwd):/user-config  -p 11222:11222 infinispan/server:15.0 -c /user-config/infinispan-noauth.yaml
```

# The Code

## Pick up an embedding model

In this demo we're using
a HuggingFace embedding mode.


```python
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L12-v2"
hf = HuggingFaceEmbeddings(model_name=model_name)
```

## Setup Infinispan cache

Infinispan is a very flexible key-value store, it can store raw bits as well as complex data type.
User has complete freedom in the datagrid configuration, but for simple data type everything is automatically
configured by the python layer. We take advantage of this feature so we can focus on our application.

## Prepare the data

In this demo we rely on the default configuration, thus texts, metadatas and vectors in the same cache, but other options are possible: i.e. content can be store somewhere else and vector store could contain only a reference to the actual content.


```python
import csv
import gzip
import time

# Open the news file and process it as a csv
with gzip.open("bbc_news.csv.gz", "rt", newline="") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
    i = 0
    texts = []
    metas = []
    embeds = []
    for row in spamreader:
        # first and fifth values are joined to form the content
        # to be processed
        text = row[0] + "." + row[4]
        texts.append(text)
        # Store text and title as metadata
        meta = {"text": row[4], "title": row[0]}
        metas.append(meta)
        i = i + 1
        # Change this to change the number of news you want to load
        if i >= 5000:
            break
```

# Populate the vector store


```python
# add texts and fill vector db

from langchain_community.vectorstores import InfinispanVS

ispnvs = InfinispanVS.from_texts(texts, hf, metas)
```

# An helper func that prints the result documents

By default InfinispanVS returns the protobuf `ŧext` field in the `Document.page_content`
and all the remaining protobuf fields (except the vector) in the `metadata`. This behaviour is
configurable via lambda functions at setup.


```python
def print_docs(docs):
    for res, i in zip(docs, range(len(docs))):
        print("----" + str(i + 1) + "----")
        print("TITLE: " + res.metadata["title"])
        print(res.page_content)
```

# Try it!!!

Below some sample queries


```python
docs = ispnvs.similarity_search("European nations", 5)
print_docs(docs)
```


```python
print_docs(ispnvs.similarity_search("Milan fashion week begins", 2))
```


```python
print_docs(ispnvs.similarity_search("Stock market is rising today", 4))
```


```python
print_docs(ispnvs.similarity_search("Why cats are so viral?", 2))
```


```python
print_docs(ispnvs.similarity_search("How to stay young", 5))
```


```python
!docker rm --force infinispanvs-demo
```
