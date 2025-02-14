# MyScale

>[MyScale](https://docs.myscale.com/en/overview/) is a cloud-based database optimized for AI applications and solutions, built on the open-source [ClickHouse](https://github.com/ClickHouse/ClickHouse). 

This notebook shows how to use functionality related to the `MyScale` vector database.

## Setting up environments


```python
%pip install --upgrade --quiet  clickhouse-connect langchain-community
```

We want to use OpenAIEmbeddings so we have to get the OpenAI API Key.


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
if "OPENAI_API_BASE" not in os.environ:
    os.environ["OPENAI_API_BASE"] = getpass.getpass("OpenAI Base:")
if "MYSCALE_HOST" not in os.environ:
    os.environ["MYSCALE_HOST"] = getpass.getpass("MyScale Host:")
if "MYSCALE_PORT" not in os.environ:
    os.environ["MYSCALE_PORT"] = getpass.getpass("MyScale Port:")
if "MYSCALE_USERNAME" not in os.environ:
    os.environ["MYSCALE_USERNAME"] = getpass.getpass("MyScale Username:")
if "MYSCALE_PASSWORD" not in os.environ:
    os.environ["MYSCALE_PASSWORD"] = getpass.getpass("MyScale Password:")
```

There are two ways to set up parameters for myscale index.

1. Environment Variables

    Before you run the app, please set the environment variable with `export`:
    `export MYSCALE_HOST='<your-endpoints-url>' MYSCALE_PORT=<your-endpoints-port> MYSCALE_USERNAME=<your-username> MYSCALE_PASSWORD=<your-password> ...`

    You can easily find your account, password and other info on our SaaS. For details please refer to [this document](https://docs.myscale.com/en/cluster-management/)

    Every attributes under `MyScaleSettings` can be set with prefix `MYSCALE_` and is case insensitive.

2. Create `MyScaleSettings` object with parameters


    ```python
    from langchain_community.vectorstores import MyScale, MyScaleSettings
    config = MyScaleSetting(host="<your-backend-url>", port=8443, ...)
    index = MyScale(embedding_function, config)
    index.add_documents(...)
    ```


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MyScale
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
```


```python
for d in docs:
    d.metadata = {"some": "metadata"}
docsearch = MyScale.from_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
```

    Inserting data...: 100%|██████████| 42/42 [00:15<00:00,  2.66it/s]
    


```python
print(docs[0].page_content)
```

    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    

## Get connection info and data schema


```python
print(str(docsearch))
```

## Filtering

You can have direct access to myscale SQL where statement. You can write `WHERE` clause following standard SQL.

**NOTE**: Please be aware of SQL injection, this interface must not be directly called by end-user.

If you customized your `column_map` under your setting, you search with filter like this:


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MyScale

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

for i, d in enumerate(docs):
    d.metadata = {"doc_id": i}

docsearch = MyScale.from_documents(docs, embeddings)
```

    Inserting data...: 100%|██████████| 42/42 [00:15<00:00,  2.68it/s]
    

### Similarity search with score

The returned distance score is cosine distance. Therefore, a lower score is better.


```python
meta = docsearch.metadata_column
output = docsearch.similarity_search_with_relevance_scores(
    "What did the president say about Ketanji Brown Jackson?",
    k=4,
    where_str=f"{meta}.doc_id<10",
)
for d, dist in output:
    print(dist, d.metadata, d.page_content[:20] + "...")
```

    0.229655921459198 {'doc_id': 0} Madam Speaker, Madam...
    0.24506962299346924 {'doc_id': 8} And so many families...
    0.24786919355392456 {'doc_id': 1} Groups of citizens b...
    0.24875116348266602 {'doc_id': 6} And I’m taking robus...
    

## Deleting your data

You can either drop the table with `.drop()` method or partially delete your data with `.delete()` method.


```python
# use directly a `where_str` to delete
docsearch.delete(where_str=f"{docsearch.metadata_column}.doc_id < 5")
meta = docsearch.metadata_column
output = docsearch.similarity_search_with_relevance_scores(
    "What did the president say about Ketanji Brown Jackson?",
    k=4,
    where_str=f"{meta}.doc_id<10",
)
for d, dist in output:
    print(dist, d.metadata, d.page_content[:20] + "...")
```

    0.24506962299346924 {'doc_id': 8} And so many families...
    0.24875116348266602 {'doc_id': 6} And I’m taking robus...
    0.26027143001556396 {'doc_id': 7} We see the unity amo...
    0.26390212774276733 {'doc_id': 9} And unlike the $2 Tr...
    


```python
docsearch.drop()
```


```python

```
