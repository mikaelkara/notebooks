---
sidebar_label: LangSmith
---
# LangSmithLoader

This notebook provides a quick overview for getting started with the LangSmith [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all LangSmithLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html).

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [LangSmithLoader](https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html) | [langchain-core](https://python.langchain.com/api_reference/core/index.html) | ❌ | ❌ | ❌ | 

### Loader features
| Source | Lazy loading | Native async
| :---: | :---: | :---: | 
| LangSmithLoader | ✅ | ❌ | 

## Setup

To access the LangSmith document loader you'll need to install `langchain-core`, create a [LangSmith](https://langsmith.com) account and get an API key.

### Credentials

Sign up at https://langsmith.com and generate an API key. Once you've done this set the LANGSMITH_API_KEY environment variable:


```python
import getpass
import os

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
```

If you want to get automated best-in-class tracing, you can also turn on LangSmith tracing:


```python
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

Install `langchain-core`:


```python
%pip install -qU langchain-core
```

### Clone example dataset

For this example, we'll clone and load a public LangSmith dataset. Cloning creates a copy of this dataset on our personal LangSmith account. You can only load datasets that you have a personal copy of.


```python
from langsmith import Client as LangSmithClient

ls_client = LangSmithClient()

dataset_name = "LangSmith Few Shot Datasets Notebook"
dataset_public_url = (
    "https://smith.langchain.com/public/55658626-124a-4223-af45-07fb774a6212/d"
)

ls_client.clone_public_dataset(dataset_public_url)
```

## Initialization

Now we can instantiate our document loader and load documents:


```python
from langchain_core.document_loaders import LangSmithLoader

loader = LangSmithLoader(
    dataset_name=dataset_name,
    content_key="question",
    limit=50,
    # format_content=...,
    # ...
)
```

## Load


```python
docs = loader.load()
print(docs[0].page_content)
```

    Show me an example using Weaviate, but customizing the vectorStoreRetriever to return the top 10 k nearest neighbors. 
    


```python
print(docs[0].metadata["inputs"])
```

    {'question': 'Show me an example using Weaviate, but customizing the vectorStoreRetriever to return the top 10 k nearest neighbors. '}
    


```python
print(docs[0].metadata["outputs"])
```

    {'answer': 'To customize the Weaviate client and return the top 10 k nearest neighbors, you can utilize the `as_retriever` method with the appropriate parameters. Here\'s how you can achieve this:\n\n```python\n# Assuming you have imported the necessary modules and classes\n\n# Create the Weaviate client\nclient = weaviate.Client(url=os.environ["WEAVIATE_URL"], ...)\n\n# Initialize the Weaviate wrapper\nweaviate = Weaviate(client, index_name, text_key)\n\n# Customize the client to return top 10 k nearest neighbors using as_retriever\ncustom_retriever = weaviate.as_retriever(\n    search_type="similarity",\n    search_kwargs={\n        \'k\': 10  # Customize the value of k as needed\n    }\n)\n\n# Now you can use the custom_retriever to perform searches\nresults = custom_retriever.search(query, ...)\n```'}
    


```python
list(docs[0].metadata.keys())
```




    ['dataset_id',
     'inputs',
     'outputs',
     'metadata',
     'id',
     'created_at',
     'modified_at',
     'runs',
     'source_run_id']



## Lazy Load


```python
page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)
        # page = []
        break
len(page)
```




    10



## API reference

For detailed documentation of all LangSmithLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/core/document_loaders/langchain_core.document_loaders.langsmith.LangSmithLoader.html
