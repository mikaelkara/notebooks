# Apify Dataset

>[Apify Dataset](https://docs.apify.com/platform/storage/dataset) is a scalable append-only storage with sequential access built for storing structured web scraping results, such as a list of products or Google SERPs, and then export them to various formats like JSON, CSV, or Excel. Datasets are mainly used to save results of [Apify Actors](https://apify.com/store)—serverless cloud programs for various web scraping, crawling, and data extraction use cases.

This notebook shows how to load Apify datasets to LangChain.


## Prerequisites

You need to have an existing dataset on the Apify platform. This example shows how to load a dataset produced by the [Website Content Crawler](https://apify.com/apify/website-content-crawler).


```python
%pip install --upgrade --quiet  apify-client
```

First, import `ApifyDatasetLoader` into your source code:


```python
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.documents import Document
```

Then provide a function that maps Apify dataset record fields to LangChain `Document` format.

For example, if your dataset items are structured like this:

```json
{
    "url": "https://apify.com",
    "text": "Apify is the best web scraping and automation platform."
}
```

The mapping function in the code below will convert them to LangChain `Document` format, so that you can use them further with any LLM model (e.g. for question answering).


```python
loader = ApifyDatasetLoader(
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda dataset_item: Document(
        page_content=dataset_item["text"], metadata={"source": dataset_item["url"]}
    ),
)
```


```python
data = loader.load()
```

## An example with question answering

In this example, we use data from a dataset to answer a question.


```python
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.utilities import ApifyWrapper
from langchain_core.documents import Document
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
```


```python
loader = ApifyDatasetLoader(
    dataset_id="your-dataset-id",
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)
```


```python
index = VectorstoreIndexCreator(embedding=OpenAIEmbeddings()).from_loaders([loader])
```


```python
query = "What is Apify?"
result = index.query_with_sources(query, llm=OpenAI())
```


```python
print(result["answer"])
print(result["sources"])
```

     Apify is a platform for developing, running, and sharing serverless cloud programs. It enables users to create web scraping and automation tools and publish them on the Apify platform.
    
    https://docs.apify.com/platform/actors, https://docs.apify.com/platform/actors/running/actors-in-store, https://docs.apify.com/platform/security, https://docs.apify.com/platform/actors/examples
    
