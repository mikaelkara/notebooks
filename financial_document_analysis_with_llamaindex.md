# Financial Document Analysis with LlamaIndex

In this example notebook, we showcase how to perform financial analysis over [**10-K**](https://en.wikipedia.org/wiki/Form_10-K) documents with the [**LlamaIndex**](https://gpt-index.readthedocs.io/en/latest/) framework with just a few lines of code.

## Notebook Outline
* [Introduction](#Introduction)
* [Setup](#Setup)
* [Data Loading & Indexing](#Data-Loading-and-Indexing)
* [Simple QA](#Simple-QA)
* [Advanced QA - Compare and Contrast](#Advanced-QA---Compare-and-Contrast)


## Introduction

### LLamaIndex
[LlamaIndex](https://gpt-index.readthedocs.io/en/latest/) is a data framework for LLM applications.
You can get started with just a few lines of code and build a retrieval-augmented generation (RAG) system in minutes.
For more advanced users, LlamaIndex offers a rich toolkit for ingesting and indexing your data, modules for retrieval and re-ranking, and composable components for building custom query engines.

See [full documentation](https://gpt-index.readthedocs.io/en/latest/) for more details.

### Financial Analysis over 10-K documents
A key part of a financial analyst's job is to extract information and synthesize insight from long financial documents.
A great example is the 10-K form - an annual report required by the U.S. Securities and Exchange Commission (SEC), that gives a comprehensive summary of a company's financial performance.
These documents typically run hundred of pages in length, and contain domain-specific terminology that makes it challenging for a layperson to digest quickly.


We showcase how LlamaIndex can support a financial analyst in quickly extracting information and synthesize insights **across multiple documents** with very little coding. 

## Setup

To begin, we need to install the llama-index library


```python
!pip install llama-index pypdf
```

Now, we import all modules used in this tutorial


```python
from langchain import OpenAI

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index import set_global_service_context
from llama_index.response.pprint_utils import pprint_response
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
```

Before we start, we can configure the LLM provider and model that will power our RAG system.  
Here, we pick `gpt-3.5-turbo-instruct` from OpenAI.  


```python
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct", max_tokens=-1)
```

We construct a `ServiceContext` and set it as the global default, so all subsequent operations that depends on LLM calls will use the model we configured here.


```python
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context=service_context)
```

## Data Loading and Indexing

Now, we load and parse 2 PDFs (one for Uber 10-K in 2021 and another for Lyft 10-k in 2021).    
Under the hood, the PDFs are converted to plain text `Document` objects, separate by page.  

> Note: this operation might take a while to run, since each document is more than 100 pages.


```python
lyft_docs = SimpleDirectoryReader(input_files=["../data/10k/lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=["../data/10k/uber_2021.pdf"]).load_data()
```


```python
print(f'Loaded lyft 10-K with {len(lyft_docs)} pages')
print(f'Loaded Uber 10-K with {len(uber_docs)} pages')
```

    Loaded lyft 10-K with 238 pages
    Loaded Uber 10-K with 307 pages
    

Now, we can build an (in-memory) `VectorStoreIndex` over the documents that we've loaded.  

> Note: this operation might take a while to run, since it calls OpenAI API for computing vector embedding over document chunks.


```python
lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)
```

## Simple QA

Now we are ready to run some queries against our indices!  
To do so, we first configure a `QueryEngine`, which just captures a set of configurations for how we want to query the underlying index.

For a `VectorStoreIndex`, the most common configuration to adjust is `similarity_top_k` which controls how many document chunks (which we call `Node` objects) are retrieved to use as context for answering our question.


```python
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
```


```python
uber_engine = uber_index.as_query_engine(similarity_top_k=3)
```

Let's see some queries in action!


```python
response = await lyft_engine.aquery('What is the revenue of Lyft in 2021? Answer in millions with page reference')
```


```python
print(response)
```

    
    $3,208.3 million (page 63)
    


```python
response = await uber_engine.aquery('What is the revenue of Uber in 2021? Answer in millions, with page reference')
```


```python
print(response)
```

    
    $17,455 (page 53)
    

## Advanced QA - Compare and Contrast

For more complex financial analysis, one often needs to reference multiple documents.  

As a example, let's take a look at how to do compare-and-contrast queries over both Lyft and Uber financials.  
For this, we build a `SubQuestionQueryEngine`, which breaks down a complex compare-and-contrast query, into simpler sub-questions to execute on respective sub query engine backed by individual indices.


```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine, 
        metadata=ToolMetadata(name='lyft_10k', description='Provides information about Lyft financials for year 2021')
    ),
    QueryEngineTool(
        query_engine=uber_engine, 
        metadata=ToolMetadata(name='uber_10k', description='Provides information about Uber financials for year 2021')
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
```

Let's see these queries in action!


```python
response = await s_engine.aquery('Compare and contrast the customer segments and geographies that grew the fastest')
```

    Generated 4 sub questions.
    [36;1m[1;3m[uber_10k] Q: What customer segments grew the fastest for Uber
    [0m[36;1m[1;3m[uber_10k] A: in 2021?
    
    The customer segments that grew the fastest for Uber in 2021 were its Mobility Drivers, Couriers, Riders, and Eaters. These segments experienced growth due to the continued stay-at-home order demand related to COVID-19, as well as Uber's introduction of its Uber One, Uber Pass, Eats Pass, and Rides Pass membership programs. Additionally, Uber's marketplace-centric advertising helped to connect merchants and brands with its platform network, further driving growth.
    [0m[33;1m[1;3m[uber_10k] Q: What geographies grew the fastest for Uber
    [0m[33;1m[1;3m[uber_10k] A: 
    Based on the context information, it appears that Uber experienced the most growth in large metropolitan areas, such as Chicago, Miami, New York City, Sao Paulo, and London. Additionally, Uber experienced growth in suburban and rural areas, as well as in countries such as Argentina, Germany, Italy, Japan, South Korea, and Spain.
    [0m[38;5;200m[1;3m[lyft_10k] Q: What customer segments grew the fastest for Lyft
    [0m[38;5;200m[1;3m[lyft_10k] A: 
    The customer segments that grew the fastest for Lyft were ridesharing, light vehicles, and public transit. Ridesharing grew as Lyft was able to predict demand and proactively incentivize drivers to be available for rides in the right place at the right time. Light vehicles grew as users were looking for options that were more active, usually lower-priced, and often more efficient for short trips during heavy traffic. Public transit grew as Lyft integrated third-party public transit data into the Lyft App to offer users a robust view of transportation options around them.
    [0m[32;1m[1;3m[lyft_10k] Q: What geographies grew the fastest for Lyft
    [0m[32;1m[1;3m[lyft_10k] A: 
    It is not possible to answer this question with the given context information.
    [0m


```python
print(response)
```

    
    The customer segments that grew the fastest for Uber in 2021 were its Mobility Drivers, Couriers, Riders, and Eaters. These segments experienced growth due to the continued stay-at-home order demand related to COVID-19, as well as Uber's introduction of its Uber One, Uber Pass, Eats Pass, and Rides Pass membership programs. Additionally, Uber's marketplace-centric advertising helped to connect merchants and brands with its platform network, further driving growth. Uber experienced the most growth in large metropolitan areas, such as Chicago, Miami, New York City, Sao Paulo, and London. Additionally, Uber experienced growth in suburban and rural areas, as well as in countries such as Argentina, Germany, Italy, Japan, South Korea, and Spain.
    
    The customer segments that grew the fastest for Lyft were ridesharing, light vehicles, and public transit. Ridesharing grew as Lyft was able to predict demand and proactively incentivize drivers to be available for rides in the right place at the right time. Light vehicles grew as users were looking for options that were more active, usually lower-priced, and often more efficient for short trips during heavy traffic. Public transit grew as Lyft integrated third-party public transit data into the Lyft App to offer users a robust view of transportation options around them. It is not possible to answer the question of which geographies grew the fastest for Lyft with the given context information.
    
    In summary, Uber and Lyft both experienced growth in customer segments related to mobility, couriers, riders, and eaters. Uber experienced the most growth in large metropolitan areas, as well as in suburban and rural areas, and in countries such as Argentina, Germany, Italy, Japan, South Korea, and Spain. Lyft experienced the most growth in ridesharing, light vehicles, and public transit. It is not possible to answer the question of which geographies grew the fastest for Lyft with the given context information.
    


```python
response = await s_engine.aquery('Compare revenue growth of Uber and Lyft from 2020 to 2021')
```

    Generated 2 sub questions.
    [36;1m[1;3m[uber_10k] Q: What is the revenue growth of Uber from 2020 to 2021
    [0m[36;1m[1;3m[uber_10k] A: 
    The revenue growth of Uber from 2020 to 2021 was 57%, or 54% on a constant currency basis.
    [0m[33;1m[1;3m[lyft_10k] Q: What is the revenue growth of Lyft from 2020 to 2021
    [0m[33;1m[1;3m[lyft_10k] A: 
    The revenue growth of Lyft from 2020 to 2021 is 36%, increasing from $2,364,681 thousand to $3,208,323 thousand.
    [0m


```python
print(response)
```

    
    The revenue growth of Uber from 2020 to 2021 was 57%, or 54% on a constant currency basis, while the revenue growth of Lyft from 2020 to 2021 was 36%. This means that Uber had a higher revenue growth than Lyft from 2020 to 2021.
    
