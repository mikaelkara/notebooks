##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Question Answering LlamaIndex and Chroma

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/llamaindex/Gemini_LlamaIndex_QA_Chroma_WebPageReader.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

## Overview

[Gemini](https://ai.google.dev/models/gemini) is a family of generative AI models that lets developers generate content and solve problems. These models are designed and trained to handle both text and images as input.

[LlamaIndex](https://www.llamaindex.ai/) is a simple, flexible data framework that can be used by Large Language Model(LLM) applications to connect custom data sources to LLMs.

[Chroma](https://docs.trychroma.com/) is an open-source embedding database focused on simplicity and developer productivity. Chroma allows users to store embeddings and their metadata, embed documents and queries, and search the embeddings quickly.

In this notebook, you'll learn how to create an application that answers questions using data from a website with the help of Gemini, LlamaIndex, and Chroma.

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install LlamaIndex's Python library, `llama-index`. Install LlamaIndex's integration package for Gemini, `llama-index-llms-gemini` and the integration package for Gemini embedding model, `llama-index-embeddings-gemini`. Next, install LlamaIndex's web page reader, `llama-index-readers-web`. Finally, install ChromaDB's Python client SDK, `chromadb` and


```
# This guide was tested with 0.10.17, but feel free to try newer versions.
!pip install -q llama-index==0.10.17
!pip install -q llama-index-llms-gemini
!pip install -q llama-index-embeddings-gemini
!pip install -q llama-index-readers-web
!pip install -q llama-index-vector-stores-chroma
!pip install -q chromadb
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15.4/15.4 MB[0m [31m26.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m29.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.8/130.8 kB[0m [31m11.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m327.4/327.4 kB[0m [31m8.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m52.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m141.9/141.9 kB[0m [31m9.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m290.4/290.4 kB[0m [31m18.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m4.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.2/49.2 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m44.1 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    imageio 2.31.6 requires pillow<10.1.0,>=8.3.2, but you have pillow 10.3.0 which is incompatible.[0m[31m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m72.9/72.9 kB[0m [31m432.3 kB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.1/211.1 kB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m37.8/37.8 MB[0m [31m32.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.4/9.4 MB[0m [31m68.4 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.3/81.3 kB[0m [31m7.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m97.6/97.6 kB[0m [31m11.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.4/7.4 MB[0m [31m74.0 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
      Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m467.7/467.7 kB[0m [31m31.0 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
      Building wheel for tinysegmenter (setup.py) ... [?25l[?25hdone
      Building wheel for spider-client (setup.py) ... [?25l[?25hdone
      Building wheel for feedfinder2 (setup.py) ... [?25l[?25hdone
      Building wheel for jieba3k (setup.py) ... [?25l[?25hdone
      Building wheel for sgmllib3k (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m559.5/559.5 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m14.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m92.0/92.0 kB[0m [31m9.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.4/62.4 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.3/41.3 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.8/6.8 MB[0m [31m34.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m59.9/59.9 kB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m107.0/107.0 kB[0m [31m13.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.3/67.3 kB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m283.7/283.7 kB[0m [31m27.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m47.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.6/67.6 kB[0m [31m6.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m145.0/145.0 kB[0m [31m17.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m71.9/71.9 kB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.6/53.6 kB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.0/46.0 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m52.5/52.5 kB[0m [31m5.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.5/130.5 kB[0m [31m14.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m341.4/341.4 kB[0m [31m32.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.4/3.4 MB[0m [31m60.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m61.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.2/130.2 kB[0m [31m14.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m307.7/307.7 kB[0m [31m26.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m86.8/86.8 kB[0m [31m10.3 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for pypika (pyproject.toml) ... [?25l[?25hdone
    

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

## Basic steps
LLMs are trained offline on a large corpus of public data. Hence they cannot answer questions based on custom or private data accurately without additional context.

If you want to make use of LLMs to answer questions based on private data, you have to provide the relevant documents as context alongside your prompt. This approach is called Retrieval Augmented Generation (RAG).

You will use this approach to create a question-answering assistant using the Gemini text model integrated through LlamaIndex. The assistant is expected to answer questions about Google's Gemini model. To make this possible you will add more context to the assistant using data from a website.

In this tutorial, you'll implement the two main components in a RAG-based architecture:

1. Retriever

    Based on the user's query, the retriever retrieves relevant snippets that add context from the document. In this tutorial, the document is the website data.
    The relevant snippets are passed as context to the next stage - "Generator".

2. Generator

    The relevant snippets from the website data are passed to the LLM along with the user's query to generate accurate answers.

You'll learn more about these stages in the upcoming sections while implementing the application.

## Import the required libraries


```
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader

from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
import re
```

## 1. Retriever

In this stage, you will perform the following steps:

1. Read and parse the website data using LlamaIndex.

2. Create embeddings of the website data.

    Embeddings are numerical representations (vectors) of text. Hence, text with similar meaning will have similar embedding vectors. You'll make use of Gemini's embedding model to create the embedding vectors of the website data.

3. Store the embeddings in Chroma's vector store.
    
    Chroma is a vector database. The Chroma vector store helps in the efficient retrieval of similar vectors. Thus, for adding context to the prompt for the LLM, relevant embeddings of the text matching the user's question can be retrieved easily using Chroma.

4. Create a Retriever from the Chroma vector store.

    The retriever will be used to pass relevant website embeddings to the LLM along with user queries.

### Read and parse the website data

LlamaIndex provides a wide variety of data loaders. To read the website data as a document, you will use the `SimpleWebPageReader` from LlamaIndex.

To know more about how to read and parse input data from different sources using the data loaders of LlamaIndex, read LlamaIndex's [loading data guide](https://docs.llamaindex.ai/en/stable/understanding/loading/loading.html).


```
web_documents = SimpleWebPageReader().load_data(
    ["https://blog.google/technology/ai/google-gemini-ai/"]
)

# Extract the content from the website data document
html_content = web_documents[0].text
```

You can use variety of HTML parsers to extract the required text from the html content.

In this example, you'll use Python's `BeautifulSoup` library to parse the website data. After processing, the extracted text should be converted back to LlamaIndex's `Document` format.


```
# Parse the data.
soup = BeautifulSoup(html_content, 'html.parser')
p_tags = soup.findAll('p')
text_content = ""
for each in p_tags:
    text_content += each.text + "\n"

# Convert back to Document format
documents = [Document(text=text_content)]
```

### Initialize Gemini's embedding model

To create the embeddings from the website data, you'll use Gemini's embedding model, **embedding-001** which supports creating text embeddings.

To use this embedding model, you have to import `GeminiEmbedding` from LlamaIndex. To know more about the embedding model, read Google AI's [language documentation](https://ai.google.dev/models/gemini).


```
from llama_index.embeddings.gemini import GeminiEmbedding

gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
```

### Initialize Gemini

You must import `Gemini` from LlamaIndex to initialize your model.
 In this example, you will use **gemini-1.5-flash-latest**, as it supports text summarization. To know more about the text model, read Google AI's [model documentation](https://ai.google.dev/models/gemini).

You can configure the model parameters such as ***temperature*** or ***top_p***,  using the  ***generation_config*** parameter when initializing the `Gemini` LLM.  To learn more about the model parameters and their uses, read Google AI's [concepts guide](https://ai.google.dev/docs/concepts#model_parameters).


```
from llama_index.llms.gemini import Gemini

# To configure model parameters use the `generation_config` parameter.
# eg. generation_config = {"temperature": 0.7, "topP": 0.8, "topK": 40}
# If you only want to set a custom temperature for the model use the
# "temperature" parameter directly.

llm = Gemini(model_name="models/gemini-1.5-flash-latest")
```

### Store the data using Chroma

 Next, you'll store the embeddings of the website data in Chroma's vector store using LlamaIndex.

 First, you have to initiate a Python client in `chromadb`. Since the plan is to save the data to the disk, you will use the `PersistentClient`. You can read more about the different clients in Chroma in the [client reference guide](https://docs.trychroma.com/reference/Client).

After initializing the client, you have to create a Chroma collection. You'll then initialize the `ChromaVectorStore` class in LlamaIndex using the collection created in the previous step.

Next, you have to set `Settings` and create storage contexts for the vector store.

`Settings` is a collection of commonly used resources that are utilized during the indexing and querying phase in a LlamaIndex pipeline. You can specify the LLM, Embedding model, etc that will be used to create the application in the `Settings`. To know more about `Settings`, read the [module guide for Settings](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings.html).

`StorageContext` is an abstraction offered by LlamaIndex around different types of storage. To know more about storage context, read the [storage context API guide](https://docs.llamaindex.ai/en/stable/api_reference/storage.html).

The final step is to load the documents and build an index over them. LlamaIndex offers several indices that help in retrieving relevant context for a user query. Here you'll use the `VectorStoreIndex` since the website embeddings have to be stored in a vector store.

To create the index you have to pass the storage context along with the documents to the `from_documents` function of `VectorStoreIndex`.
The `VectorStoreIndex` uses the embedding model specified in the `Settings` to create embedding vectors from the documents and stores these vectors in the vector store specified in the storage context. To know more about the
`VectorStoreIndex` you can read the [Using VectorStoreIndex guide](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index.html).


```
# Create a client and a new collection
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")

# Create a vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Set Global settings
Settings.llm = llm
Settings.embed_model = gemini_embedding_model

# Create an index from the documents and save it to the disk.
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

### Create a retriever using Chroma

You'll now create a retriever that can retrieve data embeddings from the newly created Chroma vector store.

First, initialize the `PersistentClient` with the same path you specified while creating the Chroma vector store. You'll then retrieve the collection `"quickstart"` you created previously from Chroma. You can use this collection to initialize the `ChromaVectorStore` in which you store the embeddings of the website data. You can then use the `from_vector_store` function of `VectorStoreIndex` to load the index.


```
# Load from disk
load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
chroma_collection = load_client.get_collection("quickstart")

# Fetch the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store
)

# Check if the retriever is working by trying to fetch the relevant docs related
# to the phrase 'MMLU' (Multimodal Machine Learning Understanding).
# If the length is greater than zero, it means that the retriever is
# functioning well.
# You can ask questions about your data using a generic interface called
# a query engine. You have to use the `as_query_engine` function of the
# index to create a query engine and use the `query` function of query engine
# to inquire the index.
test_query_engine = index.as_query_engine()
response = test_query_engine.query("MMLU")
print(response)
```

    Gemini Ultra is the first model to outperform human experts on MMLU, which uses a combination of 57 subjects such as math, physics, history, law, medicine and ethics for testing both world knowledge and problem-solving abilities. 
    
    

## 2. Generator

The Generator prompts the LLM for an answer when the user asks a question. The retriever you created in the previous stage from the Chroma vector store will be used to pass relevant embeddings from the website data to the LLM to provide more context to the user's query.

You'll perform the following steps in this stage:

1. Create a prompt for answering any question using LlamaIndex.
    
2. Use a query engine to ask a question and prompt the model for an answer.

### Create prompt templates

You'll use LlamaIndex's [PromptTemplate](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts.html) to generate prompts to the LLM for answering questions.

In the `llm_prompt`, the variable `query_str` will be replaced later by the input question, and the variable `context_str` will be replaced by the relevant text from the website retrieved from the Chroma vector store.


```
from llama_index.core import PromptTemplate

template = (
    """ You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)
llm_prompt = PromptTemplate(template)
```

### Prompt the model using Query Engine

You will use the `as_query_engine` function of the `VectorStoreIndex` to create a query engine from the index using the `llm_prompt` passed as the value for the `text_qa_template` argument. You can then use the `query` function of the query engine to prompt the LLM. To know more about custom prompting in LlamaIndex, read LlamaIndex's [prompts usage pattern documentation](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#defining-a-custom-prompt).


```
# Query data from the persisted index
query_engine = index.as_query_engine(text_qa_template=llm_prompt)
response = query_engine.query("What is Gemini?")
print(response)
```

    Gemini is Google's most capable and general AI model yet, designed to be multimodal, meaning it can understand and operate across different types of information like text, code, audio, images, and video. It comes in three sizes: Ultra, Pro, and Nano, optimized for different uses. Gemini Ultra has surpassed human experts on the MMLU benchmark, demonstrating its advanced reasoning abilities. 
    
    
