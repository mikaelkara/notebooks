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

# Gemini API: Question Answering using LangChain and Chroma

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/langchain/Gemini_LangChain_QA_Chroma_WebLoad.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


## Overview

[Gemini](https://ai.google.dev/models/gemini) is a family of generative AI models that lets developers generate content and solve problems. These models are designed and trained to handle both text and images as input.

[LangChain](https://www.langchain.com/) is a data framework designed to make integration of Large Language Models (LLM) like Gemini easier for applications.

[Chroma](https://docs.trychroma.com/) is an open-source embedding database focused on simplicity and developer productivity. Chroma allows users to store embeddings and their metadata, embed documents and queries, and search the embeddings quickly.

In this notebook, you'll learn how to create an application that answers questions using data from a website with the help of Gemini, LangChain, and Chroma.

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install LangChain's Python library, `langchain` and LangChain's integration package for Gemini, `langchain-google-genai`. Next, install Chroma's Python client SDK, `chromadb`.


```
!pip install --quiet langchain-core==0.1.23
!pip install --quiet langchain==0.1.1
!pip install --quiet langchain-google-genai==0.0.6
!pip install --quiet -U langchain-community==0.0.20
!pip install --quiet chromadb
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m241.2/241.2 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.4/55.4 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.0/53.0 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m802.4/802.4 kB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m12.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.2/49.2 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.9/302.9 kB[0m [31m15.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m20.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m26.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m25.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m32.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m40.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m46.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m48.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.9/1.9 MB[0m [31m33.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m43.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m47.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m42.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m47.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m32.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m63.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m65.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m42.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m61.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m75.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m146.9/146.9 kB[0m [31m4.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m598.7/598.7 kB[0m [31m20.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m559.5/559.5 kB[0m [31m8.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m42.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m92.0/92.0 kB[0m [31m7.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.4/62.4 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.3/41.3 kB[0m [31m3.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.8/6.8 MB[0m [31m87.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m59.9/59.9 kB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m107.0/107.0 kB[0m [31m12.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.3/67.3 kB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [?25h  Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m283.7/283.7 kB[0m [31m25.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m74.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.6/67.6 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m145.0/145.0 kB[0m [31m16.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m8.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m71.9/71.9 kB[0m [31m8.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.6/53.6 kB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m9.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.0/46.0 kB[0m [31m6.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m52.5/52.5 kB[0m [31m6.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.5/130.5 kB[0m [31m15.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m341.4/341.4 kB[0m [31m32.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.4/3.4 MB[0m [31m87.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m64.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m130.2/130.2 kB[0m [31m15.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m307.7/307.7 kB[0m [31m30.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m86.8/86.8 kB[0m [31m11.0 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for pypika (pyproject.toml) ... [?25l[?25hdone
    


```
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
```

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

You will use this approach to create a question-answering assistant using the Gemini text model integrated through LangChain. The assistant is expected to answer questions about the Gemini model. To make this possible you will add more context to the assistant using data from a website.

In this tutorial, you'll implement the two main components in an RAG-based architecture:

1. Retriever

    Based on the user's query, the retriever retrieves relevant snippets that add context from the document. In this tutorial, the document is the website data.
    The relevant snippets are passed as context to the next stage - "Generator".

2. Generator

    The relevant snippets from the website data are passed to the LLM along with the user's query to generate accurate answers.

You'll learn more about these stages in the upcoming sections while implementing the application.

## Retriever

In this stage, you will perform the following steps:

1. Read and parse the website data using LangChain.

2. Create embeddings of the website data.

    Embeddings are numerical representations (vectors) of text. Hence, text with similar meaning will have similar embedding vectors. You'll make use of Gemini's embedding model to create the embedding vectors of the website data.

3. Store the embeddings in Chroma's vector store.
    
    Chroma is a vector database. The Chroma vector store helps in the efficient retrieval of similar vectors. Thus, for adding context to the prompt for the LLM, relevant embeddings of the text matching the user's question can be retrieved easily using Chroma.

4. Create a Retriever from the Chroma vector store.

    The retriever will be used to pass relevant website embeddings to the LLM along with user queries.

### Read and parse the website data

LangChain provides a wide variety of document loaders. To read the website data as a document, you will use the `WebBaseLoader` from LangChain.

To know more about how to read and parse input data from different sources using the document loaders of LangChain, read LangChain's [document loaders guide](https://python.langchain.com/docs/integrations/document_loaders).


```
loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/")
docs = loader.load()
```

If you only want to select a specific portion of the website data to add context to the prompt, you can use regex, text slicing, or text splitting.

In this example, you'll use Python's `split()` function to extract the required portion of the text. The extracted text should be converted back to LangChain's `Document` format.


```
# Extract the text from the website data document
text_content = docs[0].page_content

# The text content between the substrings "code, audio, image and video." to
# "Cloud TPU v5p" is relevant for this tutorial. You can use Python's `split()`
# to select the required content.
text_content_1 = text_content.split("code, audio, image and video.",1)[1]
final_text = text_content_1.split("Cloud TPU v5p",1)[0]

# Convert the text to LangChain's `Document` format
docs = [Document(page_content=final_text, metadata={"source": "local"})]
```

### Initialize Gemini's embedding model

To create the embeddings from the website data, you'll use Gemini's embedding model, **embedding-001** which supports creating text embeddings.

To use this embedding model, you have to import `GoogleGenerativeAIEmbeddings` from LangChain. To know more about the embedding model, read Google AI's [language documentation](https://ai.google.dev/models/gemini).


```
from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
```

### Store the data using Chroma

To create a Chroma vector database from the website data, you will use the `from_documents` function of `Chroma`. Under the hood, this function creates embeddings from the documents created by the document loader of LangChain using any specified embedding model and stores them in a Chroma vector database.  

You have to specify the `docs` you created from the website data using LangChain's `WebBasedLoader` and the `gemini_embeddings` as the embedding model when invoking the `from_documents` function to create the vector database from the website data. You can also specify a directory in the `persist_directory` argument to store the vector store on the disk. If you don't specify a directory, the data will be ephemeral in-memory.



```
# Save to disk
vectorstore = Chroma.from_documents(
                     documents=docs,                 # Data
                     embedding=gemini_embeddings,    # Embedding model
                     persist_directory="./chroma_db" # Directory to save data
                     )
```

### Create a retriever using Chroma

You'll now create a retriever that can retrieve website data embeddings from the newly created Chroma vector store. This retriever can be later used to pass embeddings that provide more context to the LLM for answering user's queries.


To load the vector store that you previously stored in the disk, you can specify the name of the directory that contains the vector store in `persist_directory` and the embedding model in the `embedding_function` arguments of Chroma's initializer.

You can then invoke the `as_retriever` function of `Chroma` on the vector store to create a retriever.


```
# Load from disk
vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=gemini_embeddings   # Embedding model
                   )
# Get the Retriever interface for the store to use later.
# When an unstructured query is given to a retriever it will return documents.
# Read more about retrievers in the following link.
# https://python.langchain.com/docs/modules/data_connection/retrievers/
#
# Since only 1 document is stored in the Chroma vector store, search_kwargs `k`
# is set to 1 to decrease the `k` value of chroma's similarity search from 4 to
# 1. If you don't pass this value, you will get a warning.
retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

# Check if the retriever is working by trying to fetch the relevant docs related
# to the word 'MMLU' (Massive Multitask Language Understanding). If the length is greater than zero, it means that
# the retriever is functioning well.
print(len(retriever.get_relevant_documents("MMLU")))
```

    1
    

## Generator

The Generator prompts the LLM for an answer when the user asks a question. The retriever you created in the previous stage from the Chroma vector store will be used to pass relevant embeddings from the website data to the LLM to provide more context to the user's query.

You'll perform the following steps in this stage:

1. Chain together the following:
    * A prompt for extracting the relevant embeddings using the retriever.
    * A prompt for answering any question using LangChain.
    * An LLM model from Gemini for prompting.
    
2. Run the created chain with a question as input to prompt the model for an answer.


### Initialize Gemini

You must import `ChatGoogleGenerativeAI` from LangChain to initialize your model.
 In this example, you will use **gemini-1.5-flash-latest**, as it supports text summarization. To know more about the text model, read Google AI's [language documentation](https://ai.google.dev/models/gemini).

You can configure the model parameters such as ***temperature*** or ***top_p***,  by passing the appropriate values when initializing the `ChatGoogleGenerativeAI` LLM.  To learn more about the parameters and their uses, read Google AI's [concepts guide](https://ai.google.dev/docs/concepts#model_parameters).


```
from langchain_google_genai import ChatGoogleGenerativeAI

# To configure model parameters use the `generation_config` parameter.
# eg. generation_config = {"temperature": 0.7, "topP": 0.8, "topK": 40}
# If you only want to set a custom temperature for the model use the
# "temperature" parameter directly.

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
```

### Create prompt templates

You'll use LangChain's [PromptTemplate](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/) to generate prompts to the LLM for answering questions.

In the `llm_prompt`, the variable `question` will be replaced later by the input question, and the variable `context` will be replaced by the relevant text from the website retrieved from the Chroma vector store.


```
# Prompt template to query Gemini
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)
```

    input_variables=['context', 'question'] template="You are an assistant for question-answering tasks.\nUse the following context to answer the question.\nIf you don't know the answer, just say that you don't know.\nUse five sentences maximum and keep the answer concise.\n\nQuestion: {question} \nContext: {context} \nAnswer:"
    

### Create a stuff documents chain

LangChain provides [Chains](https://python.langchain.com/docs/modules/chains/) for chaining together LLMs with each other or other components for complex applications. You will create a **stuff documents chain** for this application. A stuff documents chain lets you combine all the relevant documents, insert them into the prompt, and pass that prompt to the LLM.

You can create a stuff documents chain using the [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language).

To learn more about different types of document chains, read LangChain's [chains guide](https://python.langchain.com/docs/modules/chains/document/).

The stuff documents chain for this application retrieves the relevant website data and passes it as the context to an LLM prompt along with the input question.


```
# Combine data from documents to readable string format.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create stuff documents chain using LCEL.
#
# This is called a chain because you are chaining together different elements
# with the LLM. In the following example, to create the stuff chain, you will
# combine the relevant context from the website data matching the question, the
# LLM model, and the output parser together like a chain using LCEL.
#
# The chain implements the following pipeline:
# 1. Extract the website data relevant to the question from the Chroma
#    vector store and save it to the variable `context`.
# 2. `RunnablePassthrough` option to provide `question` when invoking
#    the chain.
# 3. The `context` and `question` are then passed to the prompt where they
#    are populated in the respective variables.
# 4. This prompt is then passed to the LLM (`gemini-pro`).
# 5. Output from the LLM is passed through an output parser
#    to structure the model's response.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
```

### Prompt the model

You can now query the LLM by passing any question to the `invoke()` function of the stuff documents chain you created previously.


```
rag_chain.invoke("What is Gemini?")
```




    "Gemini is Google's largest and most capable AI model, designed to be highly flexible and efficient across various devices, from data centers to mobile devices. It's optimized in three sizes: Ultra, Pro, and Nano, catering to different complexity levels and task requirements. Gemini surpasses state-of-the-art performance on multiple benchmarks, including text, code, and multimodal tasks, showcasing its advanced reasoning abilities. This model is trained to understand and process information across various modalities, including text, images, audio, and more, making it ideal for complex tasks like coding and scientific research. \n"



# Conclusion

That's it. You have successfully created an LLM application that answers questions using data from a website with the help of Gemini, LangChain, and Chroma.
