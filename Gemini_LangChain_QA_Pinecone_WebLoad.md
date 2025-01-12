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

# Gemini API: Question Answering using LangChain and Pinecone

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/langchain/Gemini_LangChain_QA_Pinecone_WebLoad.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


## Overview

[Gemini](https://ai.google.dev/models/gemini) is a family of generative AI models that lets developers generate content and solve problems. These models are designed and trained to handle both text and images as input.

[LangChain](https://www.langchain.com/) is a data framework designed to make integration of Large Language Models (LLM) like Gemini easier for applications.

[Pinecone](https://www.pinecone.io/) is a cloud-first vector database that allows users to search across billions of embeddings with ultra-low query latency.

In this notebook, you'll learn how to create an application that answers questions using data from a website with the help of Gemini, LangChain, and Pinecone.

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install LangChain's Python library, `langchain` and LangChain's integration package for Gemini, `langchain-google-genai`. Next, install LangChain's integration package for the new version of Pinecone, `langchain-pinecone` and the `pinecone-client`, which is Pinecone's Python SDK. Finally, install `langchain-community` to access the `WebBaseLoader` module later.


```
!pip install --quiet langchain==0.1.1
!pip install --quiet langchain-google-genai==0.0.6
!pip install --quiet langchain-pinecone
!pip install --quiet pinecone-client
!pip install --quiet langchain-community==0.0.20
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

### Setup Pinecone

To use Pinecone in your application, you must have an API key. To create an API key you have to set up a Pinecone account. Visit [Pinecone's app page](https://app.pinecone.io/), and Sign up/Log in to your account. Then navigate to the "API Keys" section and copy your API key.

For more detailed instructions on getting the API key, you can read Pinecone's [Quickstart documentation](https://docs.pinecone.io/docs/quickstart#2-get-your-api-key).

Set the environment variable `PINECONE_API_KEY` to configure Pinecone to use your API key.



```
PINECONE_API_KEY=userdata.get('PINECONE_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
```

## Basic steps
LLMs are trained offline on a large corpus of public data. Hence they cannot answer questions based on custom or private data accurately without additional context.

If you want to make use of LLMs to answer questions based on private data, you have to provide the relevant documents as context alongside your prompt. This approach is called Retrieval Augmented Generation (RAG).

You will use this approach to create a question-answering assistant using the Gemini text model integrated through LangChain. The assistant is expected to answer questions about Gemini model. To make this possible you will add more context to the assistant using data from a website.

In this tutorial, you'll implement the two main components in an RAG-based architecture:

1. Retriever

    Based on the user's query, the retriever retrieves relevant snippets that add context from the document. In this tutorial, the document is the website data.
    The relevant snippets are passed as context to the next stage - "Generator".

2. Generator

    The relevant snippets from the website data are passed to the LLM along with the user's query to generate accurate answers.

You'll learn more about these stages in the upcoming sections while implementing the application.

## Import the required libraries


```
from langchain import hub
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone

from pinecone import Pinecone as pc
from pinecone import PodSpec
```

## Retriever

In this stage, you will perform the following steps:

1. Read and parse the website data using LangChain.

2. Create embeddings of the website data.

    Embeddings are numerical representations (vectors) of text. Hence, text with similar meaning will have similar embedding vectors. You'll make use of Gemini's embedding model to create the embedding vectors of the website data.

3. Store the embeddings in Pinecone's vector store.
    
    Pinecone is a vector database. The Pinecone vector store helps in the efficient retrieval of similar vectors. Thus, for adding context to the prompt for the LLM, relevant embeddings of the text matching the user's question can be retrieved easily using Pinecone.

4. Create a Retriever from the Pinecone vector store.

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

### Store the data using Pinecone


To create a Pinecone vector database, first, you have to initialize your Pinecone client connection using the API key you set previously.

In Pinecone, vector embeddings have to be stored in indexes. An index represents the vector data's top-level organizational unit. The vectors in any index must have the same dimensionality and distance metric for calculating similarity. You can read more about indexes in [Pinecone's Indexes documentation](https://docs.pinecone.io/docs/indexes).

First, you'll create an index using Pinecone's `create_index` function. Pinecone allows you to create two types of indexes, Serverless indexes and Pod-based indexes. Pinecone's free starter plan lets you create only one project and one pod-based starter index with sufficient resources to support 100,000 vectors. For this tutorial, you have to create a pod-based starter index. To know more about different indexes and how they can be created, read Pinecone's [create indexes guide](https://docs.pinecone.io/docs/new-api#creating-indexes).


Next, you'll insert the documents you extracted earlier from the website data into the newly created index using LangChain's `Pinecone.from_documents`. Under the hood, this function creates embeddings from the documents created by the document loader of LangChain using any specified embedding model and inserts them into the specified index in a Pinecone vector database.  

You have to specify the `docs` you created from the website data using LangChain's `WebBasedLoader` and the `gemini_embeddings` as the embedding model when invoking the `from_documents` function to create the vector database from the website data.


```
# Initialize Pinecone client

pine_client= pc(
    api_key = os.getenv("PINECONE_API_KEY"),  # API key from app.pinecone.io
    )
index_name = "langchain-demo"

# First, check if the index already exists. If it doesn't, create a new one.
if index_name not in pine_client.list_indexes().names():
    # Create a new index.
    # https://docs.pinecone.io/docs/new-api#creating-a-starter-index
    print("Creating index")
    pine_client.create_index(name=index_name,
                      # `cosine` distance metric compares different documents
                      # for similarity.
                      # Read more about different distance metrics from
                      # https://docs.pinecone.io/docs/indexes#distance-metrics.
                      metric="cosine",
                      # The Gemini embedding model `embedding-001` uses
                      # 768 dimensions.
                      dimension=768,
                      # Specify the pod details.
                      spec=PodSpec(
                        # Starter indexes are hosted in the `gcp-starter`
                        # environment.
                        environment="gcp-starter",
                        pod_type="starter",
                        pods=1)
    )
    print(pine_client.describe_index(index_name))

vectorstore = Pinecone.from_documents(docs,
                      gemini_embeddings, index_name=index_name)

```

### Create a retriever using Pinecone

You'll now create a retriever that can retrieve website data embeddings from the newly created Pinecone vector store. This retriever can be later used to pass embeddings that provide more context to the LLM for answering user's queries.

Invoke the `as_retriever` function of the vector store you initialized in the last step, to create a retriever.


```
retriever = vectorstore.as_retriever()
# Check if the retriever is working by trying to fetch the relevant docs related
# to the word 'MMLU'(Massive Multitask Language Understanding). If the length is
# greater than zero, it means that the retriever is functioning well.
print(len(retriever.invoke("MMLU")))
```

    3
    

## Generator

The Generator prompts the LLM for an answer when the user asks a question. The retriever you created in the previous stage from the Pinecone vector store will be used to pass relevant embeddings from the website data to the LLM to provide more context to the user's query.

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

In the `llm_prompt`, the variable `question` will be replaced later by the input question, and the variable `context` will be replaced by the relevant text from the website retrieved from the Pinecone vector store.


```
# Prompt template to query Gemini
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)
```

    input_variables=['context', 'question'] template="You are an assistant for question-answering tasks.\nUse the following context to answer the question.\nIf you don't know the answer, just say that you don't know.\nUse five sentences maximum and keep the answer concise.\n\nQuestion: {question}\nContext: {context}\nAnswer:"
    

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
# This is called a chain because you are chaining
# together different elements with the LLM.
# In the following example, to create a stuff chain,
# you will combine content, prompt, LLM model, and
# output parser together like a chain using LCEL.
#
# The chain implements the following pipeline:
# 1. Extract data from documents and save to the variable `context`.
# 2. Use the `RunnablePassthrough` option to provide question during invoke.
# 3. The `context` and `question` are then passed to the prompt and
#    input variables in the prompt are populated.
# 4. The prompt is then passed to the LLM (`gemini-pro`).
# 5. Output from the LLM is passed through an output parser
#    to structure the model response.
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




    "Gemini is Google's latest and most capable AI model, designed to be flexible and efficient across various platforms. It excels in various tasks like image, audio, and video understanding, mathematical reasoning, and coding.  Gemini comes in three sizes: Ultra, Pro, and Nano, catering to different complexities and computational needs.  Its multimodal capabilities and advanced reasoning abilities are considered state-of-the-art in many domains. \n"


