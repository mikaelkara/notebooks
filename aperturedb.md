# ApertureDB

[ApertureDB](https://docs.aperturedata.io) is a database that stores, indexes, and manages multi-modal data like text, images, videos, bounding boxes, and embeddings, together with their associated metadata.

This notebook explains how to use the embeddings functionality of ApertureDB.

## Install ApertureDB Python SDK

This installs the [Python SDK](https://docs.aperturedata.io/category/aperturedb-python-sdk) used to write client code for ApertureDB.


```python
%pip install --upgrade --quiet aperturedb
```

    Note: you may need to restart the kernel to use updated packages.
    

## Run an ApertureDB instance

To continue, you should have an [ApertureDB instance up and running](https://docs.aperturedata.io/HowToGuides/start/Setup) and configure your environment to use it.  
There are various ways to do that, for example:

```bash
docker run --publish 55555:55555 aperturedata/aperturedb-standalone
adb config create local --active --no-interactive
```

## Download some web documents
We're going to do a mini-crawl here of one web page.


```python
# For loading documents from web
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.aperturedata.io")
docs = loader.load()
```

    USER_AGENT environment variable not set, consider setting it to identify your requests.
    

## Select embeddings model

We want to use OllamaEmbeddings so we have to import the necessary modules.

Ollama can be set up as a docker container as described in the [documentation](https://hub.docker.com/r/ollama/ollama), for example:
```bash
# Run server
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# Tell server to load a specific model
docker exec ollama ollama run llama2
```


```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings()
```

## Split documents into segments

We want to turn our single document into multiple segments.


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
```

## Create vectorstore from documents and embeddings

This code creates a vectorstore in the ApertureDB instance.
Within the instance, this vectorstore is represented as a "[descriptor set](https://docs.aperturedata.io/category/descriptorset-commands)".
By default, the descriptor set is named `langchain`.  The following code will generate embeddings for each document and store them in ApertureDB as descriptors.  This will take a few seconds as the embeddings are bring generated.


```python
from langchain_community.vectorstores import ApertureDB

vector_db = ApertureDB.from_documents(documents, embeddings)
```

## Select a large language model

Again, we use the Ollama server we set up for local processing.


```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")
```

## Build a RAG chain

Now we have all the components we need to create a RAG (Retrieval-Augmented Generation) chain.  This chain does the following:
1. Generate embedding descriptor for user query
2. Find text segments that are similar to the user query using the vector store
3. Pass user query and context documents to the LLM using a prompt template
4. Return the LLM's answer


```python
# Create prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")


# Create a chain that passes documents to an LLM
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)


# Treat the vectorstore as a document retriever
retriever = vector_db.as_retriever()


# Create a RAG chain that connects the retriever to the LLM
from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

    Based on the provided context, ApertureDB can store images. In fact, it is specifically designed to manage multimodal data such as images, videos, documents, embeddings, and associated metadata including annotations. So, ApertureDB has the capability to store and manage images.
    

## Run the RAG chain

Finally we pass a question to the chain and get our answer.  This will take a few seconds to run as the LLM generates an answer from the query and context documents.


```python
user_query = "How can ApertureDB store images?"
response = retrieval_chain.invoke({"input": user_query})
print(response["answer"])
```

    Based on the provided context, ApertureDB can store images in several ways:
    
    1. Multimodal data management: ApertureDB offers a unified interface to manage multimodal data such as images, videos, documents, embeddings, and associated metadata including annotations. This means that images can be stored along with other types of data in a single database instance.
    2. Image storage: ApertureDB provides image storage capabilities through its integration with the public cloud providers or on-premise installations. This allows customers to host their own ApertureDB instances and store images on their preferred cloud provider or on-premise infrastructure.
    3. Vector database: ApertureDB also offers a vector database that enables efficient similarity search and classification of images based on their semantic meaning. This can be useful for applications where image search and classification are important, such as in computer vision or machine learning workflows.
    
    Overall, ApertureDB provides flexible and scalable storage options for images, allowing customers to choose the deployment model that best suits their needs.
    
