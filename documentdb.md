# Amazon Document DB

>[Amazon DocumentDB (with MongoDB Compatibility)](https://docs.aws.amazon.com/documentdb/) makes it easy to set up, operate, and scale MongoDB-compatible databases in the cloud.
> With Amazon DocumentDB, you can run the same application code and use the same drivers and tools that you use with MongoDB.
> Vector search for Amazon DocumentDB combines the flexibility and rich querying capability of a JSON-based document database with the power of vector search.


This notebook shows you how to use [Amazon Document DB Vector Search](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html) to store documents in collections, create indicies and perform vector search queries using approximate nearest neighbor algorithms such "cosine", "euclidean", and "dotProduct". By default, DocumentDB creates Hierarchical Navigable Small World (HNSW) indexes. To learn about other supported vector index types, please refer to the document linked above.

To use DocumentDB, you must first deploy a cluster. Please refer to the [Developer Guide](https://docs.aws.amazon.com/documentdb/latest/developerguide/what-is.html) for more details.

[Sign Up](https://aws.amazon.com/free/) for free to get started today.
        


```python
!pip install pymongo
```


```python
import getpass

# DocumentDB connection string
# i.e., "mongodb://{username}:{pass}@{cluster_endpoint}:{port}/?{params}"
CONNECTION_STRING = getpass.getpass("DocumentDB Cluster URI:")

INDEX_NAME = "izzy-test-index"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")
```

We want to use `OpenAIEmbeddings` so we need to set up our OpenAI environment variables. 


```python
import getpass
import os

# Set up the OpenAI Environment Variables
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["OPENAI_EMBEDDINGS_DEPLOYMENT"] = (
    "smart-agent-embedding-ada"  # the deployment name for the embedding model
)
os.environ["OPENAI_EMBEDDINGS_MODEL_NAME"] = "text-embedding-ada-002"  # the model name
```

Now, we will load the documents into the collection, create the index, and then perform queries against the index.

Please refer to the [documentation](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html) if you have questions about certain parameters


```python
from langchain.vectorstores.documentdb import (
    DocumentDBSimilarityType,
    DocumentDBVectorSearch,
)
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

SOURCE_FILE_NAME = "../../how_to/state_of_the_union.txt"

loader = TextLoader(SOURCE_FILE_NAME)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# OpenAI Settings
model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")


openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    deployment=model_deployment, model=model_name
)
```


```python
from pymongo import MongoClient

INDEX_NAME = "izzy-test-index-2"
NAMESPACE = "izzy_test_db.izzy_test_collection"
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

client: MongoClient = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]

model_deployment = os.getenv(
    "OPENAI_EMBEDDINGS_DEPLOYMENT", "smart-agent-embedding-ada"
)
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")

vectorstore = DocumentDBVectorSearch.from_documents(
    documents=docs,
    embedding=openai_embeddings,
    collection=collection,
    index_name=INDEX_NAME,
)

# number of dimensions used by model above
dimensions = 1536

# specify similarity algorithm, valid options are:
#   cosine (COS), euclidean (EUC), dotProduct (DOT)
similarity_algorithm = DocumentDBSimilarityType.COS

vectorstore.create_index(dimensions, similarity_algorithm)
```




    { 'createdCollectionAutomatically' : false,
       'numIndexesBefore' : 1,
       'numIndexesAfter' : 2,
       'ok' : 1,
       'operationTime' : Timestamp(1703656982, 1)}




```python
# perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What did the President say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)
```


```python
print(docs[0].page_content)
```

    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    

Once the documents have been loaded and the index has been created, you can now instantiate the vector store directly and run queries against the index


```python
vectorstore = DocumentDBVectorSearch.from_connection_string(
    connection_string=CONNECTION_STRING,
    namespace=NAMESPACE,
    embedding=openai_embeddings,
    index_name=INDEX_NAME,
)

# perform a similarity search between a query and the ingested documents
query = "What did the president say about Ketanji Brown Jackson"
docs = vectorstore.similarity_search(query)
```


```python
print(docs[0].page_content)
```

    Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections. 
    
    Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service. 
    
    One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court. 
    
    And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
    


```python
# perform a similarity search between a query and the ingested documents
query = "Which stats did the President share about the U.S. economy"
docs = vectorstore.similarity_search(query)
```


```python
print(docs[0].page_content)
```

    And unlike the $2 Trillion tax cut passed in the previous administration that benefitted the top 1% of Americans, the American Rescue Plan helped working people—and left no one behind. 
    
    And it worked. It created jobs. Lots of jobs. 
    
    In fact—our economy created over 6.5 Million new jobs just last year, more jobs created in one year  
    than ever before in the history of America. 
    
    Our economy grew at a rate of 5.7% last year, the strongest growth in nearly 40 years, the first step in bringing fundamental change to an economy that hasn’t worked for the working people of this nation for too long.  
    
    For the past 40 years we were told that if we gave tax breaks to those at the very top, the benefits would trickle down to everyone else. 
    
    But that trickle-down theory led to weaker economic growth, lower wages, bigger deficits, and the widest gap between those at the top and everyone else in nearly a century.
    

## Question Answering


```python
qa_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)
```


```python
from langchain_core.prompts import PromptTemplate

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
```


```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=qa_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

docs = qa({"query": "gpt-4 compute requirements"})

print(docs["result"])
print(docs["source_documents"])
```
