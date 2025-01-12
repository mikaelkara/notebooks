[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/advanced-rag-langchain-mongodb/)


# Adding Semantic Caching and Memory to your RAG Application using MongoDB and LangChain

In this notebook, we will see how to use the new MongoDBCache and MongoDBChatMessageHistory in your RAG application.


## Step 1: Install required libraries

- **datasets**: Python library to get access to datasets available on Hugging Face Hub

- **langchain**: Python toolkit for LangChain

- **langchain-mongodb**: Python package to use MongoDB as a vector store, semantic cache, chat history store etc. in LangChain

- **langchain-openai**: Python package to use OpenAI models with LangChain

- **pymongo**: Python toolkit for MongoDB

- **pandas**: Python library for data analysis, exploration, and manipulation


```python
! pip install -qU datasets langchain langchain-mongodb langchain-openai pymongo pandas
```

## Step 2: Setup pre-requisites

* Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

* Set the OpenAI API key. Steps to obtain an API key as [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)


```python
import getpass
```


```python
MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")
```

    Enter your MongoDB connection string:········
    


```python
OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key:")
```

    Enter your OpenAI API key:········
    


```python
# Optional-- If you want to enable Langsmith -- good for debugging
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

    ········
    

## Step 3: Download the dataset

We will be using MongoDB's [embedded_movies](https://huggingface.co/datasets/MongoDB/embedded_movies) dataset


```python
import pandas as pd
from datasets import load_dataset
```


```python
# Ensure you have an HF_TOKEN in your development enviornment:
# access tokens can be created or copied from the Hugging Face platform (https://huggingface.co/docs/hub/en/security-tokens)

# Load MongoDB's embedded_movies dataset from Hugging Face
# https://huggingface.co/datasets/MongoDB/airbnb_embeddings

data = load_dataset("MongoDB/embedded_movies")
```


```python
df = pd.DataFrame(data["train"])
```

## Step 4: Data analysis

Make sure length of the dataset is what we expect, drop Nones etc.


```python
# Previewing the contents of the data
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fullplot</th>
      <th>type</th>
      <th>plot_embedding</th>
      <th>num_mflix_comments</th>
      <th>runtime</th>
      <th>writers</th>
      <th>imdb</th>
      <th>countries</th>
      <th>rated</th>
      <th>plot</th>
      <th>title</th>
      <th>languages</th>
      <th>metacritic</th>
      <th>directors</th>
      <th>awards</th>
      <th>genres</th>
      <th>poster</th>
      <th>cast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Young Pauline is left a lot of money when her ...</td>
      <td>movie</td>
      <td>[0.00072939653, -0.026834568, 0.013515796, -0....</td>
      <td>0</td>
      <td>199.0</td>
      <td>[Charles W. Goddard (screenplay), Basil Dickey...</td>
      <td>{'id': 4465, 'rating': 7.6, 'votes': 744}</td>
      <td>[USA]</td>
      <td>None</td>
      <td>Young Pauline is left a lot of money when her ...</td>
      <td>The Perils of Pauline</td>
      <td>[English]</td>
      <td>NaN</td>
      <td>[Louis J. Gasnier, Donald MacKenzie]</td>
      <td>{'nominations': 0, 'text': '1 win.', 'wins': 1}</td>
      <td>[Action]</td>
      <td>https://m.media-amazon.com/images/M/MV5BMzgxOD...</td>
      <td>[Pearl White, Crane Wilbur, Paul Panzer, Edwar...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Only keep records where the fullplot field is not null
df = df[df["fullplot"].notna()]
```


```python
# Renaming the embedding field to "embedding" -- required by LangChain
df.rename(columns={"plot_embedding": "embedding"}, inplace=True)
```

## Step 5: Create a simple RAG chain using MongoDB as the vector store


```python
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Initialize MongoDB python client
client = MongoClient(MONGODB_URI, appname="devrel.content.python")

DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]
```


```python
# Delete any existing records in the collection
collection.delete_many({})
```




    DeleteResult({'n': 1000, 'electionId': ObjectId('7fffffff00000000000000f6'), 'opTime': {'ts': Timestamp(1710523288, 1033), 't': 246}, 'ok': 1.0, '$clusterTime': {'clusterTime': Timestamp(1710523288, 1042), 'signature': {'hash': b"i\xa8\xe9'\x1ed\xf2u\xf3L\xff\xb1\xf5\xbfA\x90\xabJ\x12\x83", 'keyId': 7299545392000008318}}, 'operationTime': Timestamp(1710523288, 1033)}, acknowledged=True)




```python
# Data Ingestion
records = df.to_dict("records")
collection.insert_many(records)

print("Data ingestion into MongoDB completed")
```

    Data ingestion into MongoDB completed
    


```python
from langchain_openai import OpenAIEmbeddings

# Using the text-embedding-ada-002 since that's what was used to create embeddings in the movies dataset
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)
```


```python
# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot",
)
```


```python
# Using the MongoDB vector store as a retriever in a RAG chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Generate context using the retriever, and pass the user question through
retrieve = {
    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough(),
}
template = """Answer the question based only on the following context: \
{context}

Question: {question}
"""
# Defining the chat prompt
prompt = ChatPromptTemplate.from_template(template)
# Defining the model to be used for chat completion
model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# Parse output as a string
parse_output = StrOutputParser()

# Naive RAG chain
naive_rag_chain = retrieve | prompt | model | parse_output
```


```python
naive_rag_chain.invoke("What is the best movie to watch when sad?")
```




    'Once a Thief'



## Step 6: Create a RAG chain with chat history


```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
```


```python
def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGODB_URI, session_id, database_name=DB_NAME, collection_name="history"
    )
```


```python
# Given a follow-up question and history, create a standalone question
standalone_system_prompt = """
Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
Do NOT answer the question, just reformulate it if needed, otherwise return it as is. \
Only return the final standalone question. \
"""
standalone_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", standalone_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

question_chain = standalone_question_prompt | model | parse_output
```


```python
# Generate context by passing output of the question_chain i.e. the standalone question to the retriever
retriever_chain = RunnablePassthrough.assign(
    context=question_chain
    | retriever
    | (lambda docs: "\n\n".join([d.page_content for d in docs]))
)
```


```python
# Create a prompt that includes the context, history and the follow-up question
rag_system_prompt = """Answer the question based only on the following context: \
{context}
"""
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
```


```python
# RAG chain
rag_chain = retriever_chain | rag_prompt | model | parse_output
```


```python
# RAG chain with history
with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
with_message_history.invoke(
    {"question": "What is the best movie to watch when sad?"},
    {"configurable": {"session_id": "1"}},
)
```




    'The best movie to watch when feeling down could be "Last Action Hero." It\'s a fun and action-packed film that blends reality and fantasy, offering an escape from the real world and providing an entertaining distraction.'




```python
with_message_history.invoke(
    {
        "question": "Hmmm..I don't want to watch that one. Can you suggest something else?"
    },
    {"configurable": {"session_id": "1"}},
)
```




    'I apologize for the confusion. Another movie that might lift your spirits when you\'re feeling sad is "Smilla\'s Sense of Snow." It\'s a mystery thriller that could engage your mind and distract you from your sadness with its intriguing plot and suspenseful storyline.'




```python
with_message_history.invoke(
    {"question": "How about something more light?"},
    {"configurable": {"session_id": "1"}},
)
```




    'For a lighter movie option, you might enjoy "Cousins." It\'s a comedy film set in Barcelona with action and humor, offering a fun and entertaining escape from reality. The storyline is engaging and filled with comedic moments that could help lift your spirits.'



## Step 7: Get faster responses using Semantic Cache

**NOTE:** Semantic cache only caches the input to the LLM. When using it in retrieval chains, remember that documents retrieved can change between runs resulting in cache misses for semantically similar queries.


```python
from langchain_core.globals import set_llm_cache
from langchain_mongodb.cache import MongoDBAtlasSemanticCache

set_llm_cache(
    MongoDBAtlasSemanticCache(
        connection_string=MONGODB_URI,
        embedding=embeddings,
        collection_name="semantic_cache",
        database_name=DB_NAME,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        wait_until_ready=True,  # Optional, waits until the cache is ready to be used
    )
)
```


```python
%%time
naive_rag_chain.invoke("What is the best movie to watch when sad?")
```

    CPU times: user 87.8 ms, sys: 670 µs, total: 88.5 ms
    Wall time: 1.24 s
    




    'Once a Thief'




```python
%%time
naive_rag_chain.invoke("What is the best movie to watch when sad?")
```

    CPU times: user 43.5 ms, sys: 4.16 ms, total: 47.7 ms
    Wall time: 255 ms
    




    'Once a Thief'




```python
%%time
naive_rag_chain.invoke("Which movie do I watch when sad?")
```

    CPU times: user 115 ms, sys: 171 µs, total: 115 ms
    Wall time: 1.38 s
    




    'I would recommend watching "Last Action Hero" when sad, as it is a fun and action-packed film that can help lift your spirits.'


