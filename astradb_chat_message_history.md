# Astra DB 

> DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Cassandra and made conveniently available through an easy-to-use JSON API.

This notebook goes over how to use Astra DB to store chat message history.

## Setting up

To run this notebook you need a running Astra DB. Get the connection secrets on your Astra dashboard:

- the API Endpoint looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`;
- the Token looks like `AstraCS:6gBhNmsk135...`.


```python
%pip install --upgrade --quiet  "astrapy>=0.7.1 langchain-community" 
```

### Set up the database connection parameters and secrets


```python
import getpass

ASTRA_DB_API_ENDPOINT = input("ASTRA_DB_API_ENDPOINT = ")
ASTRA_DB_APPLICATION_TOKEN = getpass.getpass("ASTRA_DB_APPLICATION_TOKEN = ")
```

    ASTRA_DB_API_ENDPOINT =  https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com
    ASTRA_DB_APPLICATION_TOKEN =  ········
    

Depending on whether local or cloud-based Astra DB, create the corresponding database connection "Session" object.

## Example


```python
from langchain_community.chat_message_histories import AstraDBChatMessageHistory

message_history = AstraDBChatMessageHistory(
    session_id="test-session",
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

message_history.add_user_message("hi!")

message_history.add_ai_message("whats up?")
```


```python
message_history.messages
```




    [HumanMessage(content='hi!'), AIMessage(content='whats up?')]


