# AstraDB

DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) is a serverless vector-capable database built on Cassandra and made conveniently available through an easy-to-use JSON API.

## Overview

The AstraDB Document Loader returns a list of Langchain Documents from an AstraDB database.

The Loader takes the following parameters:

* `api_endpoint`: AstraDB API endpoint. Looks like `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`
* `token`: AstraDB token. Looks like `AstraCS:6gBhNmsk135....`
* `collection_name` : AstraDB collection name
* `namespace`: (Optional) AstraDB namespace
* `filter_criteria`: (Optional) Filter used in the find query
* `projection`: (Optional) Projection used in the find query
* `find_options`: (Optional) Options used in the find query
* `nb_prefetched`: (Optional) Number of documents pre-fetched by the loader
* `extraction_function`: (Optional) A function to convert the AstraDB document to the LangChain `page_content` string. Defaults to `json.dumps`

The following metadata is set to the LangChain Documents metadata output:

```python
{
    metadata : {
        "namespace": "...", 
        "api_endpoint": "...", 
        "collection": "..."
    }
}
```

## Load documents with the Document Loader


```python
from langchain_community.document_loaders import AstraDBLoader
```


```python
from getpass import getpass

ASTRA_DB_API_ENDPOINT = input("ASTRA_DB_API_ENDPOINT = ")
ASTRA_DB_APPLICATION_TOKEN = getpass("ASTRA_DB_APPLICATION_TOKEN = ")
```


```python
loader = AstraDBLoader(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    collection_name="movie_reviews",
    projection={"title": 1, "reviewtext": 1},
    find_options={"limit": 10},
)
```


```python
docs = loader.load()
```


```python
docs[0]
```




    Document(page_content='{"_id": "659bdffa16cbc4586b11a423", "title": "Dangerous Men", "reviewtext": "\\"Dangerous Men,\\" the picture\'s production notes inform, took 26 years to reach the big screen. After having seen it, I wonder: What was the rush?"}', metadata={'namespace': 'default_keyspace', 'api_endpoint': 'https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com', 'collection': 'movie_reviews'})


