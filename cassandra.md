# Cassandra

[Cassandra](https://cassandra.apache.org/) is a NoSQL, row-oriented, highly scalable and highly available database.Starting with version 5.0, the database ships with [vector search capabilities](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html).

## Overview

The Cassandra Document Loader returns a list of Langchain Documents from a Cassandra database.

You must either provide a CQL query or a table name to retrieve the documents.
The Loader takes the following parameters:

* table: (Optional) The table to load the data from.
* session: (Optional) The cassandra driver session. If not provided, the cassio resolved session will be used.
* keyspace: (Optional) The keyspace of the table. If not provided, the cassio resolved keyspace will be used.
* query: (Optional) The query used to load the data.
* page_content_mapper: (Optional) a function to convert a row to string page content. The default converts the row to JSON.
* metadata_mapper: (Optional) a function to convert a row to metadata dict.
* query_parameters: (Optional) The query parameters used when calling session.execute .
* query_timeout: (Optional) The query timeout used when calling session.execute .
* query_custom_payload: (Optional) The query custom_payload used when calling `session.execute`.
* query_execution_profile: (Optional) The query execution_profile used when calling `session.execute`.
* query_host: (Optional) The query host used when calling `session.execute`.
* query_execute_as: (Optional) The query execute_as used when calling `session.execute`.

## Load documents with the Document Loader


```python
from langchain_community.document_loaders import CassandraLoader
```

### Init from a cassandra driver Session

You need to create a `cassandra.cluster.Session` object, as described in the [Cassandra driver documentation](https://docs.datastax.com/en/developer/python-driver/latest/api/cassandra/cluster/#module-cassandra.cluster). The details vary (e.g. with network settings and authentication), but this might be something like:


```python
from cassandra.cluster import Cluster

cluster = Cluster()
session = cluster.connect()
```

You need to provide the name of an existing keyspace of the Cassandra instance:


```python
CASSANDRA_KEYSPACE = input("CASSANDRA_KEYSPACE = ")
```

Creating the document loader:


```python
loader = CassandraLoader(
    table="movie_reviews",
    session=session,
    keyspace=CASSANDRA_KEYSPACE,
)
```


```python
docs = loader.load()
```


```python
docs[0]
```




    Document(page_content='Row(_id=\'659bdffa16cbc4586b11a423\', title=\'Dangerous Men\', reviewtext=\'"Dangerous Men,"  the picture\\\'s production notes inform, took 26 years to reach the big screen. After having seen it, I wonder: What was the rush?\')', metadata={'table': 'movie_reviews', 'keyspace': 'default_keyspace'})



### Init from cassio

It's also possible to use cassio to configure the session and keyspace.


```python
import cassio

cassio.init(contact_points="127.0.0.1", keyspace=CASSANDRA_KEYSPACE)

loader = CassandraLoader(
    table="movie_reviews",
)

docs = loader.load()
```

#### Attribution statement

> Apache Cassandra, Cassandra and Apache are either registered trademarks or trademarks of the [Apache Software Foundation](http://www.apache.org/) in the United States and/or other countries.
