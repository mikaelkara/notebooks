# MongoDB

[MongoDB](https://www.mongodb.com/) is a NoSQL , document-oriented database that supports JSON-like documents with a dynamic schema.

## Overview

The MongoDB Document Loader returns a list of Langchain Documents from a MongoDB database.

The Loader requires the following parameters:

*   MongoDB connection string
*   MongoDB database name
*   MongoDB collection name
*   (Optional) Content Filter dictionary
*   (Optional) List of field names to include in the output

The output takes the following format:

- pageContent= Mongo Document
- metadata=\{'database': '[database_name]', 'collection': '[collection_name]'\}

## Load the Document Loader


```python
# add this import for running in jupyter notebook
import nest_asyncio

nest_asyncio.apply()
```


```python
from langchain_community.document_loaders.mongodb import MongodbLoader
```


```python
loader = MongodbLoader(
    connection_string="mongodb://localhost:27017/",
    db_name="sample_restaurants",
    collection_name="restaurants",
    filter_criteria={"borough": "Bronx", "cuisine": "Bakery"},
    field_names=["name", "address"],
)
```


```python
docs = loader.load()

len(docs)
```




    71




```python
docs[0]
```




    Document(page_content="Morris Park Bake Shop {'building': '1007', 'coord': [-73.856077, 40.848447], 'street': 'Morris Park Ave', 'zipcode': '10462'}", metadata={'database': 'sample_restaurants', 'collection': 'restaurants'})


