# Using Weaviate with OpenAI vectorize module for Embeddings Search

This notebook is prepared for a scenario where:
* Your data is not vectorized
* You want to run Vector Search on your data
* You want to use Weaviate with the OpenAI module ([text2vec-openai](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-openai)), to generate vector embeddings for you.

This notebook takes you through a simple flow to set up a Weaviate instance, connect to it (with OpenAI API key), configure data schema, import data (which will automatically generate vector embeddings for your data), and run semantic search.

This is a common requirement for customers who want to store and search our embeddings with their own data in a secure environment to support production use cases such as chatbots, topic modelling and more.

## What is Weaviate

Weaviate is an open-source vector search engine that stores data objects together with their vectors. This allows for combining vector search with structured filtering.

Weaviate uses KNN algorithms to create an vector-optimized index, which allows your queries to run extremely fast. Learn more [here](https://weaviate.io/blog/why-is-vector-search-so-fast).

Weaviate let you use your favorite ML-models, and scale seamlessly into billions of data objects.

### Deployment options

Whatever your scenario or production setup, Weaviate has an option for you. You can deploy Weaviate in the following setups:
* Self-hosted – you can deploy Weaviate with docker locally, or any server you want.
* SaaS – you can use [Weaviate Cloud Service (WCS)](https://console.weaviate.io/) to host your Weaviate instances.
* Hybrid-SaaS – you can deploy Weaviate in your own private Cloud Service.

### Programming languages

Weaviate offers four [client libraries](https://weaviate.io/developers/weaviate/client-libraries), which allow you to communicate from your apps:
* [Python](https://weaviate.io/developers/weaviate/client-libraries/python)
* [JavaScript](https://weaviate.io/developers/weaviate/client-libraries/javascript)
* [Java](https://weaviate.io/developers/weaviate/client-libraries/java)
* [Go](https://weaviate.io/developers/weaviate/client-libraries/go)

Additionally, Weaviate has a [REST layer](https://weaviate.io/developers/weaviate/api/rest/objects). Basically you can call Weaviate from any language that supports REST requests.

## Demo Flow
The demo flow is:
- **Prerequisites Setup**: Create a Weaviate instance and install the required libraries
- **Connect**: Connect to your Weaviate instance 
- **Schema Configuration**: Configure the schema of your data
    - *Note*: Here we can define which OpenAI Embedding Model to use
    - *Note*: Here we can configure which properties to index
- **Import data**: Load a demo dataset and import it into Weaviate
    - *Note*: The import process will automatically index your data - based on the configuration in the schema
    - *Note*: You don't need to explicitly vectorize your data, Weaviate will communicate with OpenAI to do it for you
- **Run Queries**: Query 
    - *Note*: You don't need to explicitly vectorize your queries, Weaviate will communicate with OpenAI to do it for you

Once you've run through this notebook you should have a basic understanding of how to setup and use vector databases, and can move on to more complex use cases making use of our embeddings.

## OpenAI Module in Weaviate
All Weaviate instances come equipped with the [text2vec-openai](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-openai) module.

This module is responsible for handling vectorization during import (or any CRUD operations) and when you run a query.

### No need to manually vectorize data
This is great news for you. With [text2vec-openai](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-openai) you don't need to manually vectorize your data, as Weaviate will call OpenAI for you whenever necessary.

All you need to do is:
1. provide your OpenAI API Key – when you connected to the Weaviate Client
2. define which OpenAI vectorizer to use in your Schema

## Prerequisites

Before we start this project, we need setup the following:

* create a `Weaviate` instance
* install libraries
    * `weaviate-client`
    * `datasets`
    * `apache-beam`
* get your [OpenAI API key](https://beta.openai.com/account/api-keys)

===========================================================
### Create a Weaviate instance

To create a Weaviate instance we have 2 options:

1. (Recommended path) [Weaviate Cloud Service](https://console.weaviate.io/) – to host your Weaviate instance in the cloud. The free sandbox should be more than enough for this cookbook.
2. Install and run Weaviate locally with Docker.

#### Option 1 – WCS Installation Steps

Use [Weaviate Cloud Service](https://console.weaviate.io/) (WCS) to create a free Weaviate cluster.
1. create a free account and/or login to [WCS](https://console.weaviate.io/)
2. create a `Weaviate Cluster` with the following settings:
    * Sandbox: `Sandbox Free`
    * Weaviate Version: Use default (latest)
    * OIDC Authentication: `Disabled`
3. your instance should be ready in a minute or two
4. make a note of the `Cluster Id`. The link will take you to the full path of your cluster (you will need it later to connect to it). It should be something like: `https://your-project-name.weaviate.network` 

#### Option 2 – local Weaviate instance with Docker

Install and run Weaviate locally with Docker.
1. Download the [./docker-compose.yml](./docker-compose.yml) file
2. Then open your terminal, navigate to where your docker-compose.yml file is located, and start docker with: `docker-compose up -d`
3. Once this is ready, your instance should be available at [http://localhost:8080](http://localhost:8080)

Note. To shut down your docker instance you can call: `docker-compose down`

##### Learn more
To learn more, about using Weaviate with Docker see the [installation documentation](https://weaviate.io/developers/weaviate/installation/docker-compose).

===========================================================    
## Install required libraries

Before running this project make sure to have the following libraries:

### Weaviate Python client

The [Weaviate Python client](https://weaviate.io/developers/weaviate/client-libraries/python) allows you to communicate with your Weaviate instance from your Python project.

### datasets & apache-beam

To load sample data, you need the `datasets` library and its dependency `apache-beam`.


```python
# Install the Weaviate client for Python
!pip install weaviate-client>=3.11.0

# Install datasets and apache-beam to load the sample datasets
!pip install datasets apache-beam
```

===========================================================
## Prepare your OpenAI API key

The `OpenAI API key` is used for vectorization of your data at import, and for running queries.

If you don't have an OpenAI API key, you can get one from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys).

Once you get your key, please add it to your environment variables as `OPENAI_API_KEY`.


```python
# Export OpenAI API Key
!export OPENAI_API_KEY="your key"
```


```python
# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.
import os

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'your-key-goes-here'

if os.getenv("OPENAI_API_KEY") is not None:
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")
```

## Connect to your Weaviate instance

In this section, we will:

1. test env variable `OPENAI_API_KEY` – **make sure** you completed the step in [#Prepare-your-OpenAI-API-key](#Prepare-your-OpenAI-API-key)
2. connect to your Weaviate with your `OpenAI API Key`
3. and test the client connection

### The client 

After this step, the `client` object will be used to perform all Weaviate-related operations.


```python
import weaviate
from datasets import load_dataset
import os

# Connect to your Weaviate instance
client = weaviate.Client(
    url="https://your-wcs-instance-name.weaviate.network/",
    # url="http://localhost:8080/",
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="<YOUR-WEAVIATE-API-KEY>"), # comment out this line if you are not using authentication for your Weaviate instance (i.e. for locally deployed instances)
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

# Check if your instance is live and ready
# This should return `True`
client.is_ready()
```

# Schema

In this section, we will:
1. configure the data schema for your data
2. select OpenAI module

> This is the second and final step, which requires OpenAI specific configuration.
> After this step, the rest of instructions wlll only touch on Weaviate, as the OpenAI tasks will be handled automatically.


## What is a schema

In Weaviate you create __schemas__ to capture each of the entities you will be searching.

A schema is how you tell Weaviate:
* what embedding model should be used to vectorize the data
* what your data is made of (property names and types)
* which properties should be vectorized and indexed

In this cookbook we will use a dataset for `Articles`, which contains:
* `title`
* `content`
* `url`

We want to vectorize `title` and `content`, but not the `url`.

To vectorize and query the data, we will use `text-embedding-3-small`.


```python
# Clear up the schema, so that we can recreate it
client.schema.delete_all()
client.schema.get()

# Define the Schema object to use `text-embedding-3-small` on `title` and `content`, but skip it for `url`
article_schema = {
    "class": "Article",
    "description": "A collection of articles",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
          "model": "ada",
          "modelVersion": "002",
          "type": "text"
        }
    },
    "properties": [{
        "name": "title",
        "description": "Title of the article",
        "dataType": ["string"]
    },
    {
        "name": "content",
        "description": "Contents of the article",
        "dataType": ["text"]
    },
    {
        "name": "url",
        "description": "URL to the article",
        "dataType": ["string"],
        "moduleConfig": { "text2vec-openai": { "skip": True } }
    }]
}

# add the Article schema
client.schema.create_class(article_schema)

# get the schema to make sure it worked
client.schema.get()
```

## Import data

In this section we will:
1. load the Simple Wikipedia dataset
2. configure Weaviate Batch import (to make the import more efficient)
3. import the data into Weaviate

> Note: <br/>
> Like mentioned before. We don't need to manually vectorize the data.<br/>
> The [text2vec-openai](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-openai) module will take care of that.


```python
### STEP 1 - load the dataset

from datasets import load_dataset
from typing import List, Iterator

# We'll use the datasets library to pull the Simple Wikipedia dataset for embedding
dataset = list(load_dataset("wikipedia", "20220301.simple")["train"])

# For testing, limited to 2.5k articles for demo purposes
dataset = dataset[:2_500]

# Limited to 25k articles for larger demo purposes
# dataset = dataset[:25_000]

# for free OpenAI acounts, you can use 50 objects
# dataset = dataset[:50]
```


```python
### Step 2 - configure Weaviate Batch, with
# - starting batch size of 100
# - dynamically increase/decrease based on performance
# - add timeout retries if something goes wrong

client.batch.configure(
    batch_size=10, 
    dynamic=True,
    timeout_retries=3,
#   callback=None,
)
```


```python
### Step 3 - import data

print("Importing Articles")

counter=0

with client.batch as batch:
    for article in dataset:
        if (counter %10 == 0):
            print(f"Import {counter} / {len(dataset)} ")

        properties = {
            "title": article["title"],
            "content": article["text"],
            "url": article["url"]
        }
        
        batch.add_data_object(properties, "Article")
        counter = counter+1

print("Importing Articles complete")       
```


```python
# Test that all data has loaded – get object count
result = (
    client.query.aggregate("Article")
    .with_fields("meta { count }")
    .do()
)
print("Object count: ", result["data"]["Aggregate"]["Article"], "\n")
```


```python
# Test one article has worked by checking one object
test_article = (
    client.query
    .get("Article", ["title", "url", "content"])
    .with_limit(1)
    .do()
)["data"]["Get"]["Article"][0]

print(test_article['title'])
print(test_article['url'])
print(test_article['content'])
```

### Search Data

As above, we'll fire some queries at our new Index and get back results based on the closeness to our existing vectors


```python
def query_weaviate(query, collection_name):
    
    nearText = {
        "concepts": [query],
        "distance": 0.7,
    }

    properties = [
        "title", "content", "url",
        "_additional {certainty distance}"
    ]

    result = (
        client.query
        .get(collection_name, properties)
        .with_near_text(nearText)
        .with_limit(10)
        .do()
    )
    
    # Check for errors
    if ("errors" in result):
        print ("\033[91mYou probably have run out of OpenAI API calls for the current minute – the limit is set at 60 per minute.")
        raise Exception(result["errors"][0]['message'])
    
    return result["data"]["Get"][collection_name]
```


```python
query_result = query_weaviate("modern art in Europe", "Article")

for i, article in enumerate(query_result):
    print(f"{i+1}. { article['title']} (Score: {round(article['_additional']['certainty'],3) })")
```


```python
query_result = query_weaviate("Famous battles in Scottish history", "Article")

for i, article in enumerate(query_result):
    print(f"{i+1}. { article['title']} (Score: {round(article['_additional']['certainty'],3) })")
```

Thanks for following along, you're now equipped to set up your own vector databases and use embeddings to do all kinds of cool things - enjoy! For more complex use cases please continue to work through other cookbook examples in this repo.
