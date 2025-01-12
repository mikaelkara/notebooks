# Using Tair as a vector database for OpenAI embeddings

This notebook guides you step by step on using Tair as a vector database for OpenAI embeddings.

This notebook presents an end-to-end process of:
1. Using precomputed embeddings created by OpenAI API.
2. Storing the embeddings in a cloud instance of Tair.
3. Converting raw text query to an embedding with OpenAI API.
4. Using Tair to perform the nearest neighbour search in the created collection.

### What is Tair

[Tair](https://www.alibabacloud.com/help/en/tair/latest/what-is-tair) is a cloud native in-memory database service that is developed by Alibaba Cloud. Tair is compatible with open source Redis and provides a variety of data models and enterprise-class capabilities to support your real-time online scenarios. Tair also introduces persistent memory-optimized instances that are based on the new non-volatile memory (NVM) storage medium. These instances can reduce costs by 30%, ensure data persistence, and provide almost the same performance as in-memory databases. Tair has been widely used in areas such as government affairs, finance, manufacturing, healthcare, and pan-Internet to meet their high-speed query and computing requirements.

[Tairvector](https://www.alibabacloud.com/help/en/tair/latest/tairvector) is an in-house data structure that provides high-performance real-time storage and retrieval of vectors. TairVector provides two indexing algorithms: Hierarchical Navigable Small World (HNSW) and Flat Search. Additionally, TairVector supports multiple distance functions, such as Euclidean distance, inner product, and Jaccard distance. Compared with traditional vector retrieval services, TairVector has the following advantages:
- Stores all data in memory and supports real-time index updates to reduce latency of read and write operations.
- Uses an optimized data structure in memory to better utilize storage capacity.
- Functions as an out-of-the-box data structure in a simple and efficient architecture without complex modules or dependencies.

### Deployment options

- Using [Tair Cloud Vector Database](https://www.alibabacloud.com/help/en/tair/latest/getting-started-overview). [Click here](https://www.alibabacloud.com/product/tair) to fast deploy it.


## Prerequisites

For the purposes of this exercise we need to prepare a couple of things:

1. Tair cloud server instance.
2. The 'tair' library to interact with the tair database.
3. An [OpenAI API key](https://beta.openai.com/account/api-keys).



### Install requirements

This notebook obviously requires the `openai` and `tair` packages, but there are also some other additional libraries we will use. The following command installs them all:



```python
! pip install openai redis tair pandas wget
```

    Looking in indexes: http://sg.mirrors.cloud.aliyuncs.com/pypi/simple/
    Requirement already satisfied: openai in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (0.28.0)
    Requirement already satisfied: redis in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (5.0.0)
    Requirement already satisfied: tair in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (1.3.6)
    Requirement already satisfied: pandas in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (2.1.0)
    Requirement already satisfied: wget in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (3.2)
    Requirement already satisfied: requests>=2.20 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from openai) (2.31.0)
    Requirement already satisfied: tqdm in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from openai) (4.66.1)
    Requirement already satisfied: aiohttp in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from openai) (3.8.5)
    Requirement already satisfied: async-timeout>=4.0.2 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from redis) (4.0.3)
    Requirement already satisfied: numpy>=1.22.4 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from pandas) (1.25.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from pandas) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from pandas) (2023.3)
    Requirement already satisfied: six>=1.5 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from requests>=2.20->openai) (3.2.0)
    Requirement already satisfied: idna<4,>=2.5 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from requests>=2.20->openai) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from requests>=2.20->openai) (2.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from requests>=2.20->openai) (2023.7.22)
    Requirement already satisfied: attrs>=17.3.0 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from aiohttp->openai) (22.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from aiohttp->openai) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from aiohttp->openai) (1.9.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from aiohttp->openai) (1.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /root/anaconda3/envs/notebook/lib/python3.10/site-packages (from aiohttp->openai) (1.3.1)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

### Prepare your OpenAI API key

The OpenAI API key is used for vectorization of the documents and queries.

If you don't have an OpenAI API key, you can get one from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys).

Once you get your key, please add it by getpass.


```python
import getpass
import openai

openai.api_key = getpass.getpass("Input your OpenAI API key:")
```

    Input your OpenAI API key:········
    

## Connect to Tair
First add it to your environment variables.

Connecting to a running instance of Tair server is easy with the official Python library.


```python
# The format of url: redis://[[username]:[password]]@localhost:6379/0
TAIR_URL = getpass.getpass("Input your tair url:")
```

    Input your tair url:········
    


```python
from tair import Tair as TairClient

# connect to tair from url and create a client

url = TAIR_URL
client = TairClient.from_url(url)
```

We can test the connection by ping:


```python
client.ping()
```




    True




```python
import wget

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)
```

    100% [......................................................................] 698933052 / 698933052




    'vector_database_wikipedia_articles_embedded (1).zip'



The downloaded file has to then be extracted:


```python
import zipfile
import os
import re
import tempfile

current_directory = os.getcwd()
zip_file_path = os.path.join(current_directory, "vector_database_wikipedia_articles_embedded.zip")
output_directory = os.path.join(current_directory, "../../data")

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(output_directory)


# check the csv file exist
file_name = "vector_database_wikipedia_articles_embedded.csv"
data_directory = os.path.join(current_directory, "../../data")
file_path = os.path.join(data_directory, file_name)


if os.path.exists(file_path):
    print(f"The file {file_name} exists in the data directory.")
else:
    print(f"The file {file_name} does not exist in the data directory.")

```

    The file vector_database_wikipedia_articles_embedded.csv exists in the data directory.
    

## Create Index

Tair stores data in indexes where each object is described by one key. Each key contains a vector and multiple attribute_keys.

We will start with creating two indexes, one for **title_vector** and one for **content_vector**, and then we will fill it with our precomputed embeddings.


```python
# set index parameters
index = "openai_test"
embedding_dim = 1536
distance_type = "L2"
index_type = "HNSW"
data_type = "FLOAT32"

# Create two indexes, one for title_vector and one for content_vector, skip if already exists
index_names = [index + "_title_vector", index+"_content_vector"]
for index_name in index_names:
    index_connection = client.tvs_get_index(index_name)
    if index_connection is not None:
        print("Index already exists")
    else:
        client.tvs_create_index(name=index_name, dim=embedding_dim, distance_type=distance_type,
                                index_type=index_type, data_type=data_type)
```

    Index already exists
    Index already exists
    

## Load data

In this section we are going to load the data prepared previous to this session, so you don't have to recompute the embeddings of Wikipedia articles with your own credits.


```python
import pandas as pd
from ast import literal_eval
# Path to your local CSV file
csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'
article_df = pd.read_csv(csv_file_path)

# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval).values
article_df['content_vector'] = article_df.content_vector.apply(literal_eval).values

# add/update data to indexes
for i in range(len(article_df)):
    # add data to index with title_vector
    client.tvs_hset(index=index_names[0], key=article_df.id[i].item(), vector=article_df.title_vector[i], is_binary=False,
                    **{"url": article_df.url[i], "title": article_df.title[i], "text": article_df.text[i]})
    # add data to index with content_vector
    client.tvs_hset(index=index_names[1], key=article_df.id[i].item(), vector=article_df.content_vector[i], is_binary=False,
                    **{"url": article_df.url[i], "title": article_df.title[i], "text": article_df.text[i]})
```


```python
# Check the data count to make sure all the points have been stored
for index_name in index_names:
    stats = client.tvs_get_index(index_name)
    count = int(stats["current_record_count"]) - int(stats["delete_record_count"])
    print(f"Count in {index_name}:{count}")

```

    Count in openai_test_title_vector:25000
    Count in openai_test_content_vector:25000
    

## Search data

Once the data is put into Tair we will start querying the collection for the closest vectors. We may provide an additional parameter `vector_name` to switch from title to content based search. Since the precomputed embeddings were created with `text-embedding-3-small` OpenAI model, we also have to use it during search.



```python
def query_tair(client, query, vector_name="title_vector", top_k=5):

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input= query,
        model="text-embedding-3-small",
    )["data"][0]['embedding']
    embedded_query = np.array(embedded_query)

    # search for the top k approximate nearest neighbors of vector in an index
    query_result = client.tvs_knnsearch(index=index+"_"+vector_name, k=top_k, vector=embedded_query)

    return query_result
```


```python
import openai
import numpy as np

query_result = query_tair(client=client, query="modern art in Europe", vector_name="title_vector")
for i in range(len(query_result)):
    title = client.tvs_hmget(index+"_"+"content_vector", query_result[i][0].decode('utf-8'), "title")
    print(f"{i + 1}. {title[0].decode('utf-8')} (Distance: {round(query_result[i][1],3)})")
```

    1. Museum of Modern Art (Distance: 0.125)
    2. Western Europe (Distance: 0.133)
    3. Renaissance art (Distance: 0.136)
    4. Pop art (Distance: 0.14)
    5. Northern Europe (Distance: 0.145)
    


```python
# This time we'll query using content vector
query_result = query_tair(client=client, query="Famous battles in Scottish history", vector_name="content_vector")
for i in range(len(query_result)):
    title = client.tvs_hmget(index+"_"+"content_vector", query_result[i][0].decode('utf-8'), "title")
    print(f"{i + 1}. {title[0].decode('utf-8')} (Distance: {round(query_result[i][1],3)})")
```

    1. Battle of Bannockburn (Distance: 0.131)
    2. Wars of Scottish Independence (Distance: 0.139)
    3. 1651 (Distance: 0.147)
    4. First War of Scottish Independence (Distance: 0.15)
    5. Robert I of Scotland (Distance: 0.154)
    


```python

```
