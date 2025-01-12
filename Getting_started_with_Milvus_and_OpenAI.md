# Getting Started with Milvus and OpenAI
### Finding your next book

In this notebook we will be going over generating embeddings of book descriptions with OpenAI and using those embeddings within Milvus to find relevant books. The dataset in this example is sourced from HuggingFace datasets, and contains a little over 1 million title-description pairs.

Lets begin by first downloading the required libraries for this notebook:
- `openai` is used for communicating with the OpenAI embedding service
- `pymilvus` is used for communicating with the Milvus server
- `datasets` is used for downloading the dataset
- `tqdm` is used for the progress bars



```python
! pip install openai pymilvus datasets tqdm
```

    Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
    Requirement already satisfied: openai in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (0.27.2)
    Requirement already satisfied: pymilvus in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (2.2.2)
    Requirement already satisfied: datasets in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (2.10.1)
    Requirement already satisfied: tqdm in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (4.64.1)
    Requirement already satisfied: aiohttp in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from openai) (3.8.4)
    Requirement already satisfied: requests>=2.20 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from openai) (2.28.2)
    Requirement already satisfied: pandas>=1.2.4 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (1.5.3)
    Requirement already satisfied: ujson<=5.4.0,>=2.0.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (5.1.0)
    Requirement already satisfied: mmh3<=3.0.0,>=2.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (3.0.0)
    Requirement already satisfied: grpcio<=1.48.0,>=1.47.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (1.47.2)
    Requirement already satisfied: grpcio-tools<=1.48.0,>=1.47.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (1.47.2)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.2.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.12.1)
    Requirement already satisfied: dill<0.3.7,>=0.3.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.3.6)
    Requirement already satisfied: xxhash in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (3.2.0)
    Requirement already satisfied: pyyaml>=5.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (5.4.1)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (2023.1.0)
    Requirement already satisfied: packaging in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (23.0)
    Requirement already satisfied: numpy>=1.17 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (1.23.5)
    Requirement already satisfied: multiprocess in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.70.14)
    Requirement already satisfied: pyarrow>=6.0.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (10.0.1)
    Requirement already satisfied: responses<0.19 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.18.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (6.0.4)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (1.3.3)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (1.8.2)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (1.3.1)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (3.0.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (22.2.0)
    Requirement already satisfied: six>=1.5.2 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from grpcio<=1.48.0,>=1.47.0->pymilvus) (1.16.0)
    Requirement already satisfied: protobuf<4.0dev,>=3.12.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from grpcio-tools<=1.48.0,>=1.47.0->pymilvus) (3.20.1)
    Requirement already satisfied: setuptools in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from grpcio-tools<=1.48.0,>=1.47.0->pymilvus) (65.6.3)
    Requirement already satisfied: filelock in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (3.9.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (4.5.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pandas>=1.2.4->pymilvus) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pandas>=1.2.4->pymilvus) (2022.7.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from requests>=2.20->openai) (1.26.14)
    Requirement already satisfied: idna<4,>=2.5 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from requests>=2.20->openai) (3.4)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from requests>=2.20->openai) (2022.12.7)
    

With the required packages installed we can get started. Lets begin by launching the Milvus service. The file being run is the `docker-compose.yaml` found in the folder of this file. This command launches a Milvus standalone instance which we will use for this test.  


```python
! docker compose up -d
```

    [1A[1B[0G[?25l[+] Running 0/0
    [37m ⠋ Network milvus  Creating                                                0.1s
    [0m[?25h[1A[1A[0G[?25l[34m[+] Running 1/1[0m
    [34m ⠿ Network milvus          Created                                         0.1s
    [0m[37m ⠋ Container milvus-minio  Creating                                        0.1s
    [0m[37m ⠋ Container milvus-etcd   Creating                                        0.1s
    [0m[?25h[1A[1A[1A[1A[0G[?25l[+] Running 1/3
    [34m ⠿ Network milvus          Created                                         0.1s
    [0m[37m ⠙ Container milvus-minio  Creating                                        0.2s
    [0m[37m ⠙ Container milvus-etcd   Creating                                        0.2s
    [0m[?25h[1A[1A[1A[1A[0G[?25l[+] Running 1/3
    [34m ⠿ Network milvus          Created                                         0.1s
    [0m[37m ⠹ Container milvus-minio  Creating                                        0.3s
    [0m[37m ⠹ Container milvus-etcd   Creating                                        0.3s
    [0m[?25h[1A[1A[1A[1A[0G[?25l[34m[+] Running 3/3[0m
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Created                                    0.3s
    [0m[34m ⠿ Container milvus-etcd        Created                                    0.3s
    [0m[37m ⠋ Container milvus-standalone  Creating                                   0.1s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Created                                    0.3s
    [0m[34m ⠿ Container milvus-etcd        Created                                    0.3s
    [0m[37m ⠙ Container milvus-standalone  Creating                                   0.2s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[34m[+] Running 4/4[0m
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Created                                    0.3s
    [0m[34m ⠿ Container milvus-etcd        Created                                    0.3s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   0.7s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   0.7s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   0.8s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   0.8s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   0.9s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   0.9s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.0s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.0s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.1s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.1s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.2s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.2s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.3s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.3s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.4s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.4s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.5s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.5s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.6s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.6s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 2/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.7s
    [0m[37m ⠿ Container milvus-etcd        Starting                                   1.7s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[37m ⠿ Container milvus-minio       Starting                                   1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[34m ⠿ Container milvus-standalone  Created                                    0.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   1.6s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   1.7s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   1.8s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   1.9s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.0s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.1s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.2s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.3s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.4s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.5s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[+] Running 3/4
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[37m ⠿ Container milvus-standalone  Starting                                   2.6s
    [0m[?25h[1A[1A[1A[1A[1A[0G[?25l[34m[+] Running 4/4[0m
    [34m ⠿ Network milvus               Created                                    0.1s
    [0m[34m ⠿ Container milvus-minio       Started                                    1.8s
    [0m[34m ⠿ Container milvus-etcd        Started                                    1.7s
    [0m[34m ⠿ Container milvus-standalone  Started                                    2.6s
    [0m[?25h

With Milvus running we can setup our global variables:
- HOST: The Milvus host address
- PORT: The Milvus port number
- COLLECTION_NAME: What to name the collection within Milvus
- DIMENSION: The dimension of the embeddings
- OPENAI_ENGINE: Which embedding model to use
- openai.api_key: Your OpenAI account key
- INDEX_PARAM: The index settings to use for the collection
- QUERY_PARAM: The search parameters to use
- BATCH_SIZE: How many texts to embed and insert at once


```python
import openai

HOST = 'localhost'
PORT = 19530
COLLECTION_NAME = 'book_search'
DIMENSION = 1536
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your_key'

INDEX_PARAM = {
    'metric_type':'L2',
    'index_type':"HNSW",
    'params':{'M': 8, 'efConstruction': 64}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {"ef": 64},
}

BATCH_SIZE = 1000
```

## Milvus
This segment deals with Milvus and setting up the database for this use case. Within Milvus we need to setup a collection and index the collection. 


```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# Connect to Milvus Database
connections.connect(host=HOST, port=PORT)
```


```python
# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
```


```python
# Create collection which includes the id, title, and embedding.
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
```


```python
# Create the index on the collection and load it.
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()
```

## Dataset
With Milvus up and running we can begin grabbing our data. Hugging Face Datasets is a hub that holds many different user datasets, and for this example we are using Skelebor's book dataset. This dataset contains title-description pairs for over 1 million books. We are going to embed each description and store it within Milvus along with its title. 


```python
import datasets

# Download the dataset and only use the `train` portion (file is around 800Mb)
dataset = datasets.load_dataset('Skelebor/book_titles_and_descriptions_en_clean', split='train')
```

    /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Found cached dataset parquet (/Users/filiphaltmayer/.cache/huggingface/datasets/Skelebor___parquet/Skelebor--book_titles_and_descriptions_en_clean-3596935b1d8a7747/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec)
    

## Insert the Data
Now that we have our data on our machine we can begin embedding it and inserting it into Milvus. The embedding function takes in text and returns the embeddings in a list format. 


```python
# Simple function that converts the texts to embeddings
def embed(texts):
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    return [x['embedding'] for x in embeddings['data']]

```

This next step does the actual inserting. Due to having so many datapoints, if you want to immidiately test it out you can stop the inserting cell block early and move along. Doing this will probably decrease the accuracy of the results due to less datapoints, but it should still be good enough. 


```python
from tqdm import tqdm

data = [
    [], # title
    [], # description
]

# Embed and insert in batches
for i in tqdm(range(0, len(dataset))):
    data[0].append(dataset[i]['title'])
    data[1].append(dataset[i]['description'])
    if len(data[0]) % BATCH_SIZE == 0:
        data.append(embed(data[1]))
        collection.insert(data)
        data = [[],[]]

# Embed and insert the remainder 
if len(data[0]) != 0:
    data.append(embed(data[1]))
    collection.insert(data)
    data = [[],[]]

```

      0%|          | 1999/1032335 [00:06<57:22, 299.31it/s]  
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[18], line 13
         11 data[1].append(dataset[i]['description'])
         12 if len(data[0]) % BATCH_SIZE == 0:
    ---> 13     data.append(embed(data[1]))
         14     collection.insert(data)
         15     data = [[],[]]
    

    Cell In[17], line 3, in embed(texts)
          2 def embed(texts):
    ----> 3     embeddings = openai.Embedding.create(
          4         input=texts,
          5         engine=OPENAI_ENGINE
          6     )
          7     return [x['embedding'] for x in embeddings['data']]
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/openai/api_resources/embedding.py:33, in Embedding.create(cls, *args, **kwargs)
         31 while True:
         32     try:
    ---> 33         response = super().create(*args, **kwargs)
         35         # If a user specifies base64, we'll just return the encoded string.
         36         # This is only for the default case.
         37         if not user_provided_encoding_format:
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/openai/api_resources/abstract/engine_api_resource.py:153, in EngineAPIResource.create(cls, api_key, api_base, api_type, request_id, api_version, organization, **params)
        127 @classmethod
        128 def create(
        129     cls,
       (...)
        136     **params,
        137 ):
        138     (
        139         deployment_id,
        140         engine,
       (...)
        150         api_key, api_base, api_type, api_version, organization, **params
        151     )
    --> 153     response, _, api_key = requestor.request(
        154         "post",
        155         url,
        156         params=params,
        157         headers=headers,
        158         stream=stream,
        159         request_id=request_id,
        160         request_timeout=request_timeout,
        161     )
        163     if stream:
        164         # must be an iterator
        165         assert not isinstance(response, OpenAIResponse)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/openai/api_requestor.py:216, in APIRequestor.request(self, method, url, params, headers, files, stream, request_id, request_timeout)
        205 def request(
        206     self,
        207     method,
       (...)
        214     request_timeout: Optional[Union[float, Tuple[float, float]]] = None,
        215 ) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool, str]:
    --> 216     result = self.request_raw(
        217         method.lower(),
        218         url,
        219         params=params,
        220         supplied_headers=headers,
        221         files=files,
        222         stream=stream,
        223         request_id=request_id,
        224         request_timeout=request_timeout,
        225     )
        226     resp, got_stream = self._interpret_response(result, stream)
        227     return resp, got_stream, self.api_key
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/openai/api_requestor.py:516, in APIRequestor.request_raw(self, method, url, params, supplied_headers, files, stream, request_id, request_timeout)
        514     _thread_context.session = _make_session()
        515 try:
    --> 516     result = _thread_context.session.request(
        517         method,
        518         abs_url,
        519         headers=headers,
        520         data=data,
        521         files=files,
        522         stream=stream,
        523         timeout=request_timeout if request_timeout else TIMEOUT_SECS,
        524     )
        525 except requests.exceptions.Timeout as e:
        526     raise error.Timeout("Request timed out: {}".format(e)) from e
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/requests/sessions.py:587, in Session.request(self, method, url, params, data, headers, cookies, files, auth, timeout, allow_redirects, proxies, hooks, stream, verify, cert, json)
        582 send_kwargs = {
        583     "timeout": timeout,
        584     "allow_redirects": allow_redirects,
        585 }
        586 send_kwargs.update(settings)
    --> 587 resp = self.send(prep, **send_kwargs)
        589 return resp
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/requests/sessions.py:701, in Session.send(self, request, **kwargs)
        698 start = preferred_clock()
        700 # Send the request
    --> 701 r = adapter.send(request, **kwargs)
        703 # Total elapsed time of the request (approximately)
        704 elapsed = preferred_clock() - start
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/requests/adapters.py:489, in HTTPAdapter.send(self, request, stream, timeout, verify, cert, proxies)
        487 try:
        488     if not chunked:
    --> 489         resp = conn.urlopen(
        490             method=request.method,
        491             url=url,
        492             body=request.body,
        493             headers=request.headers,
        494             redirect=False,
        495             assert_same_host=False,
        496             preload_content=False,
        497             decode_content=False,
        498             retries=self.max_retries,
        499             timeout=timeout,
        500         )
        502     # Send the request.
        503     else:
        504         if hasattr(conn, "proxy_pool"):
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/urllib3/connectionpool.py:703, in HTTPConnectionPool.urlopen(self, method, url, body, headers, retries, redirect, assert_same_host, timeout, pool_timeout, release_conn, chunked, body_pos, **response_kw)
        700     self._prepare_proxy(conn)
        702 # Make the request on the httplib connection object.
    --> 703 httplib_response = self._make_request(
        704     conn,
        705     method,
        706     url,
        707     timeout=timeout_obj,
        708     body=body,
        709     headers=headers,
        710     chunked=chunked,
        711 )
        713 # If we're going to release the connection in ``finally:``, then
        714 # the response doesn't need to know about the connection. Otherwise
        715 # it will also try to release it and we'll have a double-release
        716 # mess.
        717 response_conn = conn if not release_conn else None
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/urllib3/connectionpool.py:449, in HTTPConnectionPool._make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        444             httplib_response = conn.getresponse()
        445         except BaseException as e:
        446             # Remove the TypeError from the exception chain in
        447             # Python 3 (including for exceptions like SystemExit).
        448             # Otherwise it looks like a bug in the code.
    --> 449             six.raise_from(e, None)
        450 except (SocketTimeout, BaseSSLError, SocketError) as e:
        451     self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
    

    File <string>:3, in raise_from(value, from_value)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/urllib3/connectionpool.py:444, in HTTPConnectionPool._make_request(self, conn, method, url, timeout, chunked, **httplib_request_kw)
        441 except TypeError:
        442     # Python 3
        443     try:
    --> 444         httplib_response = conn.getresponse()
        445     except BaseException as e:
        446         # Remove the TypeError from the exception chain in
        447         # Python 3 (including for exceptions like SystemExit).
        448         # Otherwise it looks like a bug in the code.
        449         six.raise_from(e, None)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/http/client.py:1377, in HTTPConnection.getresponse(self)
       1375 try:
       1376     try:
    -> 1377         response.begin()
       1378     except ConnectionError:
       1379         self.close()
    

    File ~/miniconda3/envs/haystack/lib/python3.9/http/client.py:320, in HTTPResponse.begin(self)
        318 # read until we get a non-100 response
        319 while True:
    --> 320     version, status, reason = self._read_status()
        321     if status != CONTINUE:
        322         break
    

    File ~/miniconda3/envs/haystack/lib/python3.9/http/client.py:281, in HTTPResponse._read_status(self)
        280 def _read_status(self):
    --> 281     line = str(self.fp.readline(_MAXLINE + 1), "iso-8859-1")
        282     if len(line) > _MAXLINE:
        283         raise LineTooLong("status line")
    

    File ~/miniconda3/envs/haystack/lib/python3.9/socket.py:704, in SocketIO.readinto(self, b)
        702 while True:
        703     try:
    --> 704         return self._sock.recv_into(b)
        705     except timeout:
        706         self._timeout_occurred = True
    

    File ~/miniconda3/envs/haystack/lib/python3.9/ssl.py:1242, in SSLSocket.recv_into(self, buffer, nbytes, flags)
       1238     if flags != 0:
       1239         raise ValueError(
       1240           "non-zero flags not allowed in calls to recv_into() on %s" %
       1241           self.__class__)
    -> 1242     return self.read(nbytes, buffer)
       1243 else:
       1244     return super().recv_into(buffer, nbytes, flags)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/ssl.py:1100, in SSLSocket.read(self, len, buffer)
       1098 try:
       1099     if buffer is not None:
    -> 1100         return self._sslobj.read(len, buffer)
       1101     else:
       1102         return self._sslobj.read(len)
    

    KeyboardInterrupt: 


## Query the Database
With our data safely inserted in Milvus, we can now perform a query. The query takes in a string or a list of strings and searches them. The resuts print out your provided description and the results that include the result score, the result title, and the result book description. 


```python
import textwrap

def query(queries, top_k = 5):
    if type(queries) != list:
        queries = [queries]
    res = collection.search(embed(queries), anns_field='embedding', param=QUERY_PARAM, limit = top_k, output_fields=['title', 'description'])
    for i, hit in enumerate(res):
        print('Description:', queries[i])
        print('Results:')
        for ii, hits in enumerate(hit):
            print('\t' + 'Rank:', ii + 1, 'Score:', hits.score, 'Title:', hits.entity.get('title'))
            print(textwrap.fill(hits.entity.get('description'), 88))
            print()
```


```python
query('Book about a k-9 from europe')
```

    RPC error: [search], <MilvusException: (code=1, message=code: UnexpectedError, reason: code: CollectionNotExists, reason: can't find collection: book_search)>, <Time:{'RPC start': '2023-03-17 14:22:18.368461', 'RPC error': '2023-03-17 14:22:18.382086'}>
    


    ---------------------------------------------------------------------------

    MilvusException                           Traceback (most recent call last)

    Cell In[32], line 1
    ----> 1 query('Book about a k-9 from europe')
    

    Cell In[31], line 6, in query(queries, top_k)
          4 if type(queries) != list:
          5     queries = [queries]
    ----> 6 res = collection.search(embed(queries), anns_field='embedding', param=QUERY_PARAM, limit = top_k, output_fields=['title', 'description'])
          7 for i, hit in enumerate(res):
          8     print('Description:', queries[i])
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/orm/collection.py:614, in Collection.search(self, data, anns_field, param, limit, expr, partition_names, output_fields, timeout, round_decimal, **kwargs)
        611     raise DataTypeNotMatchException(message=ExceptionsMessage.ExprType % type(expr))
        613 conn = self._get_connection()
    --> 614 res = conn.search(self._name, data, anns_field, param, limit, expr,
        615                   partition_names, output_fields, round_decimal, timeout=timeout,
        616                   schema=self._schema_dict, **kwargs)
        617 if kwargs.get("_async", False):
        618     return SearchFuture(res)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/decorators.py:109, in error_handler.<locals>.wrapper.<locals>.handler(*args, **kwargs)
        107     record_dict["RPC error"] = str(datetime.datetime.now())
        108     LOGGER.error(f"RPC error: [{inner_name}], {e}, <Time:{record_dict}>")
    --> 109     raise e
        110 except grpc.FutureTimeoutError as e:
        111     record_dict["gRPC timeout"] = str(datetime.datetime.now())
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/decorators.py:105, in error_handler.<locals>.wrapper.<locals>.handler(*args, **kwargs)
        103 try:
        104     record_dict["RPC start"] = str(datetime.datetime.now())
    --> 105     return func(*args, **kwargs)
        106 except MilvusException as e:
        107     record_dict["RPC error"] = str(datetime.datetime.now())
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/decorators.py:136, in tracing_request.<locals>.wrapper.<locals>.handler(self, *args, **kwargs)
        134 if req_id:
        135     self.set_onetime_request_id(req_id)
    --> 136 ret = func(self, *args, **kwargs)
        137 return ret
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/decorators.py:85, in retry_on_rpc_failure.<locals>.wrapper.<locals>.handler(self, *args, **kwargs)
         83         back_off = min(back_off * back_off_multiplier, max_back_off)
         84     else:
    ---> 85         raise e
         86 except Exception as e:
         87     raise e
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/decorators.py:50, in retry_on_rpc_failure.<locals>.wrapper.<locals>.handler(self, *args, **kwargs)
         48 while True:
         49     try:
    ---> 50         return func(self, *args, **kwargs)
         51     except grpc.RpcError as e:
         52         # DEADLINE_EXCEEDED means that the task wat not completed
         53         # UNAVAILABLE means that the service is not reachable currently
         54         # Reference: https://grpc.github.io/grpc/python/grpc.html#grpc-status-code
         55         if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED and e.code() != grpc.StatusCode.UNAVAILABLE:
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/client/grpc_handler.py:472, in GrpcHandler.search(self, collection_name, data, anns_field, param, limit, expression, partition_names, output_fields, round_decimal, timeout, schema, **kwargs)
        467 requests = Prepare.search_requests_with_expr(collection_name, data, anns_field, param, limit, schema,
        468                                              expression, partition_names, output_fields, round_decimal,
        469                                              **kwargs)
        471 auto_id = schema["auto_id"]
    --> 472 return self._execute_search_requests(requests, timeout, round_decimal=round_decimal, auto_id=auto_id, **kwargs)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/client/grpc_handler.py:441, in GrpcHandler._execute_search_requests(self, requests, timeout, **kwargs)
        439 if kwargs.get("_async", False):
        440     return SearchFuture(None, None, True, pre_err)
    --> 441 raise pre_err
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/client/grpc_handler.py:432, in GrpcHandler._execute_search_requests(self, requests, timeout, **kwargs)
        429     response = self._stub.Search(request, timeout=timeout)
        431     if response.status.error_code != 0:
    --> 432         raise MilvusException(response.status.error_code, response.status.reason)
        434     raws.append(response)
        435 round_decimal = kwargs.get("round_decimal", -1)
    

    MilvusException: <MilvusException: (code=1, message=code: UnexpectedError, reason: code: CollectionNotExists, reason: can't find collection: book_search)>

