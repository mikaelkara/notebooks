# Getting Started with Zilliz and OpenAI
### Finding your next book

In this notebook we will be going over generating embeddings of book descriptions with OpenAI and using those embeddings within Zilliz to find relevant books. The dataset in this example is sourced from HuggingFace datasets, and contains a little over 1 million title-description pairs.

Lets begin by first downloading the required libraries for this notebook:
- `openai` is used for communicating with the OpenAI embedding service
- `pymilvus` is used for communicating with the Zilliz instance
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
    Requirement already satisfied: requests>=2.20 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from openai) (2.28.2)
    Requirement already satisfied: aiohttp in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from openai) (3.8.4)
    Requirement already satisfied: ujson<=5.4.0,>=2.0.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (5.1.0)
    Requirement already satisfied: grpcio-tools<=1.48.0,>=1.47.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (1.47.2)
    Requirement already satisfied: grpcio<=1.48.0,>=1.47.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (1.47.2)
    Requirement already satisfied: mmh3<=3.0.0,>=2.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (3.0.0)
    Requirement already satisfied: pandas>=1.2.4 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pymilvus) (1.5.3)
    Requirement already satisfied: numpy>=1.17 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (1.23.5)
    Requirement already satisfied: xxhash in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (3.2.0)
    Requirement already satisfied: responses<0.19 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.18.0)
    Requirement already satisfied: dill<0.3.7,>=0.3.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.3.6)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.2.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.12.1)
    Requirement already satisfied: pyarrow>=6.0.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (10.0.1)
    Requirement already satisfied: multiprocess in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (0.70.14)
    Requirement already satisfied: pyyaml>=5.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (5.4.1)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (2023.1.0)
    Requirement already satisfied: packaging in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from datasets) (23.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (1.3.3)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (4.0.2)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (22.2.0)
    Requirement already satisfied: charset-normalizer<4.0,>=2.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (3.0.1)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (1.8.2)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from aiohttp->openai) (6.0.4)
    Requirement already satisfied: six>=1.5.2 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from grpcio<=1.48.0,>=1.47.0->pymilvus) (1.16.0)
    Requirement already satisfied: protobuf<4.0dev,>=3.12.0 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from grpcio-tools<=1.48.0,>=1.47.0->pymilvus) (3.20.1)
    Requirement already satisfied: setuptools in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from grpcio-tools<=1.48.0,>=1.47.0->pymilvus) (65.6.3)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (4.5.0)
    Requirement already satisfied: filelock in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from huggingface-hub<1.0.0,>=0.2.0->datasets) (3.9.0)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pandas>=1.2.4->pymilvus) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from pandas>=1.2.4->pymilvus) (2022.7.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from requests>=2.20->openai) (1.26.14)
    Requirement already satisfied: idna<4,>=2.5 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from requests>=2.20->openai) (3.4)
    Requirement already satisfied: certifi>=2017.4.17 in /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages (from requests>=2.20->openai) (2022.12.7)
    

To get Zilliz up and running take a look [here](https://zilliz.com/doc/quick_start). With your account and database set up, proceed to set the following values:
- URI: The URI your database is running on
- USER: Your database username
- PASSWORD: Your database password
- COLLECTION_NAME: What to name the collection within Zilliz
- DIMENSION: The dimension of the embeddings
- OPENAI_ENGINE: Which embedding model to use
- openai.api_key: Your OpenAI account key
- INDEX_PARAM: The index settings to use for the collection
- QUERY_PARAM: The search parameters to use
- BATCH_SIZE: How many texts to embed and insert at once


```python
import openai

URI = 'your_uri'
TOKEN = 'your_token' # TOKEN == user:password or api_key
COLLECTION_NAME = 'book_search'
DIMENSION = 1536
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your-key'

INDEX_PARAM = {
    'metric_type':'L2',
    'index_type':"AUTOINDEX",
    'params':{}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {},
}

BATCH_SIZE = 1000
```

## Zilliz
This segment deals with Zilliz and setting up the database for this use case. Within Zilliz we need to setup a collection and index it.


```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# Connect to Zilliz Database
connections.connect(uri=URI, token=TOKEN)
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
With Zilliz up and running we can begin grabbing our data. `Hugging Face Datasets` is a hub that holds many different user datasets, and for this example we are using Skelebor's book dataset. This dataset contains title-description pairs for over 1 million books. We are going to embed each description and store it within Zilliz along with its title.


```python
import datasets

# Download the dataset and only use the `train` portion (file is around 800Mb)
dataset = datasets.load_dataset('Skelebor/book_titles_and_descriptions_en_clean', split='train')
```

    /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Found cached dataset parquet (/Users/filiphaltmayer/.cache/huggingface/datasets/Skelebor___parquet/Skelebor--book_titles_and_descriptions_en_clean-3596935b1d8a7747/0.0.0/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec)
    

## Insert the Data
Now that we have our data on our machine we can begin embedding it and inserting it into Zilliz. The embedding function takes in text and returns the embeddings in a list format.


```python
# Simple function that converts the texts to embeddings
def embed(texts):
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    return [x['embedding'] for x in embeddings['data']]

```

This next step does the actual inserting. Due to having so many datapoints, if you want to immediately test it out you can stop the inserting cell block early and move along. Doing this will probably decrease the accuracy of the results due to less datapoints, but it should still be good enough.


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

      0%|          | 2999/1032335 [00:19<1:49:30, 156.66it/s]
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[10], line 14
         12     if len(data[0]) % BATCH_SIZE == 0:
         13         data.append(embed(data[1]))
    ---> 14         collection.insert(data)
         15         data = [[],[]]
         17 # Embed and insert the remainder 
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/orm/collection.py:430, in Collection.insert(self, data, partition_name, timeout, **kwargs)
        427 entities = Prepare.prepare_insert_data(data, self._schema)
        429 conn = self._get_connection()
    --> 430 res = conn.batch_insert(self._name, entities, partition_name,
        431                         timeout=timeout, schema=self._schema_dict, **kwargs)
        433 if kwargs.get("_async", False):
        434     return MutationFuture(res)
    

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
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/decorators.py:50, in retry_on_rpc_failure.<locals>.wrapper.<locals>.handler(self, *args, **kwargs)
         48 while True:
         49     try:
    ---> 50         return func(self, *args, **kwargs)
         51     except grpc.RpcError as e:
         52         # DEADLINE_EXCEEDED means that the task wat not completed
         53         # UNAVAILABLE means that the service is not reachable currently
         54         # Reference: https://grpc.github.io/grpc/python/grpc.html#grpc-status-code
         55         if e.code() != grpc.StatusCode.DEADLINE_EXCEEDED and e.code() != grpc.StatusCode.UNAVAILABLE:
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/pymilvus/client/grpc_handler.py:378, in GrpcHandler.batch_insert(self, collection_name, entities, partition_name, timeout, **kwargs)
        375     f.add_callback(ts_utils.update_ts_on_mutation(collection_name))
        376     return f
    --> 378 response = rf.result()
        379 if response.status.error_code == 0:
        380     m = MutationResult(response)
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/grpc/_channel.py:733, in _MultiThreadedRendezvous.result(self, timeout)
        728 """Returns the result of the computation or raises its exception.
        729 
        730 See grpc.Future.result for the full API contract.
        731 """
        732 with self._state.condition:
    --> 733     timed_out = _common.wait(self._state.condition.wait,
        734                              self._is_complete,
        735                              timeout=timeout)
        736     if timed_out:
        737         raise grpc.FutureTimeoutError()
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/grpc/_common.py:141, in wait(wait_fn, wait_complete_fn, timeout, spin_cb)
        139 if timeout is None:
        140     while not wait_complete_fn():
    --> 141         _wait_once(wait_fn, MAXIMUM_WAIT_TIMEOUT, spin_cb)
        142 else:
        143     end = time.time() + timeout
    

    File ~/miniconda3/envs/haystack/lib/python3.9/site-packages/grpc/_common.py:106, in _wait_once(wait_fn, timeout, spin_cb)
        105 def _wait_once(wait_fn, timeout, spin_cb):
    --> 106     wait_fn(timeout=timeout)
        107     if spin_cb is not None:
        108         spin_cb()
    

    File ~/miniconda3/envs/haystack/lib/python3.9/threading.py:316, in Condition.wait(self, timeout)
        314 else:
        315     if timeout > 0:
    --> 316         gotit = waiter.acquire(True, timeout)
        317     else:
        318         gotit = waiter.acquire(False)
    

    KeyboardInterrupt: 


## Query the Database
With our data safely inserted in Zilliz, we can now perform a query. The query takes in a string or a list of strings and searches them. The results print out your provided description and the results that include the result score, the result title, and the result book description.


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

    Description: Book about a k-9 from europe
    Results:
    	Rank: 1 Score: 0.3047754764556885 Title: Bark M For Murder
    Who let the dogs out? Evildoers beware! Four of mystery fiction's top storytellers are
    setting the hounds on your trail -- in an incomparable quartet of crime stories with a
    canine edge. Man's (and woman's) best friends take the lead in this phenomenal
    collection of tales tense and surprising, humorous and thrilling: New York
    Timesbestselling author J.A. Jance's spellbinding saga of a scam-busting septuagenarian
    and her two golden retrievers; Anthony Award winner Virginia Lanier's pureblood thriller
    featuring bloodhounds and bloody murder; Chassie West's suspenseful stunner about a
    life-saving German shepherd and a ghastly forgotten crime; rising star Lee Charles
    Kelley's edge-of-your-seat yarn that pits an ex-cop/kennel owner and a yappy toy poodle
    against a craven killer.
    
    	Rank: 2 Score: 0.3283390402793884 Title: Texas K-9 Unit Christmas: Holiday Hero\Rescuing Christmas
    CHRISTMAS COMES WRAPPED IN DANGER Holiday Hero by Shirlee McCoy Emma Fairchild never
    expected to find trouble in sleepy Sagebrush, Texas. But when she's attacked and left
    for dead in her own diner, her childhood friend turned K-9 cop Lucas Harwood offers a
    chance at justice--and love. Rescuing Christmas by Terri Reed She escaped a kidnapper,
    but now a killer has set his sights on K-9 dog trainer Lily Anderson. When fellow
    officer Jarrod Evans appoints himself her bodyguard, Lily knows more than her life is at
    risk--so is her heart. Texas K-9 Unit: These lawmen solve the toughest cases with the
    help of their brave canine partners
    
    	Rank: 3 Score: 0.33899369835853577 Title: Dogs on Duty: Soldiers' Best Friends on the Battlefield and Beyond
    When the news of the raid on Osama Bin Laden's compound broke, the SEAL team member that
    stole the show was a highly trained canine companion. Throughout history, dogs have been
    key contributors to military units. Dorothy Hinshaw Patent follows man's best friend
    onto the battlefield, showing readers why dogs are uniquely qualified for the job at
    hand, how they are trained, how they contribute to missions, and what happens when they
    retire. With full-color photographs throughout and sidebars featuring heroic canines
    throughout history, Dogs on Duty provides a fascinating look at these exceptional
    soldiers and companions.
    
    	Rank: 4 Score: 0.34207457304000854 Title: Toute Allure: Falling in Love in Rural France
    After saying goodbye to life as a successful fashion editor in London, Karen Wheeler is
    now happy in her small village house in rural France. Her idyll is complete when she
    meets the love of her life - he has shaggy hair, four paws and a wet nose!
    
    	Rank: 5 Score: 0.343595951795578 Title: Otherwise Alone (Evan Arden, #1)
    Librarian's note: This is an alternate cover edition for ASIN: B00AP5NNWC. Lieutenant
    Evan Arden sits in a shack in the middle of nowhere, waiting for orders that will send
    him back home - if he ever gets them. Other than his loyal Great Pyrenees, there's no
    one around to break up the monotony. The tedium is excruciating, but it is suddenly
    interrupted when a young woman stumbles up his path. "It's only 50-something pages, but
    in that short amount of time, the author's awesome writing packs in a whole lotta
    character detail. And sets the stage for the series, perfectly." -Maryse.net, 4.5 Stars
    He has two choices - pick her off from a distance with his trusty sniper-rifle, or dare
    let her approach his cabin and enter his life. Why not? It's been ages, and he is
    otherwise alone...
    
    
