# Filtered Search with Zilliz and OpenAI
### Finding your next movie

In this notebook we will be going over generating embeddings of movie descriptions with OpenAI and using those embeddings within Zilliz to find relevant movies. To narrow our search results and try something new, we are going to be using filtering to do metadata searches. The dataset in this example is sourced from HuggingFace datasets, and contains a little over 8 thousand movie entries.

Lets begin by first downloading the required libraries for this notebook:
- `openai` is used for communicating with the OpenAI embedding service
- `pymilvus` is used for communicating with the Zilliz server
- `datasets` is used for downloading the dataset
- `tqdm` is used for the progress bars



```python
! pip install openai pymilvus datasets tqdm
```

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
openai.api_key = 'sk-your_key'

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
    FieldSchema(name='type', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='release_year', dtype=DataType.INT64),
    FieldSchema(name='rating', dtype=DataType.VARCHAR, max_length=64000),
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
With Zilliz up and running we can begin grabbing our data. `Hugging Face Datasets` is a hub that holds many different user datasets, and for this example we are using HuggingLearners's netflix-shows dataset. This dataset contains movies and their metadata pairs for over 8 thousand movies. We are going to embed each description and store it within Zilliz along with its title, type, release_year and rating.


```python
import datasets

# Download the dataset 
dataset = datasets.load_dataset('hugginglearners/netflix-shows', split='train')
```

    /Users/filiphaltmayer/miniconda3/envs/haystack/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Found cached dataset csv (/Users/filiphaltmayer/.cache/huggingface/datasets/hugginglearners___csv/hugginglearners--netflix-shows-03475319fc65a05a/0.0.0/6b34fb8fcf56f7c8ba51dc895bfa2bfbe43546f190a60fcf74bb5e8afdcc2317)
    

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

This next step does the actual inserting. We iterate through all the entries and create batches that we insert once we hit our set batch size. After the loop is over we insert the last remaning batch if it exists. 


```python
from tqdm import tqdm

data = [
    [], # title
    [], # type
    [], # release_year
    [], # rating
    [], # description
]

# Embed and insert in batches
for i in tqdm(range(0, len(dataset))):
    data[0].append(dataset[i]['title'] or '')
    data[1].append(dataset[i]['type'] or '')
    data[2].append(dataset[i]['release_year'] or -1)
    data[3].append(dataset[i]['rating'] or '')
    data[4].append(dataset[i]['description'] or '')
    if len(data[0]) % BATCH_SIZE == 0:
        data.append(embed(data[4]))
        collection.insert(data)
        data = [[],[],[],[],[]]

# Embed and insert the remainder 
if len(data[0]) != 0:
    data.append(embed(data[4]))
    collection.insert(data)
    data = [[],[],[],[],[]]

```

    100%|██████████| 8807/8807 [00:54<00:00, 162.59it/s]
    

## Query the Database
With our data safely inserted into Zilliz, we can now perform a query. The query takes in a tuple of the movie description you are searching for and the filter to use. More info about the filter can be found [here](https://milvus.io/docs/boolean.md). The search first prints out your description and filter expression. After that for each result we print the score, title, type, release year, rating and description of the result movies. 


```python
import textwrap

def query(query, top_k = 5):
    text, expr = query
    res = collection.search(embed(text), anns_field='embedding', expr = expr, param=QUERY_PARAM, limit = top_k, output_fields=['title', 'type', 'release_year', 'rating', 'description'])
    for i, hit in enumerate(res):
        print('Description:', text, 'Expression:', expr)
        print('Results:')
        for ii, hits in enumerate(hit):
            print('\t' + 'Rank:', ii + 1, 'Score:', hits.score, 'Title:', hits.entity.get('title'))
            print('\t\t' + 'Type:', hits.entity.get('type'), 'Release Year:', hits.entity.get('release_year'), 'Rating:', hits.entity.get('rating'))
            print(textwrap.fill(hits.entity.get('description'), 88))
            print()

my_query = ('movie about a fluffly animal', 'release_year < 2019 and rating like \"PG%\"')

query(my_query)
```

    Description: movie about a fluffly animal Expression: release_year < 2019 and rating like "PG%"
    Results:
    	Rank: 1 Score: 0.30085673928260803 Title: The Lamb
    		Type: Movie Release Year: 2017 Rating: PG
    A big-dreaming donkey escapes his menial existence and befriends some free-spirited
    animal pals in this imaginative retelling of the Nativity Story.
    
    	Rank: 2 Score: 0.3352621793746948 Title: Puss in Boots
    		Type: Movie Release Year: 2011 Rating: PG
    The fabled feline heads to the Land of Giants with friends Humpty Dumpty and Kitty
    Softpaws on a quest to nab its greatest treasure: the Golden Goose.
    
    	Rank: 3 Score: 0.3415083587169647 Title: Show Dogs
    		Type: Movie Release Year: 2018 Rating: PG
    A rough and tough police dog must go undercover with an FBI agent as a prim and proper
    pet at a dog show to save a baby panda from an illegal sale.
    
    	Rank: 4 Score: 0.3428957462310791 Title: Open Season 2
    		Type: Movie Release Year: 2008 Rating: PG
    Elliot the buck and his forest-dwelling cohorts must rescue their dachshund pal from
    some spoiled pets bent on returning him to domesticity.
    
    	Rank: 5 Score: 0.34376364946365356 Title: Stuart Little 2
    		Type: Movie Release Year: 2002 Rating: PG
    Zany misadventures are in store as lovable city mouse Stuart and his human brother,
    George, raise the roof in this sequel to the 1999 blockbuster.
    
    
