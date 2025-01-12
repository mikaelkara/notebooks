<a target="_blank" href="https://colab.research.google.com/drive/1bRBU30c42fzSyN4FahY59bEq9wuMq0Az?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Movie recommender example with Fireworks + MongoDB + Nomic embedding model

## Introduction
In this tutorial, we'll explore how to create a basic movie recommendation system. We'll leverage the Fireworks API for embedding generation, MongoDB for data storage and retrieval, and the Nomic-AI embedding model for nuanced understanding of movie data.

## Setting Up Your Environment
Before we dive into the code, make sure to set up your environment. This involves installing necessary packages like pymongo and openai. Run the following command in your notebook to install these packages:


```python
!pip install -q pymongo fireworks-ai tqdm openai
```

    [?25l     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/676.9 kB[0m [31m?[0m eta [36m-:--:--[0m[2K     [91m━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m112.6/676.9 kB[0m [31m3.2 MB/s[0m eta [36m0:00:01[0m[2K     [91m━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━[0m [32m450.6/676.9 kB[0m [31m6.5 MB/s[0m eta [36m0:00:01[0m[2K     [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m675.8/676.9 kB[0m [31m7.3 MB/s[0m eta [36m0:00:01[0m[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m676.9/676.9 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m80.5/80.5 kB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m267.1/267.1 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m307.7/307.7 kB[0m [31m9.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m75.6/75.6 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.9/77.9 kB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.3/58.3 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [?25h

## Initializing Fireworks and MongoDB Clients
To interact with Fireworks and MongoDB, we need to initialize their respective clients. Replace "YOUR_FIREWORKS_API_KEY" and "YOUR_MONGO_URL" with your actual credentials.


Please create a Mongodb Atlas cluster using the link [here](https://www.mongodb.com/atlas).

```
Note:

1. You should create a create a user name, password pair and fill those details in the URI below. MongoDB URI would like `mongodb+srv://<username>:<password>@<mongodb_cluster_unique_identifier>`.   
2. In MongoDB Atlas, you can only connect to a cluster from a trusted IP address. You must add your IP address to the IP access list before you can connect to your cluster. OR open access to public internet by configuring `0.0.0.0/0`.

```


```python
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "YOUR_MONGO_URL" # you can copy uri from MongoDB Atlas Cloud Console https://cloud.mongodb.com

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
```


```python
import openai
fw_client = openai.OpenAI(
  api_key="YOUR_FIREWORKS_API_KEY", # you can find Fireworks API key under accounts -> API keys
  base_url="https://api.fireworks.ai/inference/v1"
)
```

## Indexing and retrieval for movies.
We are going to build a model to index and retrieve movie recommendations. We will setup the most basic RAG example on top of MongoDB which involves
- MongoDB Atlas database that indexes movies based on embeddings
- a system for document embedding generation. We'll use the Nomic-AI model to create embeddings from text data. The function generate_embeddings takes a list of texts and returns embeddings.
- a basic search engine that responds to user query by embedding the user query, fetching the corresponding movies, and then use an LLM to generate the recommendations.

## Understanding the Nomic-ai 1.5 Model

The Nomic AI model, specifically the `nomic-ai/nomic-embed-text-v1.5` variant, is a great open source model embedding model. It has other features such as dimensionality reduction, but needs some special prefixes to be used properly, which we can get into in the next section


```python
from typing import List

def generate_embeddings(input_texts: str, model_api_string: str, prefix="") -> List[float]:
    """Generate embeddings from Fireworks python library

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.
        prefix: what prefix to attach to the generate the embeddings, which is required for nomic 1.5. Please check out https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage for more information

    Returns:
        reduced_embeddings_list: a list of reduced-size embeddings. Each element corresponds to each input text.
    """
    if prefix:
        input_texts = [prefix + text for text in input_texts]
    return fw_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    ).data[0].embedding
```

In the function above, we did not implement batching and always return the embedding at position zero. For how to do batching, we will cover it in the next tutorial.

## Data Processing
Now, let's process our movie data. We'll extract key information from our MongoDB collection and generate embeddings for each movie. Ensure NUM_DOC_LIMIT is set to limit the number of documents processed.


```python
embedding_model_string = 'nomic-ai/nomic-embed-text-v1.5'
vector_database_field_name = 'embed' # define your embedding field name.
NUM_DOC_LIMIT = 2000 # the number of documents you will process and generate embeddings.

sample_output = generate_embeddings(["This is a test."], embedding_model_string)
print(f"Embedding size is: {str(len(sample_output))}")

```

    Embedding size is: 768
    


```python
from tqdm import tqdm
from datetime import datetime

db = client.sample_mflix # loading sample dataset from MongoDB Atlas
collection = db.movies

keys_to_extract = ["plot", "genre", "cast", "title", "fullplot", "countries", "directors"]
for doc in tqdm(collection.find(
  {
    "fullplot":{"$exists": True},
    "released": { "$gt": datetime(2000, 1, 1, 0, 0, 0)},
  }
).limit(NUM_DOC_LIMIT), desc="Document Processing "):
  extracted_str = "\n".join([k + ": " + str(doc[k]) for k in keys_to_extract if k in doc])
  if vector_database_field_name not in doc:
    doc[vector_database_field_name] = generate_embeddings([extracted_str], embedding_model_string, "search_document: ")
  collection.replace_one({'_id': doc['_id']}, doc)

```

    Document Processing : 0it [00:01, ?it/s]
    

## Setting Up the Search Index
For our system to efficiently search through movie embeddings, we need to set up a search index in MongoDB. Please run the below cell to define the index structure as shown:



```python
"""
{
  "fields": [
    {
      "type": "vector",
      "path": "embed",
      "numDimensions": 768,
      "similarity": "dotProduct"
    }
  ]
}

"""
```




    '\n{\n  "fields": [\n    {\n      "type": "vector",\n      "path": "embed",\n      "numDimensions": 768,\n      "similarity": "dotProduct"\n    }\n  ]\n}\n\n'



## Querying the Recommender System
Let's test our recommender system. We create a query for superhero movies and exclude Spider-Man movies, as per user preference.


```python
# Example query.
query = "I like Christmas movies, any recommendations?"
prefix="search_query: "
query_emb = generate_embeddings([query], embedding_model_string, prefix=prefix)

results = collection.aggregate([
  {
    "$vectorSearch": {
      "queryVector": query_emb,
      "path": vector_database_field_name,
      "numCandidates": 100, # this should be 10-20x the limit
      "limit": 10, # the number of documents to return in the results
      "index": 'movie', # the index name you used in the earlier step
    }
  }
])
results_as_dict = {doc['title']: doc for doc in results}

print(f"From your query \"{query}\", the following movie listings were found:\n")
print("\n".join([str(i+1) + ". " + name for (i, name) in enumerate(results_as_dict.keys())]))

```

    From your query "I like Christmas movies, any recommendations?", the following movie listings were found:
    
    
    

## Generating Recommendations
Finally, we use Fireworks' chat API to generate a personalized movie recommendation based on the user's query and preferences.




```python
your_task_prompt = (
    "From the given movie listing data, choose a few great movie recommendation given the user query. "
    f"User query: {query}"
)

listing_data = ""
for doc in results_as_dict.values():
  listing_data += f"Movie title: {doc['title']}\n"
  for (k, v) in doc.items():
    if not(k in keys_to_extract) or ("embedding" in k): continue
    if k == "name": continue
    listing_data += k + ": " + str(v) + "\n"
  listing_data += "\n"

augmented_prompt = (
    "movie listing data:\n"
    f"{listing_data}\n\n"
    f"{your_task_prompt}"
)

```


```python
response = fw_client.chat.completions.create(
  messages=[{"role": "user", "content": augmented_prompt}],
  model="accounts/fireworks/models/mixtral-8x7b-instruct",
)

print(response.choices[0].message.content)

```

    Based on the movie listing data you provided, here are some great Christmas movie recommendations for the user:
    
    1. "Elf" (2003) - A comedic and heartwarming story about a man raised as an elf at the North Pole who travels to New York to find his biological father.
    2. "National Lampoon's Christmas Vacation" (1989) - A classic comedy about a family's chaotic and hilarious holiday season.
    3. "The Polar Express" (2004) - An animated adventure that follows a young boy's journey to the North Pole on a magical train.
    4. "Home Alone" (1990) - A classic comedy about a young boy who is accidentally left behind at home while his family goes on vacation for Christmas.
    5. "A Christmas Story" (1983) - A classic comedy about a
    

## Conclusion
And that's it! You've successfully built a movie recommendation system using Fireworks, MongoDB, and the nomic-ai embedding model. This system can be further customized and scaled to suit various needs. There are still a few things that is missing in our guides
- we used the default 768 embedding dimension in the example. There are cases where the cost for storing the embedding is high, and you might want to reduce that, and we will walk you through another example with MongoDB + leveraging Matryoshka embedding to reduce embedding size in [this guide](examples/rag/mongo_reduced_embeddings.ipynb)
- we are only documenting 400 movies in this example, which is not a lot. This is because we wanted to keep this tutorial simple and not batching the embedding lookups, and just have a for loop that goes through all the documents and embed them manually. This method does not scale. First, we will cover basic batching in the [following guide](examples/rag/mongo_reduced_embeddings.ipynb). There are a lot of great frameworks that offer batching out of the box, and please check out our guides here for [LlamaIndex](https://github.com/run-llama/llama_index/blob/cf0da01e0cc756383e07eb499cb9825cfa17984d/docs/examples/vector_stores/MongoDBAtlasVectorSearchRAGFireworks.ipynb)
