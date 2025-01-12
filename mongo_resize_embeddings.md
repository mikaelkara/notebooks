<a target="_blank" href="https://colab.research.google.com/drive/1zwgdjjavB6rjyj-87vKGOtAXk_8-7X9O?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Reduced embedding dimension example with Fireworks + MongoDB + Nomic

## Introduction
Hopefully you have went through the [previous cookbook](examples/rag/mongo_basic.ipynb) to go through the basics. In this tutorial, we'll explore how to create an basic movie recommendation system with variable cost for storage quality trade-off. We'll leverage the Fireworks API for embedding generation, MongoDB for data storage and retrieval, and the Nomic-AI embedding model for nuanced understanding of movie data.

## Setting Up Your Environment
Before we dive into the code, make sure to set up your environment. This involves installing necessary packages like pymongo and openai. Run the following command in your notebook to install these packages:


```python
!pip install -q pymongo fireworks-ai tqdm openai
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.3.2[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    

## Initializing Fireworks and MongoDB Clients
To interact with Fireworks and MongoDB, we need to initialize their respective clients. Replace "YOUR FIREWORKS API KEY" and "YOUR MONGO URL" with your actual credentials.


```python
import pymongo

mongo_url = input()
client = pymongo.MongoClient(mongo_url)
```


```python
import openai
fw_client = openai.OpenAI(
  api_key=input(),
  base_url="https://api.fireworks.ai/inference/v1"
)
```

## Indexing and retrieval for movies.
We are going to build a model to index and retrieve movie recommendations. We will setup the most basic RAG example on top of MongoDB which involves
- MongoDB Atlas database that indexes movies based on embeddings
- a system for document embedding generation. We'll use the Nomic-AI model to create embeddings from text data. The function generate_embeddings takes a list of texts and returns dimensionality-reduced embeddings.
  - The Nomic AI model, specifically the `nomic-ai/nomic-embed-text-v1.5` variant, is a great open source model embedding model. You can ask it to not only produce embeddings with size 786, but also embeddings with smaller dimensions all the way down to 64. In this example, we can try to use dimension 128 and see if we can get the example up and running without any quality impact.
- a basic search engine that responds to user query by embedding the user query, fetching the corresponding movies, and then use an LLM to generate the recommendations.

We will update our generate_embeddings example slightly to reflect how we are going to query with variable embedding table dimensions


```python
from typing import List

def generate_embeddings(
    input_texts: List[str],
    model_api_string: str,
    embedding_dimensions: int = 768,
    prefix=""
) -> List[List[float]]:
    """Generate embeddings from Fireworks python library

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.
        embedding_dimensions: int = 768,
        prefix: what prefix to attach to the generate the embeddings, which is required for nomic 1.5. Please check out https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#usage for more information

    Returns:
        reduced_embeddings_list: a list of reduced-size embeddings. Each element corresponds to each input text.
    """
    if prefix:
        input_texts = [prefix + text for text in input_texts] 
    return [x.embedding for x in 
        fw_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
        dimensions=embedding_dimensions,
    ).data]
```

## Data Processing
Now, let's process our movie data. We'll extract key information from our MongoDB collection and generate embeddings for each movie. Ensure NUM_DOC_LIMIT is set to limit the number of documents processed.


```python
embedding_model_string = 'nomic-ai/nomic-embed-text-v1.5'
vector_database_field_name = 'embeddings_128' # define your embedding field name.
NUM_DOC_LIMIT = 2000 # the number of documents you will process and generate embeddings.

sample_output = generate_embeddings(["This is a test."], embedding_model_string, embedding_dimensions=128)
print(f"Embedding size is: {str(len(sample_output[0]))}")

```

    Embedding size is: 128
    

# Batching
we will also walk through how to do basic batching. When you are querying Fireworks API, you can add more than one documents per call, and the embedding results will be returned in the same order. we will batch the 2000 examples into units of 200.


```python
from tqdm import tqdm
from datetime import datetime

db = client.sample_mflix
collection = db.movies

keys_to_extract = ["plot", "genre", "cast", "title", "fullplot", "countries", "directors"]

extracted_str_list = []
for doc in tqdm(collection.find(
  {
    "fullplot":{"$exists": True},
    "released": { "$gt": datetime(2000, 1, 1, 0, 0, 0)},
  }
).limit(NUM_DOC_LIMIT), desc="Document Processing "):
  extracted_str = "\n".join([k + ": " + str(doc[k]) for k in keys_to_extract if k in doc])
  extracted_str_list.append((doc['_id'], extracted_str))

# Chunk extracted_str_list into batches of 512
str_batches = zip(*(iter(extracted_str_list),) * 200)

# Iterate over each batch
for batch in tqdm(str_batches, desc="generate and insert embeddings"):
  # Generate embeddings for the current batch
  embeddings = generate_embeddings(
    [t[1] for t in batch],  # Extract the extracted strings from the tuples
    embedding_model_string,
    prefix="search_document: ",
    embedding_dimensions=128,
  )

  # Update documents with the generated embeddings
  for i, embedding in enumerate(embeddings):
    doc = collection.find_one({'_id': batch[i][0]})
    doc[vector_database_field_name] = embedding
    collection.replace_one({'_id': batch[i][0]}, doc)
```

    Document Processing : 2000it [00:02, 837.45it/s] 
    generate and insert embeddings: 10it [02:54, 17.48s/it]
    

## Setting Up the Search Index
For our system to efficiently search through movie embeddings, we need to set up a search index in MongoDB. Define the index structure as shown:


```python
"""
{
  "fields": [
    {
      "type": "vector",
      "path": "embeddings",
      "numDimensions": 768,
      "similarity": "dotProduct"
    },
    {
      "type": "vector",
      "path": "embeddings_128",
      "numDimensions": 128,
      "similarity": "dotProduct"
    }
  ]
}

"""
```




    '\n{\n  "fields": [\n    {\n      "type": "vector",\n      "path": "embeddings",\n      "numDimensions": 768,\n      "similarity": "dotProduct"\n    }\n  ]\n}\n\n'



## Querying the Recommender System
Let's test our recommender system. We create a query for superhero movies and exclude Spider-Man movies, as per user preference.


```python
# Example query.
query = "I like Christmas movies, any recommendations?"
prefix="search_query: "
query_emb = generate_embeddings([query], embedding_model_string, prefix=prefix, embedding_dimensions=128)[0]

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
    
    1. Christmas Carol: The Movie
    2. Love Actually
    3. Surviving Christmas
    4. Almost Famous
    5. Dead End
    6. Up, Up, and Away!
    7. Do Fish Do It?
    8. Let It Snow
    9. The Little Polar Bear
    10. One Point O
    

We can see that the results are very similar results with just 128 dimensions. So if you feel that 128 dimensions are good enough for your use case, you can reduce the dimensions and save some database cost.

## Generating Recommendations
Finally, we use Fireworks' chat API to generate a personalized movie recommendation based on the user's query and preferences.




```python
your_task_prompt = (
    "From the given movie listing data, choose a few great movie recommendations. "
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

    Based on the user's preference for Christmas movies, here are a few great recommendations from the given movie listing data:
    
    1. Christmas Carol: The Movie - A beautiful animated movie adaptation of Charles Dickens' classic Christmas tale, featuring an all-star cast including Simon Callow, Kate Winslet, and Nicolas Cage.
    2. Love Actually - A heartwarming ensemble romantic comedy set during the Christmas season in London, starring Bill Nighy, Colin Firth, Hugh Grant, and Liam Neeson, among many others.
    3. Surviving Christmas - A funny and touching holiday movie about a rich and lonely man (Ben Affleck) who hires a family to spend Christmas with him, only to find that their presence helps him rediscover the true meaning of the season.
    
    Hope these recommendations fit your taste and bring you some holiday cheer!
    

## Conclusion
You've successfully updated a movie recommendation with batching and variable embeddings. Now if are interested in pushing further to integrate MongoDB + Fireworks into your systems, you can check out our
- [LangChain integration, with function calling](https://github.com/fw-ai/cookbook/blob/main/examples/rag/mongodb_agent.ipynb)
- [LlamaIndex](https://github.com/run-llama/llama_index/blob/cf0da01e0cc756383e07eb499cb9825cfa17984d/docs/examples/vector_stores/MongoDBAtlasVectorSearchRAGFireworks.ipynb)
