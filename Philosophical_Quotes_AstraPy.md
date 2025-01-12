# Philosophy with Vector Embeddings, OpenAI and Astra DB

### AstraPy version

In this quickstart you will learn how to build a "philosophy quote finder & generator" using OpenAI's vector embeddings and DataStax [Astra DB](https://docs.datastax.com/en/astra/home/astra.html) as the vector store for data persistence.

The basic workflow of this notebook is outlined below. You will evaluate and store the vector embeddings for a number of quotes by famous philosophers, use them to build a powerful search engine and, after that, even a generator of new quotes!

The notebook exemplifies some of the standard usage patterns of vector search -- while showing how easy is it to get started with [Astra DB](https://docs.datastax.com/en/astra/home/astra.html).

For a background on using vector search and text embeddings to build a question-answering system, please check out this excellent hands-on notebook: [Question answering using embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).

Table of contents:
- Setup
- Create vector collection
- Connect to OpenAI
- Load quotes into the Vector Store
- Use case 1: **quote search engine**
- Use case 2: **quote generator**
- Cleanup

### How it works

**Indexing**

Each quote is made into an embedding vector with OpenAI's `Embedding`. These are saved in the Vector Store for later use in searching. Some metadata, including the author's name and a few other pre-computed tags, are stored alongside, to allow for search customization.

![1_vector_indexing](https://user-images.githubusercontent.com/14221764/282422016-1d540607-eed4-4240-9c3d-22ee3a3bc90f.png)

**Search**

To find a quote similar to the provided search quote, the latter is made into an embedding vector on the fly, and this vector is used to query the store for similar vectors ... i.e. similar quotes that were previously indexed. The search can optionally be constrained by additional metadata ("find me quotes by Spinoza similar to this one ...").

![2_vector_search](https://user-images.githubusercontent.com/14221764/282422033-0a1297c4-63bb-4e04-b120-dfd98dc1a689.png)

The key point here is that "quotes similar in content" translates, in vector space, to vectors that are metrically close to each other: thus, vector similarity search effectively implements semantic similarity. _This is the key reason vector embeddings are so powerful._

The sketch below tries to convey this idea. Each quote, once it's made into a vector, is a point in space. Well, in this case it's on a sphere, since OpenAI's embedding vectors, as most others, are normalized to _unit length_. Oh, and the sphere is actually not three-dimensional, rather 1536-dimensional!

So, in essence, a similarity search in vector space returns the vectors that are closest to the query vector:

![3_vector_space](https://user-images.githubusercontent.com/14221764/262321363-c8c625c1-8be9-450e-8c68-b1ed518f990d.png)

**Generation**

Given a suggestion (a topic or a tentative quote), the search step is performed, and the first returned results (quotes) are fed into an LLM prompt which asks the generative model to invent a new text along the lines of the passed examples _and_ the initial suggestion.

![4_quote_generation](https://user-images.githubusercontent.com/14221764/282422050-2e209ff5-07d6-41ac-99ac-f442e090b3bb.png)

## Setup

Install and import the necessary dependencies:


```python
!pip install --quiet "astrapy>=0.6.0" "openai>=1.0.0" datasets
```


```python
from getpass import getpass
from collections import Counter

from astrapy.db import AstraDB
import openai
from datasets import load_dataset
```

### Connection parameters

Please retrieve your database credentials on your Astra dashboard ([info](https://docs.datastax.com/en/astra/astra-db-vector/)): you will supply them momentarily.

Example values:

- API Endpoint: `https://01234567-89ab-cdef-0123-456789abcdef-us-east1.apps.astra.datastax.com`
- Token: `AstraCS:6gBhNmsk135...`


```python
ASTRA_DB_API_ENDPOINT = input("Please enter your API Endpoint:")
ASTRA_DB_APPLICATION_TOKEN = getpass("Please enter your Token")
```

    Please enter your API Endpoint: https://4f835778-ec78-42b0-9ae3-29e3cf45b596-us-east1.apps.astra.datastax.com
    Please enter your Token ········
    

### Instantiate an Astra DB client


```python
astra_db = AstraDB(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)
```

## Create vector collection

The only parameter to specify, other than the collection name, is the dimension of the vectors you'll store. Other parameters, notably the similarity metric to use for searches, are optional.


```python
coll_name = "philosophers_astra_db"
collection = astra_db.create_collection(coll_name, dimension=1536)
```

## Connect to OpenAI

### Set up your secret key


```python
OPENAI_API_KEY = getpass("Please enter your OpenAI API Key: ")
```

    Please enter your OpenAI API Key:  ········
    

### A test call for embeddings

Quickly check how one can get the embedding vectors for a list of input texts:


```python
client = openai.OpenAI(api_key=OPENAI_API_KEY)
embedding_model_name = "text-embedding-3-small"

result = client.embeddings.create(
    input=[
        "This is a sentence",
        "A second sentence"
    ],
    model=embedding_model_name,
)
```

_Note: the above is the syntax for OpenAI v1.0+. If using previous versions, the code to get the embeddings will look different._


```python
print(f"len(result.data)              = {len(result.data)}")
print(f"result.data[1].embedding      = {str(result.data[1].embedding)[:55]}...")
print(f"len(result.data[1].embedding) = {len(result.data[1].embedding)}")
```

    len(result.data)              = 2
    result.data[1].embedding      = [-0.0108176339417696, 0.0013546717818826437, 0.00362232...
    len(result.data[1].embedding) = 1536
    

## Load quotes into the Vector Store

Get a dataset with the quotes. _(We adapted and augmented the data from [this Kaggle dataset](https://www.kaggle.com/datasets/mertbozkurt5/quotes-by-philosophers), ready to use in this demo.)_


```python
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
```

A quick inspection:


```python
print("An example entry:")
print(philo_dataset[16])
```

    An example entry:
    {'author': 'aristotle', 'quote': 'Love well, be loved and do something of value.', 'tags': 'love;ethics'}
    

Check the dataset size:


```python
author_count = Counter(entry["author"] for entry in philo_dataset)
print(f"Total: {len(philo_dataset)} quotes. By author:")
for author, count in author_count.most_common():
    print(f"    {author:<20}: {count} quotes")
```

    Total: 450 quotes. By author:
        aristotle           : 50 quotes
        schopenhauer        : 50 quotes
        spinoza             : 50 quotes
        hegel               : 50 quotes
        freud               : 50 quotes
        nietzsche           : 50 quotes
        sartre              : 50 quotes
        plato               : 50 quotes
        kant                : 50 quotes
    

### Write to the vector collection

You will compute the embeddings for the quotes and save them into the Vector Store, along with the text itself and the metadata you'll use later.

To optimize speed and reduce the calls, you'll perform batched calls to the embedding OpenAI service.

To store the quote objects, you will use the `insert_many` method of the collection (one call per batch). When preparing the documents for insertion you will choose suitable field names -- keep in mind, however, that the embedding vector must be the fixed special `$vector` field.


```python
BATCH_SIZE = 20

num_batches = ((len(philo_dataset) + BATCH_SIZE - 1) // BATCH_SIZE)

quotes_list = philo_dataset["quote"]
authors_list = philo_dataset["author"]
tags_list = philo_dataset["tags"]

print("Starting to store entries: ", end="")
for batch_i in range(num_batches):
    b_start = batch_i * BATCH_SIZE
    b_end = (batch_i + 1) * BATCH_SIZE
    # compute the embedding vectors for this batch
    b_emb_results = client.embeddings.create(
        input=quotes_list[b_start : b_end],
        model=embedding_model_name,
    )
    # prepare the documents for insertion
    b_docs = []
    for entry_idx, emb_result in zip(range(b_start, b_end), b_emb_results.data):
        if tags_list[entry_idx]:
            tags = {
                tag: True
                for tag in tags_list[entry_idx].split(";")
            }
        else:
            tags = {}
        b_docs.append({
            "quote": quotes_list[entry_idx],
            "$vector": emb_result.embedding,
            "author": authors_list[entry_idx],
            "tags": tags,
        })
    # write to the vector collection
    collection.insert_many(b_docs)
    print(f"[{len(b_docs)}]", end="")

print("\nFinished storing entries.")
```

    Starting to store entries: [20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][20][10]
    Finished storing entries.
    

## Use case 1: **quote search engine**

For the quote-search functionality, you need first to make the input quote into a vector, and then use it to query the store (besides handling the optional metadata into the search call, that is).

Encapsulate the search-engine functionality into a function for ease of re-use. At its core is the `vector_find` method of the collection:


```python
def find_quote_and_author(query_quote, n, author=None, tags=None):
    query_vector = client.embeddings.create(
        input=[query_quote],
        model=embedding_model_name,
    ).data[0].embedding
    filter_clause = {}
    if author:
        filter_clause["author"] = author
    if tags:
        filter_clause["tags"] = {}
        for tag in tags:
            filter_clause["tags"][tag] = True
    #
    results = collection.vector_find(
        query_vector,
        limit=n,
        filter=filter_clause,
        fields=["quote", "author"]
    )
    return [
        (result["quote"], result["author"])
        for result in results
    ]
```

### Putting search to test

Passing just a quote:


```python
find_quote_and_author("We struggle all our life for nothing", 3)
```




    [('Life to the great majority is only a constant struggle for mere existence, with the certainty of losing it at last.',
      'schopenhauer'),
     ('We give up leisure in order that we may have leisure, just as we go to war in order that we may have peace.',
      'aristotle'),
     ('Perhaps the gods are kind to us, by making life more disagreeable as we grow older. In the end death seems less intolerable than the manifold burdens we carry',
      'freud')]



Search restricted to an author:


```python
find_quote_and_author("We struggle all our life for nothing", 2, author="nietzsche")
```




    [('To live is to suffer, to survive is to find some meaning in the suffering.',
      'nietzsche'),
     ('What makes us heroic?--Confronting simultaneously our supreme suffering and our supreme hope.',
      'nietzsche')]



Search constrained to a tag (out of those saved earlier with the quotes):


```python
find_quote_and_author("We struggle all our life for nothing", 2, tags=["politics"])
```




    [('He who seeks equality between unequals seeks an absurdity.', 'spinoza'),
     ('The people are that part of the state that does not know what it wants.',
      'hegel')]



### Cutting out irrelevant results

The vector similarity search generally returns the vectors that are closest to the query, even if that means results that might be somewhat irrelevant if there's nothing better.

To keep this issue under control, you can get the actual "similarity" between the query and each result, and then implement a cutoff on it, effectively discarding results that are beyond that threshold.
Tuning this threshold correctly is not an easy problem: here, we'll just show you the way.

To get a feeling on how this works, try the following query and play with the choice of quote and threshold to compare the results. Note that the similarity is returned as the special `$similarity` field in each result document - and it will be returned by default, unless you pass `include_similarity = False` to the search method.

_Note (for the mathematically inclined): this value is **a rescaling between zero and one** of the cosine difference between the vectors, i.e. of the scalar product divided by the product of the norms of the two vectors. In other words, this is 0 for opposite-facing vectors and +1 for parallel vectors. For other measures of similarity (cosine is the default), check the `metric` parameter in `AstraDB.create_collection` and the [documentation on allowed values](https://docs.datastax.com/en/astra-serverless/docs/develop/dev-with-json.html#metric-types)._


```python
quote = "Animals are our equals."
# quote = "Be good."
# quote = "This teapot is strange."

metric_threshold = 0.92

quote_vector = client.embeddings.create(
    input=[quote],
    model=embedding_model_name,
).data[0].embedding

results_full = collection.vector_find(
    quote_vector,
    limit=8,
    fields=["quote"]
)
results = [res for res in results_full if res["$similarity"] >= metric_threshold]

print(f"{len(results)} quotes within the threshold:")
for idx, result in enumerate(results):
    print(f"    {idx}. [similarity={result['$similarity']:.3f}] \"{result['quote'][:70]}...\"")
```

    3 quotes within the threshold:
        0. [similarity=0.927] "The assumption that animals are without rights, and the illusion that ..."
        1. [similarity=0.922] "Animals are in possession of themselves; their soul is in possession o..."
        2. [similarity=0.920] "At his best, man is the noblest of all animals; separated from law and..."
    

## Use case 2: **quote generator**

For this task you need another component from OpenAI, namely an LLM to generate the quote for us (based on input obtained by querying the Vector Store).

You also need a template for the prompt that will be filled for the generate-quote LLM completion task.


```python
completion_model_name = "gpt-3.5-turbo"

generation_prompt_template = """"Generate a single short philosophical quote on the given topic,
similar in spirit and form to the provided actual example quotes.
Do not exceed 20-30 words in your quote.

REFERENCE TOPIC: "{topic}"

ACTUAL EXAMPLES:
{examples}
"""
```

Like for search, this functionality is best wrapped into a handy function (which internally uses search):


```python
def generate_quote(topic, n=2, author=None, tags=None):
    quotes = find_quote_and_author(query_quote=topic, n=n, author=author, tags=tags)
    if quotes:
        prompt = generation_prompt_template.format(
            topic=topic,
            examples="\n".join(f"  - {quote[0]}" for quote in quotes),
        )
        # a little logging:
        print("** quotes found:")
        for q, a in quotes:
            print(f"**    - {q} ({a})")
        print("** end of logging")
        #
        response = client.chat.completions.create(
            model=completion_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=320,
        )
        return response.choices[0].message.content.replace('"', '').strip()
    else:
        print("** no quotes found.")
        return None
```

_Note: similar to the case of the embedding computation, the code for the Chat Completion API would be slightly different for OpenAI prior to v1.0._

#### Putting quote generation to test

Just passing a text (a "quote", but one can actually just suggest a topic since its vector embedding will still end up at the right place in the vector space):


```python
q_topic = generate_quote("politics and virtue")
print("\nA new generated quote:")
print(q_topic)
```

    ** quotes found:
    **    - Happiness is the reward of virtue. (aristotle)
    **    - Our moral virtues benefit mainly other people; intellectual virtues, on the other hand, benefit primarily ourselves; therefore the former make us universally popular, the latter unpopular. (schopenhauer)
    ** end of logging
    
    A new generated quote:
    True politics lies in the virtuous pursuit of justice, for it is through virtue that we build a better world for all.
    

Use inspiration from just a single philosopher:


```python
q_topic = generate_quote("animals", author="schopenhauer")
print("\nA new generated quote:")
print(q_topic)
```

    ** quotes found:
    **    - Because Christian morality leaves animals out of account, they are at once outlawed in philosophical morals; they are mere 'things,' mere means to any ends whatsoever. They can therefore be used for vivisection, hunting, coursing, bullfights, and horse racing, and can be whipped to death as they struggle along with heavy carts of stone. Shame on such a morality that is worthy of pariahs, and that fails to recognize the eternal essence that exists in every living thing, and shines forth with inscrutable significance from all eyes that see the sun! (schopenhauer)
    **    - The assumption that animals are without rights, and the illusion that our treatment of them has no moral significance, is a positively outrageous example of Western crudity and barbarity. Universal compassion is the only guarantee of morality. (schopenhauer)
    ** end of logging
    
    A new generated quote:
    Excluding animals from ethical consideration reveals a moral blindness that allows for their exploitation and suffering. True morality embraces universal compassion.
    

## Cleanup

If you want to remove all resources used for this demo, run this cell (_warning: this will irreversibly delete the collection and its data!_):


```python
astra_db.delete_collection(coll_name)
```




    {'status': {'ok': 1}}


