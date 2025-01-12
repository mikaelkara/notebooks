# Using Weaviate with Generative OpenAI module for Generative Search

This notebook is prepared for a scenario where:
* Your data is already in Weaviate
* You want to use Weaviate with the Generative OpenAI module ([generative-openai](https://weaviate.io/developers/weaviate/modules/reader-generator-modules/generative-openai)).



## Prerequisites

This cookbook only coveres Generative Search examples, however, it doesn't cover the configuration and data imports.

In order to make the most of this cookbook, please complete the [Getting Started cookbook](./getting-started-with-weaviate-and-openai.ipynb) first, where you will learn the essentials of working with Weaviate and import the demo data.

Checklist:
* completed [Getting Started cookbook](./getting-started-with-weaviate-and-openai.ipynb),
* crated a `Weaviate` instance,
* imported data into your `Weaviate` instance,
* you have an [OpenAI API key](https://beta.openai.com/account/api-keys)

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

## Generative Search
Weaviate offers a [Generative Search OpenAI](https://weaviate.io/developers/weaviate/modules/reader-generator-modules/generative-openai) module, which generates responses based on the data stored in your Weaviate instance.

The way you construct a generative search query is very similar to a standard semantic search query in Weaviate. 

For example:
* search in "Articles", 
* return "title", "content", "url"
* look for objects related to "football clubs"
* limit results to 5 objects

```
    result = (
        client.query
        .get("Articles", ["title", "content", "url"])
        .with_near_text("concepts": "football clubs")
        .with_limit(5)
        # generative query will go here
        .do()
    )
```

Now, you can add `with_generate()` function to apply generative transformation. `with_generate` takes either:
- `single_prompt` - to generate a response for each returned object,
- `grouped_task` – to generate a single response from all returned objects.



```python
def generative_search_per_item(query, collection_name):
    prompt = "Summarize in a short tweet the following content: {content}"

    result = (
        client.query
        .get(collection_name, ["title", "content", "url"])
        .with_near_text({ "concepts": [query], "distance": 0.7 })
        .with_limit(5)
        .with_generate(single_prompt=prompt)
        .do()
    )
    
    # Check for errors
    if ("errors" in result):
        print ("\033[91mYou probably have run out of OpenAI API calls for the current minute – the limit is set at 60 per minute.")
        raise Exception(result["errors"][0]['message'])
    
    return result["data"]["Get"][collection_name]
```


```python
query_result = generative_search_per_item("football clubs", "Article")

for i, article in enumerate(query_result):
    print(f"{i+1}. { article['title']}")
    print(article['_additional']['generate']['singleResult']) # print generated response
    print("-----------------------")
```


```python
def generative_search_group(query, collection_name):
    generateTask = "Explain what these have in common"

    result = (
        client.query
        .get(collection_name, ["title", "content", "url"])
        .with_near_text({ "concepts": [query], "distance": 0.7 })
        .with_generate(grouped_task=generateTask)
        .with_limit(5)
        .do()
    )
    
    # Check for errors
    if ("errors" in result):
        print ("\033[91mYou probably have run out of OpenAI API calls for the current minute – the limit is set at 60 per minute.")
        raise Exception(result["errors"][0]['message'])
    
    return result["data"]["Get"][collection_name]
```


```python
query_result = generative_search_group("football clubs", "Article")

print (query_result[0]['_additional']['generate']['groupedResult'])
```

Thanks for following along, you're now equipped to set up your own vector databases and use embeddings to do all kinds of cool things - enjoy! For more complex use cases please continue to work through other cookbook examples in this repo.
