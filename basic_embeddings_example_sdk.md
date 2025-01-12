<h1 align ="center"> Python SDK Samples</h1>
<hr>

## Get Embeddings

Get a vector representation of a given input that can be easily consumed by machine learning models and other algorithms.
In this example we'll see how to get embeddings using the Azure endpoints.


```python
import json
import openai
import os
```

### Setup Parameters


Here we will load the configurations from _config.json_ file to setup deployment name, openai api base, openai api key and openai api version.


```python
# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)
    
# Setting up the deployment name
deployment_name = config_details['EMBEDDINGS_MODEL']

# This is set to `azure`
openai.api_type = "azure"

# The API key for your Azure OpenAI resource.
openai.api_key = os.getenv("OPENAI_API_KEY")

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai.api_base = config_details['OPENAI_API_BASE']

# Currently OPENAI API have the following versions available: 2022-12-01
openai.api_version = config_details['OPENAI_API_VERSION']
```


```python
 try:
    # Get embeddings
    embeddings = openai.Embedding.create(engine=deployment_name, input="The food was delicious and the waiter...")["data"][0]["embedding"]

    # Number of embeddings    
    len(embeddings)
    
    # Print embeddings
    print(embeddings[:20])
        
except openai.error.APIError as e:
    # Handle API error here, e.g. retry or log
    print(f"OpenAI API returned an API Error: {e}")

except openai.error.AuthenticationError as e:
    # Handle Authentication error here, e.g. invalid API key
    print(f"OpenAI API returned an Authentication Error: {e}")

except openai.error.APIConnectionError as e:
    # Handle connection error here
    print(f"Failed to connect to OpenAI API: {e}")

except openai.error.InvalidRequestError as e:
    # Handle connection error here
    print(f"Invalid Request Error: {e}")

except openai.error.RateLimitError as e:
    # Handle rate limit error
    print(f"OpenAI API request exceeded rate limit: {e}")

except openai.error.ServiceUnavailableError as e:
    # Handle Service Unavailable error
    print(f"Service Unavailable: {e}")

except openai.error.Timeout as e:
    # Handle request timeout
    print(f"Request timed out: {e}")
```

    [-0.021214839071035385, -0.006876592990010977, -0.017789261415600777, -0.04175134748220444, 0.001721267937682569, -0.022232338786125183, 0.019400300458073616, 0.0380883552134037, -0.0007700130809098482, -0.012676333077251911, -0.021096132695674896, 0.0023317669983953238, 0.012905270792543888, -0.016127347946166992, 0.004701690282672644, 0.014473913237452507, 0.006194021087139845, 0.007809299509972334, -0.012303250841796398, 0.01172666810452938]
    


```python

```
