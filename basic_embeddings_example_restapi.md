<h1 align ="center"> REST API Reference Samples</h1>
<hr>
   
# Get Embeddings
   
Get a vector representation of a given input that can be easily consumed by machine learning models and other algorithms.
In this example, we'll see how to get embeddings using REST API Call.


```python
import json
import requests
import openai
import os
```

### Setup Parameters


Here we will load the configurations from _config.json_ file to setup deployment_name, openai_api_base, openai_api_key and openai_api_version.


```python
# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)
    
# Setting up the deployment name
deployment_name = config_details['EMBEDDINGS_MODEL']

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai_api_base = config_details['OPENAI_API_BASE']

# The API key for your Azure OpenAI resource.
openai_api_key = os.getenv("OPENAI_API_KEY")

# Currently OPENAI API have the following versions available: 2022-12-01
openai_api_version = config_details['OPENAI_API_VERSION']
```


```python
# Request URL
api_url = f"{openai_api_base}/openai/deployments/{deployment_name}/embeddings?api-version={openai_api_version}"

# Example prompt for request payload
input="The food was delicious and the waiter..."

# Json payload
json_data = {
  "input": input
}

# Setting the API key in the HTTP headers
headers =  {"api-key": openai_api_key}

try:
    # The response will contain embeddings, which you can extract, save, and use.
    response = requests.post(api_url, json=json_data, headers=headers)

    # Getting the JSON object of the result
    embeddings = response.json()
    
    # Print embeddings
    print(embeddings['data'][0]['embedding'][:20])
    
except:
    print("An exception has occurred. \n")
    print("Error Message:", embeddings['error']['message'])
```

    [-0.02121484, -0.006876593, -0.017789261, -0.041751347, 0.0017212679, -0.022232339, 0.0194003, 0.038088355, -0.0007700131, -0.012676333, -0.021096133, 0.002331767, 0.012905271, -0.016127348, 0.0047016903, 0.014473913, 0.006194021, 0.0078092995, -0.012303251, 0.011726668]
    


```python

```
