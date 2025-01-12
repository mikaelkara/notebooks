# Azure OpenAI Service API with AAD authentication

The reference for the AOAI REST API `/completions` method is [here](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/reference/api-reference#completions).

### Prerequisites

1. Setup for Azure Active Directory (AAD) authentication.
    * See [Setup to use AAD and test with CLI](setup_aad.md).
2. A Python environment setup with all the requirements.  
    * See the [setup_python_env.md](setup_python_env.md) page for instructions on setting up your environment.  You don't need the `openai` package for this notebook, but you do need the `azure-identity` package.


### Define the API endpoint URL for your team


```python
# Import needed libraries
import os
import requests
import json
import datetime
import time
```

### Setup Parameters


Here we will load the configurations from _config.json_ file to setup deployment_name, base_url and openai_api_version.


```python
# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)

# Setting up the deployment name
deployment_name = config_details['COMPLETIONS_MODEL']

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
base_url = config_details['OPENAI_API_BASE']

# Currently OPENAI API have the following versions available: 2022-12-01
openai_api_version = config_details['OPENAI_API_VERSION']

# Define the API endpoint URL
request_url = base_url + "/openai/deployments/"+ deployment_name + "/completions?api-version=" + openai_api_version
```

## Set up AAD authentication

`DefaultAzureCredential` can read your credentials from the environment after doing `az login`. 

In VS Code, you can use the Azure Account extension to log in with your Azure account.  If you are running this notebook in VS Code, be sure to restart VS Code after you do `az login`.

This article gives details on what identity `DefaultAzureCredential` uses: https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python

If you get an error similar to the following, you can try using `AzureCliCredential` instead of `DefaultAzureCredential`:

```
DefaultAzureCredential failed to retrieve a token from the included credentials. Attempted credentials:     EnvironmentCredential: EnvironmentCredential authentication unavailable. Environment variables are not fully configured. Visit https://aka.ms/azsdk/python/identity/environmentcredential/troubleshoot to troubleshoot.this issue.     ManagedIdentityCredential: ManagedIdentityCredential authentication unavailable, no response from the IMDS endpoint.     SharedTokenCacheCredential: Azure Active Directory error '(invalid_scope) AADSTS70011: The provided request must include a 'scope' input parameter. The provided value for the input parameter 'scope' is not valid. The scope https://cognitiveservices.azure.com is not valid. The scope format is invalid. Scope must be in a valid URI form <https://example/scope> or a valid Guid <guid/scope>. 
```



```python
from azure.identity import DefaultAzureCredential, AzureCliCredential #DefaultAzureCredential should work but you may need AzureCliCredential to make the authentication work
default_credential = AzureCliCredential()
#default_credential = DefaultAzureCredential()
```

## Define method to get a prompt completion using AAD authentication

The `refresh_token` function below is used to get a new token when the current token expires.  The `refresh_token` method is called 
by the `get_completion` to get the token if it is not already set or if the token has expired.


```python
token = None

def refresh_token():
    global token
    # Check if Azure token is still valid
    if not token or datetime.datetime.fromtimestamp(token.expires_on) < datetime.datetime.now():
        token = default_credential.get_token("https://cognitiveservices.azure.com")

def get_completion(payload):
    # Refresh token
    refresh_token()
    
    # Yes this can be optimized to only set Authorization header when token is refreshed :D
    headers = {
        "Authorization": f"Bearer {token.token}",
        "Content-Type": "application/json"
    }

    r = requests.post(
        request_url, 
        headers=headers,
        json = payload
        )
    
    return r
```

## Test the API

The payload parameters in the `get_completion` call are for illustration only and can be changed for your use case as described in the [reference](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference).


```python
# Example prompt for request payload
prompt = "Once upon a time"

payload = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7,
            "top_p": 1
          }

r = get_completion(payload)

# printing the status code & headers
print(f'Status Code: {r.status_code}\nHeaders: \n{r.headers} \n') 

response = r.json()

# printing the response
print(f'Response:\n {response}')
```

    Status Code: 200
    Headers: 
    {'Cache-Control': 'no-cache, must-revalidate', 'Content-Length': '1239', 'Content-Type': 'application/json', 'access-control-allow-origin': '*', 'openai-model': 'text-davinci-003', 'apim-request-id': 'be5722a6-ffad-4bcb-b988-d70971ff0c84', 'openai-processing-ms': '4552.6285', 'x-content-type-options': 'nosniff', 'x-accel-buffering': 'no', 'x-request-id': '1f720ba0-6fe8-40e7-a55e-de8cb05f6e2b', 'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload', 'x-ms-region': 'East US', 'Date': 'Wed, 29 Mar 2023 11:00:51 GMT'} 
    
    {'id': 'cmpl-6zNpmCskuKaLc0heqY9DIOnzYAzUz', 'object': 'text_completion', 'created': 1680087646, 'model': 'text-davinci-003', 'choices': [{'text': ' there was a young girl named Alice. She lived in a small village in the middle of the forest. One day, Alice was walking through the woods when she stumbled upon a white rabbit. She followed the rabbit deep into the forest until she came to a strange looking tree. The tree had a door at the bottom with a sign that said "Drink Me".\n\nAlice was curious, so she opened the door and drank the liquid inside. Suddenly, Alice began to shrink and she soon found herself in a strange world. She encountered many strange creatures, including talking animals and a strange caterpillar who smoked a hookah.\n\nAlice eventually found her way back home, but she was forever changed by her experience in Wonderland. She never forgot the strange creatures and adventures she experienced.\n\nAlice\'s journey was filled with lessons about friendship, courage, and the power of imagination. It taught her that nothing is impossible and that life can be filled with exciting possibilities.', 'index': 0, 'finish_reason': 'stop', 'logprobs': None}], 'usage': {'completion_tokens': 195, 'prompt_tokens': 4, 'total_tokens': 199}}
    


```python

```
