# Azure OpenAI Service API with OpenAI Python SDK

This notebook shows how to use the Azure OpenAI (AOAI) Service API with the [OpenAI Python SDK](https://github.com/openai/openai-python).

### Prerequisites

1. Setup for Azure Active Directory (AAD) authentication.
    * See [Setup to use AAD and test with CLI](setup_aad.md).
2. A Python environment setup with all the requirements.  
    * See the [setup_python_env.md](setup_python_env.md) page for instructions on setting up your environment.

### Configuring OpenAI SDK
The OpenAI SDK can be configured via _config_ file.  

 | SDK Variable | Description |
 | -- | -- |
 | `api_type` | API Type.  Use `azure_ad` for AAD authentication.  |
 | `api_version` | API Version.  You can use `2022-12-01`. |
 | `api_base` | Base URL for the API.  This is the endpoint URL for your team. |




```python
import json
import openai
import os
```

### Setup Parameters


Here we will load the configurations from _config.json_ file to setup deployment_name, openai.api_base and openai.api_version.


```python
# Load config values
with open(r'config.json') as config_file:
    config_details = json.load(config_file)

# Setting up the deployment name
deployment_name = config_details['COMPLETIONS_MODEL']

# For Active Directory authentication
openai.api_type = "azure_ad"

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai.api_base = config_details['OPENAI_API_BASE']

# Currently OPENAI API have the following versions available: 2022-12-01
openai.api_version = config_details['OPENAI_API_VERSION']
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
#default_credential = AzureCliCredential()
default_credential = DefaultAzureCredential()
```

## Define method to get a prompt completion using AAD authentication

The `refresh_token` function below is used to get a new token when the current token expires.  The `refresh_token` method is called by the `get_completion` to get the token if it is not already set or if the token has expired.


```python
import datetime

token = None

def refresh_token():
    """Refresh AAD token"""
    global token
    # Check if Azure token is still valid
    if not token or datetime.datetime.fromtimestamp(token.expires_on) < datetime.datetime.now():
        token = default_credential.get_token("https://cognitiveservices.azure.com")

def get_completion(**kwargs):
    # Refresh token
    refresh_token()
    # Set the API key to be your Bearer token (yes this could be optimizaed to not do this every time :D) 
    openai.api_key = token.token
    
    return openai.Completion.create(
        # Contact your team admin to get the name of your engine or model deployment.  
        # This is the name that they used when they created the model deployment.
        engine=deployment_name,
        **kwargs
    )

```

## Test the API

The payload parameters in the `get_completion` call are for illustration only and can be changed for your use case as described in the [reference](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference).


```python
# Example prompt 
prompt = "Once upon a time"

response = get_completion(
    prompt=prompt,
    temperature=0.7,
    max_tokens=300,
    top_p=0.5,
    stop=None
)

# printing the response
print(response)
```

    {
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "logprobs": null,
          "text": "\nthere was a little girl named Emma. She lived in a small village with her parents and two brothers. Emma was a very curious girl and loved to explore the world around her.\n\nOne day, Emma decided to go on an adventure. She packed her bag with food and water and set off into the forest. As she walked, she noticed many different animals and plants. She also saw a lot of strange and mysterious things.\n\nEmma eventually came across a small cottage in the middle of the forest. She knocked on the door and a kindly old woman answered. The woman welcomed Emma inside and offered her a warm meal and a place to sleep.\n\nThe old woman told Emma about the magical creatures that lived in the forest. She also told her about a powerful wizard who lived nearby. Emma was so excited to learn about all of these things and she decided to stay with the old woman for a while.\n\nThe old woman taught Emma many things about the forest and its creatures. She also gave Emma a magical necklace that would protect her from harm.\n\nEmma thanked the old woman for her kindness and hospitality and set off on her journey. She explored the forest and encountered many more magical creatures. Eventually, she made it to the wizard's castle and was granted three wishes.\n\nEmma used her wishes to help her family and the people of her village. She also used her wishes to protect the magical creatures of the forest.\n\nEmma's adventure changed her life forever. She returned home with a newfound appreciation for the world around her and a newfound respect for the magical creatures that lived in the forest."
        }
      ],
      "created": 1680086585,
      "id": "cmpl-6zNYfdtf0KC5ArhdPJsUctgI7Ng3C",
      "model": "text-davinci-003",
      "object": "text_completion",
      "usage": {
        "completion_tokens": 333,
        "prompt_tokens": 5,
        "total_tokens": 338
      }
    }
    


```python

```
