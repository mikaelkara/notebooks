# Azure functions example

> Note: There is a newer version of the openai library available. See https://github.com/openai/openai-python/discussions/742

This notebook shows how to use the function calling capability with the Azure OpenAI service. Functions allow a caller of chat completions to define capabilities that the model can use to extend its
functionality into external tools and data sources.

You can read more about chat functions on OpenAI's blog: https://openai.com/blog/function-calling-and-other-api-updates

**NOTE**: Chat functions require model versions beginning with gpt-4 and gpt-35-turbo's `-0613` labels. They are not supported by older versions of the models.

## Setup

First, we install the necessary dependencies.


```python
! pip install "openai>=0.28.1,<1.0.0"
# (Optional) If you want to use Microsoft Active Directory
! pip install azure-identity
```


```python
import os
import openai
```


Additionally, to properly access the Azure OpenAI Service, we need to create the proper resources at the [Azure Portal](https://portal.azure.com) (you can check a detailed guide on how to do this in the [Microsoft Docs](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal))

Once the resource is created, the first thing we need to use is its endpoint. You can get the endpoint by looking at the *"Keys and Endpoints"* section under the *"Resource Management"* section. Having this, we will set up the SDK using this information:


```python
openai.api_base = "" # Add your endpoint here

# functions is only supported by the 2023-07-01-preview API version
openai.api_version = "2023-07-01-preview"
```

### Authentication

The Azure OpenAI service supports multiple authentication mechanisms that include API keys and Azure credentials.


```python
use_azure_active_directory = False
```


#### Authentication using API key

To set up the OpenAI SDK to use an *Azure API Key*, we need to set up the `api_type` to `azure` and set `api_key` to a key associated with your endpoint (you can find this key in *"Keys and Endpoints"* under *"Resource Management"* in the [Azure Portal](https://portal.azure.com))


```python
if not use_azure_active_directory:
    openai.api_type = "azure"
    openai.api_key = os.environ["OPENAI_API_KEY"]
```

> Note: In this example, we configured the library to use the Azure API by setting the variables in code. For development, consider setting the environment variables instead:

```
OPENAI_API_BASE
OPENAI_API_KEY
OPENAI_API_TYPE
OPENAI_API_VERSION
```

#### Authentication using Microsoft Active Directory
Let's now see how we can get a key via Microsoft Active Directory Authentication.


```python
from azure.identity import DefaultAzureCredential

if use_azure_active_directory:
    default_credential = DefaultAzureCredential()
    token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

    openai.api_type = "azure_ad"
    openai.api_key = token.token
```

A token is valid for a period of time, after which it will expire. To ensure a valid token is sent with every request, you can refresh an expiring token by hooking into requests.auth:


```python
import typing
import time
import requests

if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class TokenRefresh(requests.auth.AuthBase):

    def __init__(self, credential: "TokenCredential", scopes: typing.List[str]) -> None:
        self.credential = credential
        self.scopes = scopes
        self.cached_token: typing.Optional[str] = None

    def __call__(self, req):
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        req.headers["Authorization"] = f"Bearer {self.cached_token.token}"
        return req

if use_azure_active_directory:
    session = requests.Session()
    session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])

    openai.requestssession = session
```

## Functions

With setup and authentication complete, you can now use functions with the Azure OpenAI service. This will be split into a few steps:

1. Define the function(s)
2. Pass function definition(s) into chat completions API
3. Call function with arguments from the response
4. Feed function response back into chat completions API

#### 1. Define the function(s)

A list of functions can be defined, each containing the name of the function, an optional description, and the parameters the function accepts (described as a JSON schema).


```python
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location"],
        },
    }
]
```

#### 2. Pass function definition(s) into chat completions API

Now we can pass the function into the chat completions API. If the model determines it should call the function, a `finish_reason` of "function_call" will be populated on the choice and the details of which function to call and its arguments will be present in the `message`. Optionally, you can set the `function_call` keyword argument to force the model to call a particular function (e.g. `function_call={"name": get_current_weather}`). By default, this is set to `auto`, allowing the model to choose whether to call the function or not. 


```python
messages = [
    {"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
    {"role": "user", "content": "What's the weather like today in Seattle?"}
]

chat_completion = openai.ChatCompletion.create(
    deployment_id="gpt-35-turbo-0613",
    messages=messages,
    functions=functions,
)
print(chat_completion)
```

    {
      "choices": [
        {
          "content_filter_results": {},
          "finish_reason": "function_call",
          "index": 0,
          "message": {
            "function_call": {
              "arguments": "{\n  \"location\": \"Seattle, WA\"\n}",
              "name": "get_current_weather"
            },
            "role": "assistant"
          }
        }
      ],
      "created": 1689702512,
      "id": "chatcmpl-7dj6GkYdM7Vw9eGn02bc2qqjN70Ps",
      "model": "gpt-4",
      "object": "chat.completion",
      "prompt_annotations": [
        {
          "content_filter_results": {
            "hate": {
              "filtered": false,
              "severity": "safe"
            },
            "self_harm": {
              "filtered": false,
              "severity": "safe"
            },
            "sexual": {
              "filtered": false,
              "severity": "safe"
            },
            "violence": {
              "filtered": false,
              "severity": "safe"
            }
          },
          "prompt_index": 0
        }
      ],
      "usage": {
        "completion_tokens": 18,
        "prompt_tokens": 115,
        "total_tokens": 133
      }
    }
    

#### 3. Call function with arguments from the response

The name of the function call will be one that was provided initially and the arguments will include JSON matching the schema included in the function definition.


```python
import json

def get_current_weather(request):
    """
    This function is for illustrative purposes.
    The location and unit should be used to determine weather
    instead of returning a hardcoded response.
    """
    location = request.get("location")
    unit = request.get("unit")
    return {"temperature": "22", "unit": "celsius", "description": "Sunny"}

function_call =  chat_completion.choices[0].message.function_call
print(function_call.name)
print(function_call.arguments)

if function_call.name == "get_current_weather":
    response = get_current_weather(json.loads(function_call.arguments))
```

    get_current_weather
    {
      "location": "Seattle, WA"
    }
    

#### 4. Feed function response back into chat completions API

The response from the function should be serialized into a new message with the role set to "function". Now the model will use the response data to formulate its answer.


```python
messages.append(
    {
        "role": "function",
        "name": "get_current_weather",
        "content": json.dumps(response)
    }
)

function_completion = openai.ChatCompletion.create(
    deployment_id="gpt-35-turbo-0613",
    messages=messages,
    functions=functions,
)

print(function_completion.choices[0].message.content.strip())
```

    Today in Seattle, the weather is sunny with a temperature of 22 degrees celsius.
    
