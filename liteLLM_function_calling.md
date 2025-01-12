## Demo Notebook of Function Calling with liteLLM
- Supported Providers for Function Calling
  - OpenAI - `gpt-4-0613` and `gpt-3.5-turbo-0613`
- In this notebook we use function calling with `litellm.completion()`


```python
## Install liteLLM
!pip install litellm
```


```python
import os, litellm
from litellm import completion
```


```python
os.environ['OPENAI_API_KEY'] = "" #@param
```

## Define Messages, Functions
We create a get_current_weather() function and pass that to GPT 3.5

See OpenAI docs for this: https://openai.com/blog/function-calling-and-other-api-updates


```python
messages = [
    {"role": "user", "content": "What is the weather like in Boston?"}
]

def get_current_weather(location):
  if location == "Boston, MA":
    return "The weather is 12F"

functions = [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  ]
```

## Call gpt-3.5-turbo-0613 to Decide what Function to call


```python
response = completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
print(response)
```

    {
      "id": "chatcmpl-7mX4RiqdoislVEqfmfVjFSKp3hyIy",
      "object": "chat.completion",
      "created": 1691801223,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "function_call": {
              "name": "get_current_weather",
              "arguments": "{\n  \"location\": \"Boston, MA\"\n}"
            }
          },
          "finish_reason": "function_call"
        }
      ],
      "usage": {
        "prompt_tokens": 82,
        "completion_tokens": 18,
        "total_tokens": 100
      }
    }
    

## Parse GPT 3.5 Response
Read Information about what Function to Call


```python
function_call_data = response["choices"][0]["message"]["function_call"]
function_call_data
```




    <OpenAIObject at 0x7922c70ce930> JSON: {
      "name": "get_current_weather",
      "arguments": "{\n  \"location\": \"Boston, MA\"\n}"
    }




```python
import json
function_name = function_call_data['name']
function_args = function_call_data['arguments']
function_args = json.loads(function_args)
print(function_name, function_args)

```

    get_current_weather {'location': 'Boston, MA'}
    

## Call the get_current_weather() function


```python
if function_name == "get_current_weather":
  result = get_current_weather(**function_args)
  print(result)
```

    12F
    

## Send the response from get_current_weather back to the model to summarize


```python
messages = [
    {"role": "user", "content": "What is the weather like in Boston?"},
    {"role": "assistant", "content": None, "function_call": {"name": "get_current_weather", "arguments": "{ \"location\": \"Boston, MA\"}"}},
    {"role": "function", "name": "get_current_weather", "content": result}
]
response = completion(model="gpt-3.5-turbo-0613", messages=messages, functions=functions)
print(response)
```

    {
      "id": "chatcmpl-7mXGN62u75WXp1Lgen4iSgNvA7hHT",
      "object": "chat.completion",
      "created": 1691801963,
      "model": "gpt-3.5-turbo-0613",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The current weather in Boston is 12 degrees Fahrenheit."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 109,
        "completion_tokens": 12,
        "total_tokens": 121
      }
    }
    
