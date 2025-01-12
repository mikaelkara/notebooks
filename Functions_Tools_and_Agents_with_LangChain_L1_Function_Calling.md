<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/quickstart/agents/dlai/Functions_Tools_and_Agents_with_LangChain_L1_Function_Calling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [Functions, Tools and Agents with LangChain Lesson 1 OpenAI Function Calling](https://learn.deeplearning.ai/courses/functions-tools-agents-langchain/lesson/2/openai-function-calling) to using Llama 3. 

You should take the course before or after going through this notebook to have a deeper understanding.


```python
!pip install groq
```


```python
import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

known_functions = {
    "get_current_weather": get_current_weather
}
```


```python
# https://console.groq.com/docs/tool-use#models
# Groq API endpoints support tool use for programmatic execution of specified operations through requests with explicitly defined 
# operations. With tool use, Groq API model endpoints deliver structured JSON output that can be used to directly invoke functions.

from groq import Groq
import os
import json

client = Groq(api_key = 'your_groq_api_key' # get a free key at https://console.groq.com/keys')
```


```python
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    }
]
```


```python
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]
```


```python
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    #tools=tools, # you can also replace functions with tools, as specified in https://console.groq.com/docs/tool-use 
    max_tokens=4096, 
    temperature=0
)
```


```python
response
```


```python
response_message = response.choices[0].message
response_message
```


```python
response_message.content
```


```python
response_message.function_call
```


```python
json.loads(response_message.function_call.arguments)
```


```python
args = json.loads(response_message.function_call.arguments)
```


```python
get_current_weather(args)
```


```python
function_call = response.choices[0].message.function_call
function_call
```


```python
function_call.name, function_call.arguments
```


```python
# by defining and using known_functions, we can programatically call function
function_response = known_functions[function_call.name](function_call.arguments)
```


```python
function_response
```


```python
# add the message returned by tool and query LLM again to get final answer
messages.append(
{
    "role": "function",
    "name": function_call.name,
    "content": function_response,
}
)
```


```python
messages
```


```python
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    temperature=0
)

response.choices[0].message.content
```


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
```


```python
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    function_call="none", # default is auto (let LLM decide if using function call or not. can also be none, or a dict {{"name": "func_name"}
    temperature=0
)
```


```python
print(response)
```


```python
response_message = response.choices[0].message
response_message
```


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    function_call="auto", # default is auto (let LLM decide if using function call or not. can also be none, or a dict {{"name": "func_name"}
    temperature=0
)
print(response)
```


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    function_call="none", # default is auto (let LLM decide if using function call or not. can also be none, or a dict {{"name": "func_name"}
    temperature=0
)
print(response)
```


```python
messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    function_call="none", # default is auto (let LLM decide if using function call or not. can also be none, or a dict {{"name": "func_name"}
    temperature=0
)
print(response)
```


```python
messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"}, # default is auto (let LLM decide if using function call or not. can also be none, or a dict {{"name": "func_name"}
    temperature=0
)
print(response)
```


```python
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"}, # default is auto (let LLM decide if using function call or not. can also be none, or a dict {{"name": "func_name"}
    temperature=0
)
print(response)
```


```python
function_call = response.choices[0].message.function_call
function_call.name, function_call.arguments
```


```python
args = json.loads(response.choices[0].message.function_call.arguments)
observation = known_functions[function_call.name](args)
```


```python
observation
```


```python
messages.append(
        {
            "role": "function",
            "name": function_call.name,
            "content": observation,
        }
)
```


```python
messages
```


```python
response = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=messages,
)
print(response)
```


```python
response.choices[0].message.content
```
