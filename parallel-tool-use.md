# Parallel Tool use

### Setup

Make sure you have `ipykernel` and `pip` pre-installed


```python
%pip install -r requirements.txt
```


```python
import os
import json

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
"Groq API key configured: " + os.environ["GROQ_API_KEY"][:10] + "..."
```




    'Groq API key configured: gsk_7FdrzM...'



We will use the ```llama3-70b-8192``` model in this demo. Note that you will need a Groq API Key to proceed and can create an account [here](https://console.groq.com/) to generate one for free. Only Llama 3 models support parallel tool use at this time (05/07/2024).

We recommend using the 70B Llama 3 model, 8B has subpar consistency.


```python
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "llama3-70b-8192"
```

Let's define a dummy function we can invoke in our tool use loop


```python
def get_weather(city: str):
    if city == "Madrid":
        return 35
    elif city == "San Francisco":
        return 18
    elif city == "Paris":
        return 20
    else:
        return 15
```

Now we define our messages and tools and run the completion request.


```python
messages = [
    {"role": "system", "content": """You are a helpful assistant."""},
    {
        "role": "user",
        "content": "What is the weather in Paris, Tokyo and Madrid?",
    },
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns the weather in the given city in degrees Celsius",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city",
                    }
                },
                "required": ["city"],
            },
        },
    }
]
response = client.chat.completions.create(
    model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=4096
)

response_message = response.choices[0].message
```

# Processing the tool calls

Now we process the assistant message and construct the required messages to continue the conversation. 

*Including* invoking each tool_call against our actual function.


```python
tool_calls = response_message.tool_calls

messages.append(
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tool_call.id,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
                "type": tool_call.type,
            }
            for tool_call in tool_calls
        ],
    }
)

available_functions = {
    "get_weather": get_weather,
}
for tool_call in tool_calls:
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(**function_args)

    # Note how we create a separate tool call message for each tool call
    # the model is able to discern the tool call result through the tool_call_id
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(function_response),
            "tool_call_id": tool_call.id,
        }
    )

print(json.dumps(messages, indent=2))
```

    [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the weather in Paris, Tokyo and Madrid?"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_5ak8",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Paris\"}"
            },
            "type": "function"
          },
          {
            "id": "call_zq26",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Tokyo\"}"
            },
            "type": "function"
          },
          {
            "id": "call_znf3",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Madrid\"}"
            },
            "type": "function"
          }
        ]
      },
      {
        "role": "tool",
        "content": "20",
        "tool_call_id": "call_5ak8"
      },
      {
        "role": "tool",
        "content": "15",
        "tool_call_id": "call_zq26"
      },
      {
        "role": "tool",
        "content": "35",
        "tool_call_id": "call_znf3"
      }
    ]
    

Now we run our final completion with multiple tool call results included in the messages array.

**Note**

We pass the tool definitions again to help the model understand:

1. The assistant message with the tool call
2. Interpret the tool results.


```python
response = client.chat.completions.create(
    model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=4096
)

print(response.choices[0].message.content)
```

    The weather in Paris is 20°C, in Tokyo is 15°C, and in Madrid is 35°C.
    
