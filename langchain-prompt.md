```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from dotenv import load_dotenv
import json

import os
from literalai import LiteralClient

```


```python
load_dotenv()
print(os.getenv("LITERAL_API_KEY"))
print(os.getenv("TAVILY_API_KEY"))
print(os.getenv("OPENAI_API_KEY"))
```


```python
literalai_client = LiteralClient()
literalai_cb = literalai_client.langchain_callback()
```


```python
# ---- Option 1: Load the weather assistant prompt JSON ----
with open('weather-assistant-prompt.json', 'r') as file:
    weather_assistant_prompt = json.load(file)

prompt = literalai_client.api.get_or_create_prompt(
    name=weather_assistant_prompt["name"], 
    template_messages=weather_assistant_prompt["template_messages"],
    tools=weather_assistant_prompt["tools"],
    settings=weather_assistant_prompt["settings"]
)

```


```python
# ---- Option 2: Load the weather assistant prompt from LiteralAI ----
# prompt = literalai_client.api.get_prompt(name="weather-assistant")

```


```python
print(prompt)
```

    {
        "createdAt": "2024-10-01T14:06:14.323Z",
        "id": "0bf0ef39-81bc-412f-9039-b8987ac9ba91",
        "name": "weather-assistant",
        "provider": "",
        "settings": {
            "model": "gpt-4o-mini",
            "temperature": 0.5
        },
        "templateMessages": [
            {
                "content": "You are a weather assistant.",
                "role": "system",
                "uuid": "e2bfc8ef-eb36-4555-b788-4f90b6d14803"
            }
        ],
        "tools": [
            {
                "function": {
                    "description": "Get the current weather",
                    "name": "get_current_weather",
                    "parameters": {
                        "properties": {
                            "format": {
                                "description": "The temperature unit to use. Infer this from the users location.",
                                "enum": [
                                    "celsius",
                                    "fahrenheit"
                                ],
                                "type": "string"
                            },
                            "location": {
                                "description": "The city and state, e.g. San Francisco, CA",
                                "type": "string"
                            }
                        },
                        "required": [
                            "location",
                            "format"
                        ],
                        "type": "object"
                    }
                },
                "type": "function"
            }
        ],
        "type": "CHAT",
        "updatedAt": "",
        "url": "",
        "variables": [],
        "variablesDefaultValues": null,
        "version": 0,
        "versionDesc": null
    }
    


```python
# Messages input to the LLM
langchain_prompt = prompt.to_langchain_chat_prompt_template()
messages = langchain_prompt.format_messages()+ [HumanMessage(content="what is the weather in london?")]

# Configure the LLM
llm = ChatOpenAI(
    model=prompt.settings["model"], 
    temperature=prompt.settings["temperature"]
    )

# Bind the tools to the LLM
llm_with_tools =  llm.bind_tools(prompt.tools)
```


```python

# Run the LLM with the callback
llm_with_tools.invoke(
    messages, 
    config=RunnableConfig(callbacks=[literalai_cb])
)

```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_V1Nu6B0BBr9E44qvDVXq5CfF', 'function': {'arguments': '{"format":"celsius","location":"London"}', 'name': 'get_current_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 20, 'prompt_tokens': 94, 'total_tokens': 114, 'completion_tokens_details': {'reasoning_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_f85bea6784', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-61ecddf7-c69d-49bf-b7c7-2458821b89ab-0', tool_calls=[{'name': 'get_current_weather', 'args': {'format': 'celsius', 'location': 'London'}, 'id': 'call_V1Nu6B0BBr9E44qvDVXq5CfF', 'type': 'tool_call'}], usage_metadata={'input_tokens': 94, 'output_tokens': 20, 'total_tokens': 114})




```python

```
