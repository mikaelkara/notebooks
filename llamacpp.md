# Llama.cpp

>[llama.cpp python](https://github.com/abetlen/llama-cpp-python) library is a simple Python bindings for `@ggerganov`
>[llama.cpp](https://github.com/ggerganov/llama.cpp).
>
>This package provides:
>
> - Low-level access to C API via ctypes interface.
> - High-level Python API for text completion
>   - `OpenAI`-like API
>   - `LangChain` compatibility
>   - `LlamaIndex` compatibility
> - OpenAI compatible web server
>   - Local Copilot replacement
>   - Function Calling support
>   - Vision API support
>   - Multiple Models


## Overview

### Integration details
| Class | Package | Local | Serializable | JS support |
| :--- | :--- | :---: | :---: |  :---: |
| [ChatLlamaCpp](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.llamacpp.ChatLlamaCpp.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | 

## Setup

To get started and use **all** the features show below, we reccomend using a model that has been fine-tuned for tool-calling.

We will use [
Hermes-2-Pro-Llama-3-8B-GGUF](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF) from NousResearch. 

> Hermes 2 Pro is an upgraded version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house. This new version of Hermes maintains its excellent general task and conversation capabilities - but also excels at Function Calling

See our guides on local models to go deeper:

* [Run LLMs locally](https://python.langchain.com/v0.1/docs/guides/development/local_llms/)
* [Using local models with RAG](https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/)

### Installation

The LangChain LlamaCpp integration lives in the `langchain-community` and `llama-cpp-python` packages:


```python
%pip install -qU langchain-community llama-cpp-python
```

## Instantiation

Now we can instantiate our model object and generate chat completions:


```python
# Path to your model weights
local_model = "local/path/to/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"
```


```python
import multiprocessing

from langchain_community.chat_models import ChatLlamaCpp

llm = ChatLlamaCpp(
    temperature=0.5,
    model_path=local_model,
    n_ctx=10000,
    n_gpu_layers=8,
    n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=512,
    n_threads=multiprocessing.cpu_count() - 1,
    repeat_penalty=1.5,
    top_p=0.5,
    verbose=True,
)
```

## Invocation


```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
ai_msg
```


```python
print(ai_msg.content)
```

    J'aime programmer. (In France, "programming" is often used in its original sense of scheduling or organizing events.) 
    
    If you meant computer-programming: 
    Je suis amoureux de la programmation informatique.
    
    (You might also say simply 'programmation', which would be understood as both meanings - depending on context).
    

## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)
```

## Tool calling

Firstly, it works mostly the same as OpenAI Function Calling

OpenAI has a [tool calling](https://platform.openai.com/docs/guides/function-calling) (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.

With `ChatLlamaCpp.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to an OpenAI tool schemas, which looks like:
```
{
    "name": "...",
    "description": "...",
    "parameters": {...}  # JSONSchema
}
```
and passed in every model invocation.


However, it cannot automatically trigger a function/tool, we need to force it by specifying the 'tool choice' parameter. This parameter is typically formatted as described below.

```{"type": "function", "function": {"name": <<tool_name>>}}.```


```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(enum=["celsius", "fahrenheit"])


@tool("get_current_weather", args_schema=WeatherInput)
def get_weather(location: str, unit: str):
    """Get the current weather in a given location"""
    return f"Now the weather in {location} is 22 {unit}"


llm_with_tools = llm.bind_tools(
    tools=[get_weather],
    tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
)
```


```python
ai_msg = llm_with_tools.invoke(
    "what is the weather like in HCMC in celsius",
)
```


```python
ai_msg.tool_calls
```




    [{'name': 'get_current_weather',
      'args': {'location': 'Ho Chi Minh City', 'unit': 'celsius'},
      'id': 'call__0_get_current_weather_cmpl-394d9943-0a1f-425b-8139-d2826c1431f2'}]




```python
class MagicFunctionInput(BaseModel):
    magic_function_input: int = Field(description="The input value for magic function")


@tool("get_magic_function", args_schema=MagicFunctionInput)
def magic_function(magic_function_input: int):
    """Get the value of magic function for an input."""
    return magic_function_input + 2


llm_with_tools = llm.bind_tools(
    tools=[magic_function],
    tool_choice={"type": "function", "function": {"name": "get_magic_function"}},
)

ai_msg = llm_with_tools.invoke(
    "What is magic function of 3?",
)

ai_msg
```


```python
ai_msg.tool_calls
```




    [{'name': 'get_magic_function',
      'args': {'magic_function_input': 3},
      'id': 'call__0_get_magic_function_cmpl-cd83a994-b820-4428-957c-48076c68335a'}]



# Structured output


```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel


class Joke(BaseModel):
    """A setup to a joke and the punchline."""

    setup: str
    punchline: str


dict_schema = convert_to_openai_tool(Joke)
structured_llm = llm.with_structured_output(dict_schema)
result = structured_llm.invoke("Tell me a joke about birds")
result
```


```python
result
```




    {'setup': '- Why did the chicken cross the playground?',
     'punchline': '\n\n- To get to its gilded cage on the other side!'}



# Streaming



```python
for chunk in llm.stream("what is 25x5"):
    print(chunk.content, end="\n", flush=True)
```

## API reference

For detailed documentation of all ChatLlamaCpp features and configurations head to the API reference: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.llamacpp.ChatLlamaCpp.html
