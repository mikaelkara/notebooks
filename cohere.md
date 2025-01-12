---
sidebar_label: Cohere
---
# Cohere

This notebook covers how to get started with [Cohere chat models](https://cohere.com/chat).

Head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.cohere.ChatCohere.html) for detailed documentation of all attributes and methods.

## Setup

The integration lives in the `langchain-cohere` package. We can install these with:

```bash
pip install -U langchain-cohere
```

We'll also need to get a [Cohere API key](https://cohere.com/) and set the `COHERE_API_KEY` environment variable:


```python
import getpass
import os

os.environ["COHERE_API_KEY"] = getpass.getpass()
```

It's also helpful (but not needed) to set up [LangSmith](https://smith.langchain.com/) for best-in-class observability


```python
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## Usage

ChatCohere supports all [ChatModel](/docs/how_to#chat-models) functionality:


```python
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
```


```python
chat = ChatCohere()
```


```python
messages = [HumanMessage(content="1"), HumanMessage(content="2 3")]
chat.invoke(messages)
```




    AIMessage(content='4 && 5 \n6 || 7 \n\nWould you like to play a game of odds and evens?', additional_kwargs={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '2076b614-52b3-4082-a259-cc92cd3d9fea', 'token_count': {'prompt_tokens': 68, 'response_tokens': 23, 'total_tokens': 91, 'billed_tokens': 77}}, response_metadata={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '2076b614-52b3-4082-a259-cc92cd3d9fea', 'token_count': {'prompt_tokens': 68, 'response_tokens': 23, 'total_tokens': 91, 'billed_tokens': 77}}, id='run-3475e0c8-c89b-4937-9300-e07d652455e1-0')




```python
await chat.ainvoke(messages)
```




    AIMessage(content='4 && 5', additional_kwargs={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': 'f0708a92-f874-46ee-9b93-334d616ad92e', 'token_count': {'prompt_tokens': 68, 'response_tokens': 3, 'total_tokens': 71, 'billed_tokens': 57}}, response_metadata={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': 'f0708a92-f874-46ee-9b93-334d616ad92e', 'token_count': {'prompt_tokens': 68, 'response_tokens': 3, 'total_tokens': 71, 'billed_tokens': 57}}, id='run-1635e63e-2994-4e7f-986e-152ddfc95777-0')




```python
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

    4 && 5


```python
chat.batch([messages])
```




    [AIMessage(content='4 && 5', additional_kwargs={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '6770ca86-f6c3-4ba3-a285-c4772160612f', 'token_count': {'prompt_tokens': 68, 'response_tokens': 3, 'total_tokens': 71, 'billed_tokens': 57}}, response_metadata={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '6770ca86-f6c3-4ba3-a285-c4772160612f', 'token_count': {'prompt_tokens': 68, 'response_tokens': 3, 'total_tokens': 71, 'billed_tokens': 57}}, id='run-8d6fade2-1b39-4e31-ab23-4be622dd0027-0')]



## Chaining

You can also easily combine with a prompt template for easy structuring of user input. We can do this using [LCEL](/docs/concepts/lcel)


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | chat
```


```python
chain.invoke({"topic": "bears"})
```




    AIMessage(content='What color socks do bears wear?\n\nThey don’t wear socks, they have bear feet. \n\nHope you laughed! If not, maybe this will help: laughter is the best medicine, and a good sense of humor is infectious!', additional_kwargs={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '6edccf44-9bc8-4139-b30e-13b368f3563c', 'token_count': {'prompt_tokens': 68, 'response_tokens': 51, 'total_tokens': 119, 'billed_tokens': 108}}, response_metadata={'documents': None, 'citations': None, 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '6edccf44-9bc8-4139-b30e-13b368f3563c', 'token_count': {'prompt_tokens': 68, 'response_tokens': 51, 'total_tokens': 119, 'billed_tokens': 108}}, id='run-ef7f9789-0d4d-43bf-a4f7-f2a0e27a5320-0')



## Tool calling

Cohere supports tool calling functionalities!


```python
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
```


```python
@tool
def magic_function(number: int) -> int:
    """Applies a magic operation to an integer
    Args:
        number: Number to have magic operation performed on
    """
    return number + 10


def invoke_tools(tool_calls, messages):
    for tool_call in tool_calls:
        selected_tool = {"magic_function": magic_function}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    return messages


tools = [magic_function]
```


```python
llm_with_tools = chat.bind_tools(tools=tools)
messages = [HumanMessage(content="What is the value of magic_function(2)?")]
```


```python
res = llm_with_tools.invoke(messages)
while res.tool_calls:
    messages.append(res)
    messages = invoke_tools(res.tool_calls, messages)
    res = llm_with_tools.invoke(messages)

res
```




    AIMessage(content='The value of magic_function(2) is 12.', additional_kwargs={'documents': [{'id': 'magic_function:0:2:0', 'output': '12', 'tool_name': 'magic_function'}], 'citations': [ChatCitation(start=34, end=36, text='12', document_ids=['magic_function:0:2:0'])], 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '96a55791-0c58-4e2e-bc2a-8550e137c46d', 'token_count': {'input_tokens': 998, 'output_tokens': 59}}, response_metadata={'documents': [{'id': 'magic_function:0:2:0', 'output': '12', 'tool_name': 'magic_function'}], 'citations': [ChatCitation(start=34, end=36, text='12', document_ids=['magic_function:0:2:0'])], 'search_results': None, 'search_queries': None, 'is_search_required': None, 'generation_id': '96a55791-0c58-4e2e-bc2a-8550e137c46d', 'token_count': {'input_tokens': 998, 'output_tokens': 59}}, id='run-f318a9cf-55c8-44f4-91d1-27cf46c6a465-0')


