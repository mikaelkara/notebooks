---
sidebar_label: Friendli
---
# ChatFriendli

> [Friendli](https://friendli.ai/) enhances AI application performance and optimizes cost savings with scalable, efficient deployment options, tailored for high-demand AI workloads.

This tutorial guides you through integrating `ChatFriendli` for chat applications using LangChain. `ChatFriendli` offers a flexible approach to generating conversational AI responses, supporting both synchronous and asynchronous calls.

## Setup

Ensure the `langchain_community` and `friendli-client` are installed.

```sh
pip install -U langchain-community friendli-client.
```

Sign in to [Friendli Suite](https://suite.friendli.ai/) to create a Personal Access Token, and set it as the `FRIENDLI_TOKEN` environment.


```python
import getpass
import os

if "FRIENDLI_TOKEN" not in os.environ:
    os.environ["FRIENDLI_TOKEN"] = getpass.getpass("Friendi Personal Access Token: ")
```

You can initialize a Friendli chat model with selecting the model you want to use. The default model is `mixtral-8x7b-instruct-v0-1`. You can check the available models at [docs.friendli.ai](https://docs.periflow.ai/guides/serverless_endpoints/pricing#text-generation-models).


```python
from langchain_community.chat_models.friendli import ChatFriendli

chat = ChatFriendli(model="llama-2-13b-chat", max_tokens=100, temperature=0)
```

## Usage

`FrienliChat` supports all methods of [`ChatModel`](/docs/how_to#chat-models) including async APIs.

You can also use functionality of  `invoke`, `batch`, `generate`, and `stream`.


```python
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage

system_message = SystemMessage(content="Answer questions as short as you can.")
human_message = HumanMessage(content="Tell me a joke.")
messages = [system_message, human_message]

chat.invoke(messages)
```




    AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!")




```python
chat.batch([messages, messages])
```




    [AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!"),
     AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!")]




```python
chat.generate([messages, messages])
```




    LLMResult(generations=[[ChatGeneration(text=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!", message=AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!"))], [ChatGeneration(text=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!", message=AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!"))]], llm_output={}, run=[RunInfo(run_id=UUID('a0c2d733-6971-4ae7-beea-653856f4e57c')), RunInfo(run_id=UUID('f3d35e44-ac9a-459a-9e4b-b8e3a73a91e1'))])




```python
for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)
```

     Knock, knock!
    Who's there?
    Cows go.
    Cows go who?
    MOO!

You can also use all functionality of async APIs: `ainvoke`, `abatch`, `agenerate`, and `astream`.


```python
await chat.ainvoke(messages)
```




    AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!")




```python
await chat.abatch([messages, messages])
```




    [AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!"),
     AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!")]




```python
await chat.agenerate([messages, messages])
```




    LLMResult(generations=[[ChatGeneration(text=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!", message=AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!"))], [ChatGeneration(text=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!", message=AIMessage(content=" Knock, knock!\nWho's there?\nCows go.\nCows go who?\nMOO!"))]], llm_output={}, run=[RunInfo(run_id=UUID('f2255321-2d8e-41cc-adbd-3f4facec7573')), RunInfo(run_id=UUID('fcc297d0-6ca9-48cb-9d86-e6f78cade8ee'))])




```python
async for chunk in chat.astream(messages):
    print(chunk.content, end="", flush=True)
```

     Knock, knock!
    Who's there?
    Cows go.
    Cows go who?
    MOO!
