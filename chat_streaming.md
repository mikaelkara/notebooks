---
sidebar_position: 1.5
---
# How to stream chat model responses


All [chat models](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html) implement the [Runnable interface](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable), which comes with a **default** implementations of standard runnable methods (i.e. `ainvoke`, `batch`, `abatch`, `stream`, `astream`, `astream_events`).

The **default** streaming implementation provides an`Iterator` (or `AsyncIterator` for asynchronous streaming) that yields a single value: the final output from the underlying chat model provider.

:::tip

The **default** implementation does **not** provide support for token-by-token streaming, but it ensures that the the model can be swapped in for any other model as it supports the same standard interface.

:::

The ability to stream the output token-by-token depends on whether the provider has implemented proper streaming support.

See which [integrations support token-by-token streaming here](/docs/integrations/chat/).

## Sync streaming

Below we use a `|` to help visualize the delimiter between tokens.


```python
from langchain_anthropic.chat_models import ChatAnthropic

chat = ChatAnthropic(model="claude-3-haiku-20240307")
for chunk in chat.stream("Write me a 1 verse song about goldfish on the moon"):
    print(chunk.content, end="|", flush=True)
```

    Here| is| a| |1| |verse| song| about| gol|dfish| on| the| moon|:|
    
    Floating| up| in| the| star|ry| night|,|
    Fins| a|-|gl|im|mer| in| the| pale| moon|light|.|
    Gol|dfish| swimming|,| peaceful| an|d free|,|
    Se|ren|ely| |drif|ting| across| the| lunar| sea|.|

## Async Streaming


```python
from langchain_anthropic.chat_models import ChatAnthropic

chat = ChatAnthropic(model="claude-3-haiku-20240307")
async for chunk in chat.astream("Write me a 1 verse song about goldfish on the moon"):
    print(chunk.content, end="|", flush=True)
```

    Here| is| a| |1| |verse| song| about| gol|dfish| on| the| moon|:|
    
    Floating| up| above| the| Earth|,|
    Gol|dfish| swim| in| alien| m|irth|.|
    In| their| bowl| of| lunar| dust|,|
    Gl|it|tering| scales| reflect| the| trust|
    Of| swimming| free| in| this| new| worl|d,|
    Where| their| aqu|atic| dream|'s| unf|ur|le|d.|

## Astream events

Chat models also support the standard [astream events](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) method.

This method is useful if you're streaming output from a larger LLM application that contains multiple steps (e.g., an LLM chain composed of a prompt, llm and parser).


```python
from langchain_anthropic.chat_models import ChatAnthropic

chat = ChatAnthropic(model="claude-3-haiku-20240307")
idx = 0

async for event in chat.astream_events(
    "Write me a 1 verse song about goldfish on the moon", version="v1"
):
    idx += 1
    if idx >= 5:  # Truncate the output
        print("...Truncated")
        break
    print(event)
```

    {'event': 'on_chat_model_start', 'run_id': '08da631a-12a0-4f07-baee-fc9a175ad4ba', 'name': 'ChatAnthropic', 'tags': [], 'metadata': {}, 'data': {'input': 'Write me a 1 verse song about goldfish on the moon'}}
    {'event': 'on_chat_model_stream', 'run_id': '08da631a-12a0-4f07-baee-fc9a175ad4ba', 'tags': [], 'metadata': {}, 'name': 'ChatAnthropic', 'data': {'chunk': AIMessageChunk(content='Here', id='run-08da631a-12a0-4f07-baee-fc9a175ad4ba')}}
    {'event': 'on_chat_model_stream', 'run_id': '08da631a-12a0-4f07-baee-fc9a175ad4ba', 'tags': [], 'metadata': {}, 'name': 'ChatAnthropic', 'data': {'chunk': AIMessageChunk(content="'s", id='run-08da631a-12a0-4f07-baee-fc9a175ad4ba')}}
    {'event': 'on_chat_model_stream', 'run_id': '08da631a-12a0-4f07-baee-fc9a175ad4ba', 'tags': [], 'metadata': {}, 'name': 'ChatAnthropic', 'data': {'chunk': AIMessageChunk(content=' a', id='run-08da631a-12a0-4f07-baee-fc9a175ad4ba')}}
    ...Truncated
    
