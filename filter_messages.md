# How to filter messages

In more complex chains and agents we might track state with a list of messages. This list can start to accumulate messages from multiple different models, speakers, sub-chains, etc., and we may only want to pass subsets of this full list of messages to each model call in the chain/agent.

The `filter_messages` utility makes it easy to filter messages by type, id, or name.

## Basic usage


```python
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

filter_messages(messages, include_types="human")
```




    [HumanMessage(content='example input', name='example_user', id='2'),
     HumanMessage(content='real input', name='bob', id='4')]




```python
filter_messages(messages, exclude_names=["example_user", "example_assistant"])
```




    [SystemMessage(content='you are a good assistant', id='1'),
     HumanMessage(content='real input', name='bob', id='4'),
     AIMessage(content='real output', name='alice', id='5')]




```python
filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])
```




    [HumanMessage(content='example input', name='example_user', id='2'),
     HumanMessage(content='real input', name='bob', id='4'),
     AIMessage(content='real output', name='alice', id='5')]



## Chaining

`filter_messages` can be used in an imperatively (like above) or declaratively, making it easy to compose with other components in a chain:


```python
%pip install -qU langchain-anthropic
```


```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
# Notice we don't pass in messages. This creates
# a RunnableLambda that takes messages as input
filter_ = filter_messages(exclude_names=["example_user", "example_assistant"])
chain = filter_ | llm
chain.invoke(messages)
```




    AIMessage(content=[], response_metadata={'id': 'msg_01Wz7gBHahAwkZ1KCBNtXmwA', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 3}}, id='run-b5d8a3fe-004f-4502-a071-a6c025031827-0', usage_metadata={'input_tokens': 16, 'output_tokens': 3, 'total_tokens': 19})



Looking at the LangSmith trace we can see that before the messages are passed to the model they are filtered: https://smith.langchain.com/public/f808a724-e072-438e-9991-657cc9e7e253/r

Looking at just the filter_, we can see that it's a Runnable object that can be invoked like all Runnables:


```python
filter_.invoke(messages)
```




    [HumanMessage(content='real input', name='bob', id='4'),
     AIMessage(content='real output', name='alice', id='5')]



## API reference

For a complete description of all arguments head to the API reference: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.filter_messages.html
