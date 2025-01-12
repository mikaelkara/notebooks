# How to merge consecutive messages of the same type

Certain models do not support passing in consecutive messages of the same type (a.k.a. "runs" of the same message type).

The `merge_message_runs` utility makes it easy to merge consecutive messages of the same type.

### Setup


```python
%pip install -qU langchain-core langchain-anthropic
```

## Basic usage


```python
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

messages = [
    SystemMessage("you're a good assistant."),
    SystemMessage("you always respond with a joke."),
    HumanMessage([{"type": "text", "text": "i wonder why it's called langchain"}]),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    AIMessage("Why, he's probably chasing after the last cup of coffee in the office!"),
]

merged = merge_message_runs(messages)
print("\n\n".join([repr(x) for x in merged]))
```

    SystemMessage(content="you're a good assistant.\nyou always respond with a joke.", additional_kwargs={}, response_metadata={})
    
    HumanMessage(content=[{'type': 'text', 'text': "i wonder why it's called langchain"}, 'and who is harrison chasing anyways'], additional_kwargs={}, response_metadata={})
    
    AIMessage(content='Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!\nWhy, he\'s probably chasing after the last cup of coffee in the office!', additional_kwargs={}, response_metadata={})
    

Notice that if the contents of one of the messages to merge is a list of content blocks then the merged message will have a list of content blocks. And if both messages to merge have string contents then those are concatenated with a newline character.

## Chaining

`merge_message_runs` can be used in an imperatively (like above) or declaratively, making it easy to compose with other components in a chain:


```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
# Notice we don't pass in messages. This creates
# a RunnableLambda that takes messages as input
merger = merge_message_runs()
chain = merger | llm
chain.invoke(messages)
```

    Note: you may need to restart the kernel to use updated packages.
    




    AIMessage(content=[], additional_kwargs={}, response_metadata={'id': 'msg_01KNGUMTuzBVfwNouLDpUMwf', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 84, 'output_tokens': 3}}, id='run-b908b198-9c24-450b-9749-9d4a8182937b-0', usage_metadata={'input_tokens': 84, 'output_tokens': 3, 'total_tokens': 87})



Looking at the LangSmith trace we can see that before the messages are passed to the model they are merged: https://smith.langchain.com/public/ab558677-cac9-4c59-9066-1ecce5bcd87c/r

Looking at just the merger, we can see that it's a Runnable object that can be invoked like all Runnables:


```python
merger.invoke(messages)
```




    [SystemMessage(content="you're a good assistant.\nyou always respond with a joke.", additional_kwargs={}, response_metadata={}),
     HumanMessage(content=[{'type': 'text', 'text': "i wonder why it's called langchain"}, 'and who is harrison chasing anyways'], additional_kwargs={}, response_metadata={}),
     AIMessage(content='Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!\nWhy, he\'s probably chasing after the last cup of coffee in the office!', additional_kwargs={}, response_metadata={})]



`merge_message_runs` can also be placed after a prompt:


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate(
    [
        ("system", "You're great a {skill}"),
        ("system", "You're also great at explaining things"),
        ("human", "{query}"),
    ]
)
chain = prompt | merger | llm
chain.invoke({"skill": "math", "query": "what's the definition of a convergent series"})
```




    AIMessage(content='A convergent series is an infinite series whose partial sums approach a finite value as more terms are added. In other words, the sequence of partial sums has a limit.\n\nMore formally, an infinite series Σ an (where an are the terms of the series) is said to be convergent if the sequence of partial sums:\n\nS1 = a1\nS2 = a1 + a2  \nS3 = a1 + a2 + a3\n...\nSn = a1 + a2 + a3 + ... + an\n...\n\nconverges to some finite number S as n goes to infinity. We write:\n\nlim n→∞ Sn = S\n\nThe finite number S is called the sum of the convergent infinite series.\n\nIf the sequence of partial sums does not approach any finite limit, the infinite series is said to be divergent.\n\nSome key properties:\n- A series converges if and only if the sequence of its partial sums is a Cauchy sequence.\n- Absolute/conditional convergence criteria help determine if a given series converges.\n- Convergent series have many important applications in mathematics, physics, engineering etc.', additional_kwargs={}, response_metadata={'id': 'msg_01MfV6y2hep7ZNvDz24A36U4', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 29, 'output_tokens': 267}}, id='run-9d925f58-021e-4bd0-94fc-f8f5e91010a4-0', usage_metadata={'input_tokens': 29, 'output_tokens': 267, 'total_tokens': 296})



LangSmith Trace: https://smith.langchain.com/public/432150b6-9909-40a7-8ae7-944b7e657438/r/f4ad5fb2-4d38-42a6-b780-25f62617d53f

## API reference

For a complete description of all arguments head to the API reference: https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.merge_message_runs.html
