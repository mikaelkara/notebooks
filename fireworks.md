---
sidebar_label: Fireworks
---
# ChatFireworks

This doc help you get started with Fireworks AI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatFireworks features and configurations head to the [API reference](https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html).

Fireworks AI is an AI inference platform to run and customize models. For a list of all models served by Fireworks see the [Fireworks docs](https://fireworks.ai/models).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/fireworks) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatFireworks](https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html) | [langchain-fireworks](https://python.langchain.com/api_reference/fireworks/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-fireworks?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-fireworks?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 

## Setup

To access Fireworks models you'll need to create a Fireworks account, get an API key, and install the `langchain-fireworks` integration package.

### Credentials

Head to (ttps://fireworks.ai/login to sign up to Fireworks and generate an API key. Once you've done this set the FIREWORKS_API_KEY environment variable:


```python
import getpass
import os

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Enter your Fireworks API key: ")
```

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

The LangChain Fireworks integration lives in the `langchain-fireworks` package:


```python
%pip install -qU langchain-fireworks
```

## Instantiation

Now we can instantiate our model object and generate chat completions:

- TODO: Update model instantiation with relevant params.


```python
from langchain_fireworks import ChatFireworks

llm = ChatFireworks(
    model="accounts/fireworks/models/llama-v3-70b-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
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




    AIMessage(content="J'adore la programmation.", response_metadata={'token_usage': {'prompt_tokens': 35, 'total_tokens': 44, 'completion_tokens': 9}, 'model_name': 'accounts/fireworks/models/llama-v3-70b-instruct', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='run-df28e69a-ff30-457e-a743-06eb14d01cb0-0', usage_metadata={'input_tokens': 35, 'output_tokens': 9, 'total_tokens': 44})




```python
print(ai_msg.content)
```

    J'adore la programmation.
    

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




    AIMessage(content='Ich liebe das Programmieren.', response_metadata={'token_usage': {'prompt_tokens': 30, 'total_tokens': 37, 'completion_tokens': 7}, 'model_name': 'accounts/fireworks/models/llama-v3-70b-instruct', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None}, id='run-ff3f91ad-ed81-4acf-9f59-7490dc8d8f48-0', usage_metadata={'input_tokens': 30, 'output_tokens': 7, 'total_tokens': 37})



## API reference

For detailed documentation of all ChatFireworks features and configurations head to the API reference: https://python.langchain.com/api_reference/fireworks/chat_models/langchain_fireworks.chat_models.ChatFireworks.html
