---
sidebar_label: MistralAI
---
# ChatMistralAI

This will help you getting started with Mistral [chat models](/docs/concepts/chat_models). For detailed documentation of all `ChatMistralAI` features and configurations head to the [API reference](https://python.langchain.com/api_reference/mistralai/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html). The `ChatMistralAI` class is built on top of the [Mistral API](https://docs.mistral.ai/api/). For a list of all the models supported by Mistral, check out [this page](https://docs.mistral.ai/getting-started/models/).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/mistral) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatMistralAI](https://python.langchain.com/api_reference/mistralai/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html) | [langchain_mistralai](https://python.langchain.com/api_reference/mistralai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_mistralai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_mistralai?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 

## Setup


To access `ChatMistralAI` models you'll need to create a Mistral account, get an API key, and install the `langchain_mistralai` integration package.

### Credentials


A valid [API key](https://console.mistral.ai/users/api-keys/) is needed to communicate with the API. Once you've done this set the MISTRAL_API_KEY environment variable:


```python
import getpass
import os

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter your Mistral API key: ")
```

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

The LangChain Mistral integration lives in the `langchain_mistralai` package:


```python
%pip install -qU langchain_mistralai
```

## Instantiation

Now we can instantiate our model object and generate chat completions:


```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
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




    AIMessage(content='Sure, I\'d be happy to help you translate that sentence into French! The English sentence "I love programming" translates to "J\'aime programmer" in French. Let me know if you have any other questions or need further assistance!', response_metadata={'token_usage': {'prompt_tokens': 32, 'total_tokens': 84, 'completion_tokens': 52}, 'model': 'mistral-small', 'finish_reason': 'stop'}, id='run-64bac156-7160-4b68-b67e-4161f63e021f-0', usage_metadata={'input_tokens': 32, 'output_tokens': 52, 'total_tokens': 84})




```python
print(ai_msg.content)
```

    Sure, I'd be happy to help you translate that sentence into French! The English sentence "I love programming" translates to "J'aime programmer" in French. Let me know if you have any other questions or need further assistance!
    

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




    AIMessage(content='Ich liebe Programmierung. (German translation)', response_metadata={'token_usage': {'prompt_tokens': 26, 'total_tokens': 38, 'completion_tokens': 12}, 'model': 'mistral-small', 'finish_reason': 'stop'}, id='run-dfd4094f-e347-47b0-9056-8ebd7ea35fe7-0', usage_metadata={'input_tokens': 26, 'output_tokens': 12, 'total_tokens': 38})



## API reference

Head to the [API reference](https://python.langchain.com/api_reference/mistralai/chat_models/langchain_mistralai.chat_models.ChatMistralAI.html) for detailed documentation of all attributes and methods.
