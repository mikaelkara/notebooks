---
sidebar_label: OCIGenAI
---
# ChatOCIGenAI

This notebook provides a quick overview for getting started with OCIGenAI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatOCIGenAI features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_generative_ai.ChatOCIGenAI.html).

Oracle Cloud Infrastructure (OCI) Generative AI is a fully managed service that provides a set of state-of-the-art, customizable large language models (LLMs) that cover a wide range of use cases, and which is available through a single API.
Using the OCI Generative AI service you can access ready-to-use pretrained models, or create and host your own fine-tuned custom models based on your own data on dedicated AI clusters. Detailed documentation of the service and API is available __[here](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)__ and __[here](https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai/20231130/)__.


## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/oci_generative_ai) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatOCIGenAI](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_generative_ai.ChatOCIGenAI.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-oci-generative-ai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-oci-generative-ai?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 

## Setup

To access OCIGenAI models you'll need to install the `oci` and `langchain-community` packages.

### Credentials

The credentials and authentication methods supported for this integration are equivalent to those used with other OCI services and follow the __[standard SDK authentication](https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm)__ methods, specifically API Key, session token, instance principal, and resource principal.

API key is the default authentication method used in the examples above. The following example demonstrates how to use a different authentication method (session token)

### Installation

The LangChain OCIGenAI integration lives in the `langchain-community` package and you will also need to install the `oci` package:


```python
%pip install -qU langchain-community oci
```

## Instantiation

Now we can instantiate our model object and generate chat completions:



```python
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

chat = ChatOCIGenAI(
    model_id="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="MY_OCID",
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)
```

## Invocation


```python
messages = [
    SystemMessage(content="your are an AI assistant."),
    AIMessage(content="Hi there human!"),
    HumanMessage(content="tell me a joke."),
]
response = chat.invoke(messages)
```


```python
print(response.content)
```

## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:



```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | chat

response = chain.invoke({"topic": "dogs"})
print(response.content)
```

## API reference

For detailed documentation of all ChatOCIGenAI features and configurations head to the API reference: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_generative_ai.ChatOCIGenAI.html
