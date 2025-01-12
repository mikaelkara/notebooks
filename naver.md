---
sidebar_label: Naver
---
# ChatClovaX

This notebook provides a quick overview for getting started with Naver’s HyperCLOVA X [chat models](https://python.langchain.com/docs/concepts/chat_models) via CLOVA Studio. For detailed documentation of all ChatClovaX features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.naver.ChatClovaX.html).

[CLOVA Studio](http://clovastudio.ncloud.com/) has several chat models. You can find information about latest models and their costs, context windows, and supported input types in the CLOVA Studio API Guide [documentation](https://api.ncloud-docs.com/docs/clovastudio-chatcompletions).

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- |:-----:| :---: |:------------------------------------------------------------------------:| :---: | :---: |
| [ChatClovaX](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.naver.ChatClovaX.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) |   ❌   | ❌ |                                    ❌                                     | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_community?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
|:------------------------------------------:| :---: | :---: | :---: |  :---: | :---: |:-----------------------------------------------------:| :---: |:------------------------------------------------------:|:----------------------------------:|
|❌| ❌ | ❌ | ❌ | ❌ | ❌ |                          ✅                            | ✅ |                           ✅                            |                 ❌                  | 

## Setup

Before using the chat model, you must go through the three steps below.

1. Creating [NAVER Cloud Platform](https://www.ncloud.com/) account 
2. Apply to use [CLOVA Studio](https://www.ncloud.com/product/aiService/clovaStudio)
3. Find API Keys after creating CLOVA Studio Test App or Service App (See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#테스트앱생성).)

### Credentials

CLOVA Studio requires 2 keys (`NCP_CLOVASTUDIO_API_KEY` and `NCP_APIGW_API_KEY`).
  - `NCP_CLOVASTUDIO_API_KEY` is issued per Test App or Service App
  - `NCP_APIGW_API_KEY` is issued per account, could be optional depending on the region you are using

The two API Keys could be found by clicking `App Request Status` > `Service App, Test App List` > `‘Details’ button for each app` in [CLOVA Studio](https://clovastudio.ncloud.com/studio-application/service-app)

You can add them to your environment variables as below:

``` bash
export NCP_CLOVASTUDIO_API_KEY="your-api-key-here"
export NCP_APIGW_API_KEY="your-api-key-here"
```


```python
import getpass
import os

if not os.getenv("NCP_CLOVASTUDIO_API_KEY"):
    os.environ["NCP_CLOVASTUDIO_API_KEY"] = getpass.getpass(
        "Enter your NCP CLOVA Studio API Key: "
    )
if not os.getenv("NCP_APIGW_API_KEY"):
    os.environ["NCP_APIGW_API_KEY"] = getpass.getpass(
        "Enter your NCP API Gateway API key: "
    )
```

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
```

### Installation

The LangChain Naver integration lives in the `langchain-community` package:


```python
# install package
!pip install -qU langchain-community
```

## Instantiation

Now we can instantiate our model object and generate chat completions:


```python
from langchain_community.chat_models import ChatClovaX

chat = ChatClovaX(
    model="HCX-003",
    max_tokens=100,
    temperature=0.5,
    # clovastudio_api_key="..."    # set if you prefer to pass api key directly instead of using environment variables
    # task_id="..."    # set if you want to use fine-tuned model
    # service_app=False    # set True if using Service App. Default value is False (means using Test App)
    # include_ai_filters=False     # set True if you want to detect inappropriate content. Default value is False
    # other params...
)
```

## Invocation

In addition to invoke, we also support batch and stream functionalities.


```python
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Korean. Translate the user sentence.",
    ),
    ("human", "I love using NAVER AI."),
]

ai_msg = chat.invoke(messages)
ai_msg
```




    AIMessage(content='저는 네이버 AI를 사용하는 것이 좋아요.', additional_kwargs={}, response_metadata={'stop_reason': 'stop_before', 'input_length': 25, 'output_length': 14, 'seed': 1112164354, 'ai_filter': None}, id='run-b57bc356-1148-4007-837d-cc409dbd57cc-0', usage_metadata={'input_tokens': 25, 'output_tokens': 14, 'total_tokens': 39})




```python
print(ai_msg.content)
```

    저는 네이버 AI를 사용하는 것이 좋아요.
    

## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:


```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}. Translate the user sentence.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | chat
chain.invoke(
    {
        "input_language": "English",
        "output_language": "Korean",
        "input": "I love using NAVER AI.",
    }
)
```




    AIMessage(content='저는 네이버 AI를 사용하는 것이 좋아요.', additional_kwargs={}, response_metadata={'stop_reason': 'stop_before', 'input_length': 25, 'output_length': 14, 'seed': 2575184681, 'ai_filter': None}, id='run-7014b330-eba3-4701-bb62-df73ce39b854-0', usage_metadata={'input_tokens': 25, 'output_tokens': 14, 'total_tokens': 39})



## Streaming


```python
system = "You are a helpful assistant that can teach Korean pronunciation."
human = "Could you let me know how to say '{phrase}' in Korean?"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

for chunk in chain.stream({"phrase": "Hi"}):
    print(chunk.content, end="", flush=True)
```

    Certainly! In Korean, "Hi" is pronounced as "안녕" (annyeong). The first syllable, "안," sounds like the "ahh" sound in "apple," while the second syllable, "녕," sounds like the "yuh" sound in "you." So when you put them together, it's like saying "ahhyuh-nyuhng." Remember to pronounce each syllable clearly and separately for accurate pronunciation.

## Additional functionalities

### Using fine-tuned models

You can call fine-tuned models by passing in your corresponding `task_id` parameter. (You don’t need to specify the `model_name` parameter when calling fine-tuned model.)

You can check `task_id` from corresponding Test App or Service App details.


```python
fine_tuned_model = ChatClovaX(
    task_id="5s8egt3a",  # set if you want to use fine-tuned model
    # other params...
)

fine_tuned_model.invoke(messages)
```




    AIMessage(content='저는 네이버 AI를 사용하는 것이 너무 좋아요.', additional_kwargs={}, response_metadata={'stop_reason': 'stop_before', 'input_length': 25, 'output_length': 15, 'seed': 52559061, 'ai_filter': None}, id='run-5bea8d4a-48f3-4c34-ae70-66e60dca5344-0', usage_metadata={'input_tokens': 25, 'output_tokens': 15, 'total_tokens': 40})



### Service App

When going live with production-level application using CLOVA Studio, you should apply for and use Service App. (See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#서비스앱신청).)

For a Service App, a corresponding `NCP_CLOVASTUDIO_API_KEY` is issued and can only be called with it.


```python
# Update environment variables

os.environ["NCP_CLOVASTUDIO_API_KEY"] = getpass.getpass(
    "Enter NCP CLOVA Studio API Key for Service App: "
)
```


```python
chat = ChatClovaX(
    service_app=True,  # True if you want to use your service app, default value is False.
    # clovastudio_api_key="..."  # if you prefer to pass api key in directly instead of using env vars
    model="HCX-003",
    # other params...
)
ai_msg = chat.invoke(messages)
```

### AI Filter

AI Filter detects inappropriate output such as profanity from the test app (or service app included) created in Playground and informs the user. See [here](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#AIFilter) for details. 


```python
chat = ChatClovaX(
    model="HCX-003",
    include_ai_filters=True,  # True if you want to enable ai filter
    # other params...
)

ai_msg = chat.invoke(messages)
```


```python
print(ai_msg.response_metadata["ai_filter"])
```

## API reference

For detailed documentation of all ChatNaver features and configurations head to the API reference: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.naver.ChatClovaX.html
