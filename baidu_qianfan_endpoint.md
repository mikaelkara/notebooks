---
sidebar_label: Baidu Qianfan
---
# QianfanChatEndpoint

Baidu AI Cloud Qianfan Platform is a one-stop large model development and service operation platform for enterprise developers. Qianfan not only provides including the model of Wenxin Yiyan (ERNIE-Bot) and the third-party open-source models, but also provides various AI development tools and the whole set of development environment, which facilitates customers to use and develop large model applications easily.

Basically, those model are split into the following type:

- Embedding
- Chat
- Completion

In this notebook, we will introduce how to use langchain with [Qianfan](https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html) mainly in `Chat` corresponding
 to the package `langchain/chat_models` in langchain:


## API Initialization

To use the LLM services based on Baidu Qianfan, you have to initialize these parameters:

You could either choose to init the AK,SK in environment variables or init params:

```base
export QIANFAN_AK=XXX
export QIANFAN_SK=XXX
```

## Current supported models:

- ERNIE-Bot-turbo (default models)
- ERNIE-Bot
- BLOOMZ-7B
- Llama-2-7b-chat
- Llama-2-13b-chat
- Llama-2-70b-chat
- Qianfan-BLOOMZ-7B-compressed
- Qianfan-Chinese-Llama-2-7B
- ChatGLM2-6B-32K
- AquilaChat-7B

## Set up


```python
"""For basic init and call"""
import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage

os.environ["QIANFAN_AK"] = "Your_api_key"
os.environ["QIANFAN_SK"] = "You_secret_Key"
```

## Usage


```python
chat = QianfanChatEndpoint(streaming=True)
messages = [HumanMessage(content="Hello")]
chat.invoke(messages)
```




    AIMessage(content='您好！请问您需要什么帮助？我将尽力回答您的问题。')




```python
await chat.ainvoke(messages)
```




    AIMessage(content='您好！有什么我可以帮助您的吗？')




```python
chat.batch([messages])
```




    [AIMessage(content='您好！有什么我可以帮助您的吗？')]



### Streaming


```python
try:
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
except TypeError as e:
    print("")
```

    您好！有什么我可以帮助您的吗？
    

## Use different models in Qianfan

The default model is ERNIE-Bot-turbo, in the case you want to deploy your own model based on Ernie Bot or third-party open-source model, you could follow these steps:

1. (Optional, if the model are included in the default models, skip it) Deploy your model in Qianfan Console, get your own customized deploy endpoint.
2. Set up the field called `endpoint` in the initialization:


```python
chatBot = QianfanChatEndpoint(
    streaming=True,
    model="ERNIE-Bot",
)

messages = [HumanMessage(content="Hello")]
chatBot.invoke(messages)
```




    AIMessage(content='Hello，可以回答问题了，我会竭尽全力为您解答，请问有什么问题吗？')



## Model Params:

For now, only `ERNIE-Bot` and `ERNIE-Bot-turbo` support model params below, we might support more models in the future.

- temperature
- top_p
- penalty_score



```python
chat.invoke(
    [HumanMessage(content="Hello")],
    **{"top_p": 0.4, "temperature": 0.1, "penalty_score": 1},
)
```




    AIMessage(content='您好！有什么我可以帮助您的吗？')


