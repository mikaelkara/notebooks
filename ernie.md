---
sidebar_label: Ernie Bot Chat
---
# ErnieBotChat

[ERNIE-Bot](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11) is a large language model developed by Baidu, covering a huge amount of Chinese data.
This notebook covers how to get started with ErnieBot chat models.

**Deprecated Warning**

We recommend users using `langchain_community.chat_models.ErnieBotChat` 
to use `langchain_community.chat_models.QianfanChatEndpoint` instead.

documentation for `QianfanChatEndpoint` is [here](/docs/integrations/chat/baidu_qianfan_endpoint/).

they are 4 why we recommend users to use `QianfanChatEndpoint`:

1. `QianfanChatEndpoint` support more LLM in the Qianfan platform.
2. `QianfanChatEndpoint` support streaming mode.
3. `QianfanChatEndpoint` support function calling usgage.
4. `ErnieBotChat` is lack of maintenance and deprecated.

Some tips for migration:

- change `ernie_client_id` to `qianfan_ak`, also change `ernie_client_secret` to `qianfan_sk`.
- install `qianfan` package. like `pip install qianfan`
- change `ErnieBotChat` to `QianfanChatEndpoint`.


```python
from langchain_community.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint

chat = QianfanChatEndpoint(
    qianfan_ak="your qianfan ak",
    qianfan_sk="your qianfan sk",
)
```

## Usage


```python
from langchain_community.chat_models import ErnieBotChat
from langchain_core.messages import HumanMessage

chat = ErnieBotChat(
    ernie_client_id="YOUR_CLIENT_ID", ernie_client_secret="YOUR_CLIENT_SECRET"
)
```

or you can set `client_id` and `client_secret` in your environment variables
```bash
export ERNIE_CLIENT_ID=YOUR_CLIENT_ID
export ERNIE_CLIENT_SECRET=YOUR_CLIENT_SECRET
```


```python
chat([HumanMessage(content="hello there, who are you?")])
```




    AIMessage(content='Hello, I am an artificial intelligence language model. My purpose is to help users answer questions or provide information. What can I do for you?', additional_kwargs={}, example=False)


