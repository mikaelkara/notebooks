---
sidebar_label: Baichuan Chat
---
# Chat with Baichuan-192K

Baichuan chat models API by Baichuan Intelligent Technology. For more information, see [https://platform.baichuan-ai.com/docs/api](https://platform.baichuan-ai.com/docs/api)


```python
from langchain_community.chat_models import ChatBaichuan
from langchain_core.messages import HumanMessage
```


```python
chat = ChatBaichuan(baichuan_api_key="YOUR_API_KEY")
```

Alternatively, you can set your API key with:


```python
import os

os.environ["BAICHUAN_API_KEY"] = "YOUR_API_KEY"
```


```python
chat([HumanMessage(content="我日薪8块钱，请问在闰年的二月，我月薪多少")])
```




    AIMessage(content='首先，我们需要确定闰年的二月有多少天。闰年的二月有29天。\n\n然后，我们可以计算你的月薪：\n\n日薪 = 月薪 / (当月天数)\n\n所以，你的月薪 = 日薪 * 当月天数\n\n将数值代入公式：\n\n月薪 = 8元/天 * 29天 = 232元\n\n因此，你在闰年的二月的月薪是232元。')



## Chat with Baichuan-192K with Streaming


```python
chat = ChatBaichuan(
    baichuan_api_key="YOUR_API_KEY",
    streaming=True,
)
```


```python
chat([HumanMessage(content="我日薪8块钱，请问在闰年的二月，我月薪多少")])
```




    AIMessageChunk(content='首先，我们需要确定闰年的二月有多少天。闰年的二月有29天。\n\n然后，我们可以计算你的月薪：\n\n日薪 = 月薪 / (当月天数)\n\n所以，你的月薪 = 日薪 * 当月天数\n\n将数值代入公式：\n\n月薪 = 8元/天 * 29天 = 232元\n\n因此，你在闰年的二月的月薪是232元。')


