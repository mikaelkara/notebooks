---
sidebar_label: LiteLLM
---
# ChatLiteLLM

[LiteLLM](https://github.com/BerriAI/litellm) is a library that simplifies calling Anthropic, Azure, Huggingface, Replicate, etc. 

This notebook covers how to get started with using Langchain + the LiteLLM I/O library. 


```python
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.messages import HumanMessage
```


```python
chat = ChatLiteLLM(model="gpt-3.5-turbo")
```


```python
messages = [
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
]
chat(messages)
```




    AIMessage(content=" J'aime la programmation.", additional_kwargs={}, example=False)



## `ChatLiteLLM` also supports async and streaming functionality:


```python
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
```


```python
await chat.agenerate([messages])
```




    LLMResult(generations=[[ChatGeneration(text=" J'aime programmer.", generation_info=None, message=AIMessage(content=" J'aime programmer.", additional_kwargs={}, example=False))]], llm_output={}, run=[RunInfo(run_id=UUID('8cc8fb68-1c35-439c-96a0-695036a93652'))])




```python
chat = ChatLiteLLM(
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
chat(messages)
```

     J'aime la programmation.




    AIMessage(content=" J'aime la programmation.", additional_kwargs={}, example=False)




```python

```
