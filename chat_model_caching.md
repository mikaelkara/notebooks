# How to cache chat model responses

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/chat_models)
- [LLMs](/docs/concepts/text_llms)

:::

LangChain provides an optional caching layer for chat models. This is useful for two main reasons:

- It can save you money by reducing the number of API calls you make to the LLM provider, if you're often requesting the same completion multiple times. This is especially useful during app development.
- It can speed up your application by reducing the number of API calls you make to the LLM provider.

This guide will walk you through how to enable this in your apps.

import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm" />



```python
# | output: false
# | echo: false

import os
from getpass import getpass

from langchain_openai import ChatOpenAI

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOpenAI()
```


```python
# <!-- ruff: noqa: F821 -->
from langchain_core.globals import set_llm_cache
```

## In Memory Cache

This is an ephemeral cache that stores model calls in memory. It will be wiped when your environment restarts, and is not shared across processes.


```python
%%time
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
llm.invoke("Tell me a joke")
```

    CPU times: user 645 ms, sys: 214 ms, total: 859 ms
    Wall time: 829 ms
    




    AIMessage(content="Why don't scientists trust atoms?\n\nBecause they make up everything!", response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 11, 'total_tokens': 24}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-b6836bdd-8c30-436b-828f-0ac5fc9ab50e-0')




```python
%%time
# The second time it is, so it goes faster
llm.invoke("Tell me a joke")
```

    CPU times: user 822 µs, sys: 288 µs, total: 1.11 ms
    Wall time: 1.06 ms
    




    AIMessage(content="Why don't scientists trust atoms?\n\nBecause they make up everything!", response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 11, 'total_tokens': 24}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-b6836bdd-8c30-436b-828f-0ac5fc9ab50e-0')



## SQLite Cache

This cache implementation uses a `SQLite` database to store responses, and will last across process restarts.


```python
!rm .langchain.db
```


```python
# We can do the same thing with a SQLite cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```


```python
%%time
# The first time, it is not yet in cache, so it should take longer
llm.invoke("Tell me a joke")
```

    CPU times: user 9.91 ms, sys: 7.68 ms, total: 17.6 ms
    Wall time: 657 ms
    




    AIMessage(content='Why did the scarecrow win an award? Because he was outstanding in his field!', response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 11, 'total_tokens': 28}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-39d9e1e8-7766-4970-b1d8-f50213fd94c5-0')




```python
%%time
# The second time it is, so it goes faster
llm.invoke("Tell me a joke")
```

    CPU times: user 52.2 ms, sys: 60.5 ms, total: 113 ms
    Wall time: 127 ms
    




    AIMessage(content='Why did the scarecrow win an award? Because he was outstanding in his field!', id='run-39d9e1e8-7766-4970-b1d8-f50213fd94c5-0')



## Next steps

You've now learned how to cache model responses to save time and money.

Next, check out the other how-to guides chat models in this section, like [how to get a model to return structured output](/docs/how_to/structured_output) or [how to create your own custom chat model](/docs/how_to/custom_chat_model).
