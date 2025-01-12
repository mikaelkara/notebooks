# How to cache LLM responses

LangChain provides an optional caching layer for LLMs. This is useful for two reasons:

It can save you money by reducing the number of API calls you make to the LLM provider, if you're often requesting the same completion multiple times.
It can speed up your application by reducing the number of API calls you make to the LLM provider.



```python
%pip install -qU langchain_openai langchain_community

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()
# Please manually enter OpenAI Key
```


```python
from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAI

# To make the caching really obvious, lets use a slower and older model.
# Caching supports newer chat models as well.
llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)
```


```python
%%time
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
llm.invoke("Tell me a joke")
```

    CPU times: user 546 ms, sys: 379 ms, total: 925 ms
    Wall time: 1.11 s
    




    "\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"




```python
%%time
# The second time it is, so it goes faster
llm.invoke("Tell me a joke")
```

    CPU times: user 192 µs, sys: 77 µs, total: 269 µs
    Wall time: 270 µs
    




    "\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"



## SQLite Cache


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

    CPU times: user 10.6 ms, sys: 4.21 ms, total: 14.8 ms
    Wall time: 851 ms
    




    "\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"




```python
%%time
# The second time it is, so it goes faster
llm.invoke("Tell me a joke")
```

    CPU times: user 59.7 ms, sys: 63.6 ms, total: 123 ms
    Wall time: 134 ms
    




    "\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything!"




```python

```
