---
sidebar_label: In-memory
---
# InMemoryByteStore

This guide will help you get started with in-memory [key-value stores](/docs/concepts/key_value_stores). For detailed documentation of all `InMemoryByteStore` features and configurations head to the [API reference](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html).

## Overview

The `InMemoryByteStore` is a non-persistent implementation of a `ByteStore` that stores everything in a Python dictionary. It's intended for demos and cases where you don't need persistence past the lifetime of the Python process.

### Integration details

| Class | Package | Local | [JS support](https://js.langchain.com/docs/integrations/stores/in_memory/) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: |
| [InMemoryByteStore](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html) | [langchain_core](https://python.langchain.com/api_reference/core/index.html) | ✅ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_core?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_core?style=flat-square&label=%20) |

### Installation

The LangChain `InMemoryByteStore` integration lives in the `langchain_core` package:


```python
%pip install -qU langchain_core
```

## Instantiation

Now you can instantiate your byte store:


```python
from langchain_core.stores import InMemoryByteStore

kv_store = InMemoryByteStore()
```

## Usage

You can set data under keys like this using the `mset` method:


```python
kv_store.mset(
    [
        ["key1", b"value1"],
        ["key2", b"value2"],
    ]
)

kv_store.mget(
    [
        "key1",
        "key2",
    ]
)
```




    [b'value1', b'value2']



And you can delete data using the `mdelete` method:


```python
kv_store.mdelete(
    [
        "key1",
        "key2",
    ]
)

kv_store.mget(
    [
        "key1",
        "key2",
    ]
)
```




    [None, None]



## API reference

For detailed documentation of all `InMemoryByteStore` features and configurations, head to the API reference: https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryByteStore.html
