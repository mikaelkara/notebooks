---
sidebar_label: Box
---

# BoxLoader

This notebook provides a quick overview for getting started with Box [document loader](/docs/integrations/document_loaders/). For detailed documentation of all BoxLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.document_loaders.box.BoxLoader.html).


## Overview

The `BoxLoader` class helps you get your unstructured content from Box in Langchain's `Document` format. You can do this with either a `List[str]` containing Box file IDs, or with a `str` containing a Box folder ID. 

You must provide either a `List[str]` containing Box file Ids, or a `str` containing a folder ID. If getting files from a folder with folder ID, you can also set a `Bool` to tell the loader to get all sub-folders in that folder, as well. 

:::info
A Box instance can contain Petabytes of files, and folders can contain millions of files. Be intentional when choosing what folders you choose to index. And we recommend never getting all files from folder 0 recursively. Folder ID 0 is your root folder.
:::

Files without a text representation will be skipped.

### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [BoxLoader](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.document_loaders.box.BoxLoader.html) | [langchain_box](https://python.langchain.com/api_reference/box/index.html) | ✅ | ❌ | ❌ | 
### Loader features
| Source | Document Lazy Loading | Async Support
| :---: | :---: | :---: | 
| BoxLoader | ✅ | ❌ | 

## Setup

In order to use the Box package, you will need a few things:

* A Box account — If you are not a current Box customer or want to test outside of your production Box instance, you can use a [free developer account](https://account.box.com/signup/n/developer#ty9l3).
* [A Box app](https://developer.box.com/guides/getting-started/first-application/) — This is configured in the [developer console](https://account.box.com/developers/console), and for Box AI, must have the `Manage AI` scope enabled. Here you will also select your authentication method
* The app must be [enabled by the administrator](https://developer.box.com/guides/authorization/custom-app-approval/#manual-approval). For free developer accounts, this is whomever signed up for the account.

### Credentials

For these examples, we will use [token authentication](https://developer.box.com/guides/authentication/tokens/developer-tokens). This can be used with any [authentication method](https://developer.box.com/guides/authentication/). Just get the token with whatever methodology. If you want to learn more about how to use other authentication types with `langchain-box`, visit the [Box provider](/docs/integrations/providers/box) document.



```python
import getpass
import os

box_developer_token = getpass.getpass("Enter your Box Developer Token: ")
```

    Enter your Box Developer Token:  ········
    

If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

Install **langchain_box**.


```python
%pip install -qU langchain_box
```

## Initialization

### Load files

If you wish to load files, you must provide the `List` of file ids at instantiation time. 

This requires 1 piece of information:

* **box_file_ids** (`List[str]`)- A list of Box file IDs. 


```python
from langchain_box.document_loaders import BoxLoader

box_file_ids = ["1514555423624", "1514553902288"]

loader = BoxLoader(
    box_developer_token=box_developer_token,
    box_file_ids=box_file_ids,
    character_limit=10000,  # Optional. Defaults to no limit
)
```

### Load from folder

If you wish to load files from a folder, you must provide a `str` with the Box folder ID at instantiation time. 

This requires 1 piece of information:

* **box_folder_id** (`str`)- A string containing a Box folder ID.  


```python
from langchain_box.document_loaders import BoxLoader

box_folder_id = "260932470532"

loader = BoxLoader(
    box_folder_id=box_folder_id,
    recursive=False,  # Optional. return entire tree, defaults to False
    character_limit=10000,  # Optional. Defaults to no limit
)
```

## Load


```python
docs = loader.load()
docs[0]
```




    Document(metadata={'source': 'https://dl.boxcloud.com/api/2.0/internal_files/1514555423624/versions/1663171610024/representations/extracted_text/content/', 'title': 'Invoice-A5555_txt'}, page_content='Vendor: AstroTech Solutions\nInvoice Number: A5555\n\nLine Items:\n    - Gravitational Wave Detector Kit: $800\n    - Exoplanet Terrarium: $120\nTotal: $920')




```python
print(docs[0].metadata)
```

    {'source': 'https://dl.boxcloud.com/api/2.0/internal_files/1514555423624/versions/1663171610024/representations/extracted_text/content/', 'title': 'Invoice-A5555_txt'}
    

## Lazy Load


```python
page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []
```

## API reference

For detailed documentation of all BoxLoader features and configurations head to the [API reference](https://python.langchain.com/api_reference/box/document_loaders/langchain_box.document_loaders.box.BoxLoader.html)


## Help

If you have questions, you can check out our [developer documentation](https://developer.box.com) or reach out to use in our [developer community](https://community.box.com).


