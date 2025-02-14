# BSHTMLLoader


This notebook provides a quick overview for getting started with BeautifulSoup4 [document loader](https://python.langchain.com/docs/concepts/document_loaders). For detailed documentation of all __ModuleName__Loader features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html).


## Overview
### Integration details


| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [BSHTMLLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| BSHTMLLoader | ✅ | ❌ | 

## Setup

To access BSHTMLLoader document loader you'll need to install the `langchain-community` integration package and the `bs4` python package.

### Credentials

No credentials are needed to use the `BSHTMLLoader` class.

If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

Install **langchain_community** and **bs4**.


```python
%pip install -qU langchain_community bs4
```

## Initialization

Now we can instantiate our model object and load documents:

- TODO: Update model instantiation with relevant params.


```python
from langchain_community.document_loaders import BSHTMLLoader

loader = BSHTMLLoader(
    file_path="./example_data/fake-content.html",
)
```

## Load


```python
docs = loader.load()
docs[0]
```




    Document(metadata={'source': './example_data/fake-content.html', 'title': 'Test Title'}, page_content='\nTest Title\n\n\nMy First Heading\nMy first paragraph.\n\n\n')




```python
print(docs[0].metadata)
```

    {'source': './example_data/fake-content.html', 'title': 'Test Title'}
    

## Lazy Load


```python
page = []
for doc in loader.lazy_load():
    page.append(doc)
    if len(page) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        page = []
page[0]
```




    Document(metadata={'source': './example_data/fake-content.html', 'title': 'Test Title'}, page_content='\nTest Title\n\n\nMy First Heading\nMy first paragraph.\n\n\n')



## Adding separator to BS4

We can also pass a separator to use when calling get_text on the soup


```python
loader = BSHTMLLoader(
    file_path="./example_data/fake-content.html", get_text_separator=", "
)

docs = loader.load()
print(docs[0])
```

    page_content='
    , Test Title, 
    , 
    , 
    , My First Heading, 
    , My first paragraph., 
    , 
    , 
    ' metadata={'source': './example_data/fake-content.html', 'title': 'Test Title'}
    

## API reference

For detailed documentation of all BSHTMLLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.html_bs.BSHTMLLoader.html
