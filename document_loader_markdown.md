# How to load Markdown

[Markdown](https://en.wikipedia.org/wiki/Markdown) is a lightweight markup language for creating formatted text using a plain-text editor.

Here we cover how to load `Markdown` documents into LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) objects that we can use downstream.

We will cover:

- Basic usage;
- Parsing of Markdown into elements such as titles, list items, and text.

LangChain implements an [UnstructuredMarkdownLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.markdown.UnstructuredMarkdownLoader.html) object which requires the [Unstructured](https://unstructured-io.github.io/unstructured/) package. First we install it:


```python
%pip install "unstructured[md]" nltk
```

Basic usage will ingest a Markdown file to a single document. Here we demonstrate on LangChain's readme:


```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "../../../README.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
assert len(data) == 1
assert isinstance(data[0], Document)
readme_content = data[0].page_content
print(readme_content[:250])
```

    🦜️🔗 LangChain
    
    ⚡ Build context-aware reasoning applications ⚡
    
    Looking for the JS/TS library? Check out LangChain.js.
    
    To help you ship LangChain apps to production faster, check out LangSmith. 
    LangSmith is a unified developer platform for building,
    

## Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.


```python
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")

data = loader.load()
print(f"Number of documents: {len(data)}\n")

for document in data[:2]:
    print(f"{document}\n")
```

    Number of documents: 66
    
    page_content='🦜️🔗 LangChain' metadata={'source': '../../../README.md', 'category_depth': 0, 'last_modified': '2024-06-28T15:20:01', 'languages': ['eng'], 'filetype': 'text/markdown', 'file_directory': '../../..', 'filename': 'README.md', 'category': 'Title'}
    
    page_content='⚡ Build context-aware reasoning applications ⚡' metadata={'source': '../../../README.md', 'last_modified': '2024-06-28T15:20:01', 'languages': ['eng'], 'parent_id': '200b8a7d0dd03f66e4f13456566d2b3a', 'filetype': 'text/markdown', 'file_directory': '../../..', 'filename': 'README.md', 'category': 'NarrativeText'}
    
    

Note that in this case we recover three distinct element types:


```python
print(set(document.metadata["category"] for document in data))
```

    {'ListItem', 'NarrativeText', 'Title'}
    


```python

```
