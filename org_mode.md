# Org-mode

>A [Org Mode document](https://en.wikipedia.org/wiki/Org-mode) is a document editing, formatting, and organizing mode, designed for notes, planning, and authoring within the free software text editor Emacs.

## `UnstructuredOrgModeLoader`

You can load data from Org-mode files with `UnstructuredOrgModeLoader` using the following workflow.


```python
from langchain_community.document_loaders import UnstructuredOrgModeLoader

loader = UnstructuredOrgModeLoader(
    file_path="./example_data/README.org", mode="elements"
)
docs = loader.load()

print(docs[0])
```

    page_content='Example Docs' metadata={'source': './example_data/README.org', 'category_depth': 0, 'last_modified': '2023-12-19T13:42:18', 'languages': ['eng'], 'filetype': 'text/org', 'file_directory': './example_data', 'filename': 'README.org', 'category': 'Title'}
    


```python

```
