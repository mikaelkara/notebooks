# Open Document Format (ODT)

>The [Open Document Format for Office Applications (ODF)](https://en.wikipedia.org/wiki/OpenDocument), also known as `OpenDocument`, is an open file format for word processing documents, spreadsheets, presentations and graphics and using ZIP-compressed XML files. It was developed with the aim of providing an open, XML-based file format specification for office applications.

>The standard is developed and maintained by a technical committee in the Organization for the Advancement of Structured Information Standards (`OASIS`) consortium. It was based on the Sun Microsystems specification for OpenOffice.org XML, the default format for `OpenOffice.org` and `LibreOffice`. It was originally developed for `StarOffice` "to provide an open standard for office documents."

The `UnstructuredODTLoader` is used to load `Open Office ODT` files.


```python
from langchain_community.document_loaders import UnstructuredODTLoader

loader = UnstructuredODTLoader("example_data/fake.odt", mode="elements")
docs = loader.load()
docs[0]
```




    Document(page_content='Lorem ipsum dolor sit amet.', metadata={'source': 'example_data/fake.odt', 'category_depth': 0, 'file_directory': 'example_data', 'filename': 'fake.odt', 'last_modified': '2023-12-19T13:42:18', 'languages': ['por', 'cat'], 'filetype': 'application/vnd.oasis.opendocument.text', 'category': 'Title'})




```python

```
