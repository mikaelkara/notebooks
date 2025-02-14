# Dedoc

This sample demonstrates the use of `Dedoc` in combination with `LangChain` as a `DocumentLoader`.

## Overview

[Dedoc](https://dedoc.readthedocs.io) is an [open-source](https://github.com/ispras/dedoc)
library/service that extracts texts, tables, attached files and document structure
(e.g., titles, list items, etc.) from files of various formats.

`Dedoc` supports `DOCX`, `XLSX`, `PPTX`, `EML`, `HTML`, `PDF`, images and more.
Full list of supported formats can be found [here](https://dedoc.readthedocs.io/en/latest/#id1).


### Integration details

| Class                                                                                                                                                | Package                                                                                        | Local | Serializable | JS support |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|:-----:|:------------:|:----------:|
| [DedocFileLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocFileLoader.html)       | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |     beta     |     ❌      |
| [DedocPDFLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.DedocPDFLoader.html)           | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |     beta     |     ❌      | 
| [DedocAPIFileLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocAPIFileLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) |   ❌   |     beta     |     ❌      | 


### Loader features

Methods for lazy loading and async loading are available, but in fact, document loading is executed synchronously.

|       Source       | Document Lazy Loading | Async Support |
|:------------------:|:---------------------:|:-------------:| 
|  DedocFileLoader   |           ❌           |       ❌       |
|   DedocPDFLoader   |           ❌           |       ❌       | 
| DedocAPIFileLoader |           ❌           |       ❌       | 

## Setup

* To access `DedocFileLoader` and `DedocPDFLoader` document loaders, you'll need to install the `dedoc` integration package.
* To access `DedocAPIFileLoader`, you'll need to run the `Dedoc` service, e.g. `Docker` container (please see [the documentation](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html#install-and-run-dedoc-using-docker) 
for more details):

```bash
docker pull dedocproject/dedoc
docker run -p 1231:1231
```

`Dedoc` installation instruction is given [here](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html).


```python
# Install package
%pip install --quiet "dedoc[torch]"
```

    Note: you may need to restart the kernel to use updated packages.
    

## Instantiation


```python
from langchain_community.document_loaders import DedocFileLoader

loader = DedocFileLoader("./example_data/state_of_the_union.txt")
```

## Load


```python
docs = loader.load()
docs[0].page_content[:100]
```




    '\nMadam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and t'



## Lazy Load


```python
docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content[:100])
    break
```

    
    Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and t
    

## API reference

For detailed information on configuring and calling `Dedoc` loaders, please see the API references: 

* https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocFileLoader.html
* https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.DedocPDFLoader.html
* https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.dedoc.DedocAPIFileLoader.html

## Loading any file

For automatic handling of any file in a [supported format](https://dedoc.readthedocs.io/en/latest/#id1),
`DedocFileLoader` can be useful.
The file loader automatically detects the file type with a correct extension.

File parsing process can be configured through `dedoc_kwargs` during the `DedocFileLoader` class initialization.
Here the basic examples of some options usage are given, 
please see the documentation of `DedocFileLoader` and 
[dedoc documentation](https://dedoc.readthedocs.io/en/latest/parameters/parameters.html) 
to get more details about configuration parameters.

### Basic example


```python
from langchain_community.document_loaders import DedocFileLoader

loader = DedocFileLoader("./example_data/state_of_the_union.txt")

docs = loader.load()

docs[0].page_content[:400]
```




    '\nMadam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\n\n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\n\n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\n\n\nWith a duty to one another to the American people to '



### Modes of split

`DedocFileLoader` supports different types of document splitting into parts (each part is returned separately).
For this purpose, `split` parameter is used with the following options:
* `document` (default value): document text is returned as a single langchain `Document` object (don't split);
* `page`: split document text into pages (works for `PDF`, `DJVU`, `PPTX`, `PPT`, `ODP`);
* `node`: split document text into `Dedoc` tree nodes (title nodes, list item nodes, raw text nodes);
* `line`: split document text into textual lines.


```python
loader = DedocFileLoader(
    "./example_data/layout-parser-paper.pdf",
    split="page",
    pages=":2",
)

docs = loader.load()

len(docs)
```




    2



### Handling tables

`DedocFileLoader` supports tables handling when `with_tables` parameter is 
set to `True` during loader initialization (`with_tables=True` by default). 

Tables are not split - each table corresponds to one langchain `Document` object.
For tables, `Document` object has additional `metadata` fields `type="table"` 
and `text_as_html` with table `HTML` representation.


```python
loader = DedocFileLoader("./example_data/mlb_teams_2012.csv")

docs = loader.load()

docs[1].metadata["type"], docs[1].metadata["text_as_html"][:200]
```




    ('table',
     '<table border="1" style="border-collapse: collapse; width: 100%;">\n<tbody>\n<tr>\n<td colspan="1" rowspan="1">Team</td>\n<td colspan="1" rowspan="1"> &quot;Payroll (millions)&quot;</td>\n<td colspan="1" r')



### Handling attached files

`DedocFileLoader` supports attached files handling when `with_attachments` is set 
to `True` during loader initialization (`with_attachments=False` by default). 

Attachments are split according to the `split` parameter.
For attachments, langchain `Document` object has an additional metadata 
field `type="attachment"`.


```python
loader = DedocFileLoader(
    "./example_data/fake-email-attachment.eml",
    with_attachments=True,
)

docs = loader.load()

docs[1].metadata["type"], docs[1].page_content
```




    ('attachment',
     '\nContent-Type\nmultipart/mixed; boundary="0000000000005d654405f082adb7"\nDate\nFri, 23 Dec 2022 12:08:48 -0600\nFrom\nMallori Harrell <mallori@unstructured.io>\nMIME-Version\n1.0\nMessage-ID\n<CAPgNNXSzLVJ-d1OCX_TjFgJU7ugtQrjFybPtAMmmYZzphxNFYg@mail.gmail.com>\nSubject\nFake email with attachment\nTo\nMallori Harrell <mallori@unstructured.io>')



## Loading PDF file

If you want to handle only `PDF` documents, you can use `DedocPDFLoader` with only `PDF` support.
The loader supports the same parameters for document split, tables and attachments extraction.

`Dedoc` can extract `PDF` with or without a textual layer, 
as well as automatically detect its presence and correctness.
Several `PDF` handlers are available, you can use `pdf_with_text_layer` 
parameter to choose one of them.
Please see [parameters description](https://dedoc.readthedocs.io/en/latest/parameters/pdf_handling.html) 
to get more details.

For `PDF` without a textual layer, `Tesseract OCR` and its language packages should be installed.
In this case, [the instruction](https://dedoc.readthedocs.io/en/latest/tutorials/add_new_language.html) can be useful.


```python
from langchain_community.document_loaders import DedocPDFLoader

loader = DedocPDFLoader(
    "./example_data/layout-parser-paper.pdf", pdf_with_text_layer="true", pages="2:2"
)

docs = loader.load()

docs[0].page_content[:400]
```




    '\n2\n\nZ. Shen et al.\n\n37], layout detection [38, 22], table detection [26], and scene text detection [4].\n\nA generalized learning-based framework dramatically reduces the need for the\n\nmanual speciﬁcation of complicated rules, which is the status quo with traditional\n\nmethods. DL has the potential to transform DIA pipelines and beneﬁt a broad\n\nspectrum of large-scale document digitization projects.\n'



## Dedoc API

If you want to get up and running with less set up, you can use `Dedoc` as a service.
**`DedocAPIFileLoader` can be used without installation of `dedoc` library.**
The loader supports the same parameters as `DedocFileLoader` and
also automatically detects input file types.

To use `DedocAPIFileLoader`, you should run the `Dedoc` service, e.g. `Docker` container (please see [the documentation](https://dedoc.readthedocs.io/en/latest/getting_started/installation.html#install-and-run-dedoc-using-docker) 
for more details):

```bash
docker pull dedocproject/dedoc
docker run -p 1231:1231
```

Please do not use our demo URL `https://dedoc-readme.hf.space` in your code.


```python
from langchain_community.document_loaders import DedocAPIFileLoader

loader = DedocAPIFileLoader(
    "./example_data/state_of_the_union.txt",
    url="https://dedoc-readme.hf.space",
)

docs = loader.load()

docs[0].page_content[:400]
```




    '\nMadam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\n\n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\n\n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\n\n\nWith a duty to one another to the American people to '




```python

```
