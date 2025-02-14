# LLM Sherpa

This notebook covers how to use `LLM Sherpa` to load files of many types. `LLM Sherpa` supports different file formats including DOCX, PPTX, HTML, TXT, and XML.

`LLMSherpaFileLoader` use LayoutPDFReader, which is part of the LLMSherpa library. This tool is designed to parse PDFs while preserving their layout information, which is often lost when using most PDF to text parsers.

Here are some key features of LayoutPDFReader:

* It can identify and extract sections and subsections along with their levels.
* It combines lines to form paragraphs.
* It can identify links between sections and paragraphs.
* It can extract tables along with the section the tables are found in.
* It can identify and extract lists and nested lists.
* It can join content spread across pages.
* It can remove repeating headers and footers.
* It can remove watermarks.

check [llmsherpa](https://llmsherpa.readthedocs.io/en/latest/) documentation.

`INFO: this library fail with some pdf files so use it with caution.`


```python
# Install package
# !pip install --upgrade --quiet llmsherpa
```

## LLMSherpaFileLoader

Under the hood LLMSherpaFileLoader defined some strategist to load file content: ["sections", "chunks", "html", "text"], setup [nlm-ingestor](https://github.com/nlmatics/nlm-ingestor) to get `llmsherpa_api_url` or use the default.

### sections strategy: return the file parsed into sections


```python
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="sections",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()
```


```python
docs[1]
```




    Document(page_content='Abstract\nWe study how to apply large language models to write grounded and organized long-form articles from scratch, with comparable breadth and depth to Wikipedia pages.\nThis underexplored problem poses new challenges at the pre-writing stage, including how to research the topic and prepare an outline prior to writing.\nWe propose STORM, a writing system for the Synthesis of Topic Outlines through\nReferences\nFull-length Article\nTopic\nOutline\n2022 Winter Olympics\nOpening Ceremony\nResearch via Question Asking\nRetrieval and Multi-perspective Question Asking.\nSTORM models the pre-writing stage by\nLLM\n(1) discovering diverse perspectives in researching the given topic, (2) simulating conversations where writers carrying different perspectives pose questions to a topic expert grounded on trusted Internet sources, (3) curating the collected information to create an outline.\nFor evaluation, we curate FreshWiki, a dataset of recent high-quality Wikipedia articles, and formulate outline assessments to evaluate the pre-writing stage.\nWe further gather feedback from experienced Wikipedia editors.\nCompared to articles generated by an outlinedriven retrieval-augmented baseline, more of STORM’s articles are deemed to be organized (by a 25% absolute increase) and broad in coverage (by 10%).\nThe expert feedback also helps identify new challenges for generating grounded long articles, such as source bias transfer and over-association of unrelated facts.\n1. Can you provide any information about the transportation arrangements for the opening ceremony?\nLLM\n2. Can you provide any information about the budget for the 2022 Winter Olympics opening ceremony?…\nLLM- Role1\nLLM- Role2\nLLM- Role1', metadata={'source': 'https://arxiv.org/pdf/2402.14207.pdf', 'section_number': 1, 'section_title': 'Abstract'})




```python
len(docs)
```




    79



### chunks strategy: return the file parsed into chunks


```python
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="chunks",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()
```


```python
docs[1]
```




    Document(page_content='Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models\nStanford University {shaoyj, yuchengj, tkanell, peterxu, okhattab}@stanford.edu lam@cs.stanford.edu', metadata={'source': 'https://arxiv.org/pdf/2402.14207.pdf', 'chunk_number': 1, 'chunk_type': 'para'})




```python
len(docs)
```




    306



### html strategy: return the file as one html document


```python
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="html",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()
```


```python
docs[0].page_content[:400]
```




    '<html><h1>Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</h1><table><th><td colSpan=1>Yijia Shao</td><td colSpan=1>Yucheng Jiang</td><td colSpan=1>Theodore A. Kanell</td><td colSpan=1>Peter Xu</td></th><tr><td colSpan=1></td><td colSpan=1>Omar Khattab</td><td colSpan=1>Monica S. Lam</td><td colSpan=1></td></tr></table><p>Stanford University {shaoyj, yuchengj, '




```python
len(docs)
```




    1



### text strategy: return the file as one text document


```python
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

loader = LLMSherpaFileLoader(
    file_path="https://arxiv.org/pdf/2402.14207.pdf",
    new_indent_parser=True,
    apply_ocr=True,
    strategy="text",
    llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
)
docs = loader.load()
```


```python
docs[0].page_content[:400]
```




    'Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models\n | Yijia Shao | Yucheng Jiang | Theodore A. Kanell | Peter Xu\n | --- | --- | --- | ---\n |  | Omar Khattab | Monica S. Lam | \n\nStanford University {shaoyj, yuchengj, tkanell, peterxu, okhattab}@stanford.edu lam@cs.stanford.edu\nAbstract\nWe study how to apply large language models to write grounded and organized long'




```python
len(docs)
```




    1


