# PDFPlumber

Like PyMuPDF, the output Documents contain detailed metadata about the PDF and its pages, and returns one document per page.

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support|
| :--- | :--- | :---: | :---: |  :---: |
| [PDFPlumberLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFPlumberLoader.html) | [langchain_community](https://python.langchain.com/api_reference/community/index.html) | ✅ | ❌ | ❌ | 
### Loader features
| Source | Document Lazy Loading | Native Async Support
| :---: | :---: | :---: | 
| PDFPlumberLoader | ✅ | ❌ | 

## Setup

### Credentials

No credentials are needed to use this loader.

If you want to get automated best in-class tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:


```python
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
# os.environ["LANGSMITH_TRACING"] = "true"
```

### Installation

Install **langchain_community**.


```python
%pip install -qU langchain_community
```

## Initialization

Now we can instantiate our model object and load documents:


```python
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("./example_data/layout-parser-paper.pdf")
```

## Load


```python
docs = loader.load()
docs[0]
```




    Document(metadata={'source': './example_data/layout-parser-paper.pdf', 'file_path': './example_data/layout-parser-paper.pdf', 'page': 0, 'total_pages': 16, 'Author': '', 'CreationDate': 'D:20210622012710Z', 'Creator': 'LaTeX with hyperref', 'Keywords': '', 'ModDate': 'D:20210622012710Z', 'PTEX.Fullbanner': 'This is pdfTeX, Version 3.14159265-2.6-1.40.21 (TeX Live 2020) kpathsea version 6.3.2', 'Producer': 'pdfTeX-1.40.21', 'Subject': '', 'Title': '', 'Trapped': 'False'}, page_content='LayoutParser: A Unified Toolkit for Deep\nLearning Based Document Image Analysis\nZejiang Shen1 ((cid:0)), Ruochen Zhang2, Melissa Dell3, Benjamin Charles Germain\nLee4, Jacob Carlson3, and Weining Li5\n1 Allen Institute for AI\nshannons@allenai.org\n2 Brown University\nruochen zhang@brown.edu\n3 Harvard University\n{melissadell,jacob carlson}@fas.harvard.edu\n4 University of Washington\nbcgl@cs.washington.edu\n5 University of Waterloo\nw422li@uwaterloo.ca\nAbstract. Recentadvancesindocumentimageanalysis(DIA)havebeen\nprimarily driven by the application of neural networks. Ideally, research\noutcomescouldbeeasilydeployedinproductionandextendedforfurther\ninvestigation. However, various factors like loosely organized codebases\nand sophisticated model configurations complicate the easy reuse of im-\nportantinnovationsbyawideaudience.Thoughtherehavebeenon-going\nefforts to improve reusability and simplify deep learning (DL) model\ndevelopmentindisciplineslikenaturallanguageprocessingandcomputer\nvision, none of them are optimized for challenges in the domain of DIA.\nThis represents a major gap in the existing toolkit, as DIA is central to\nacademicresearchacross awiderangeof disciplinesinthesocialsciences\nand humanities. This paper introduces LayoutParser, an open-source\nlibrary for streamlining the usage of DL in DIA research and applica-\ntions. The core LayoutParser library comes with a set of simple and\nintuitiveinterfacesforapplyingandcustomizingDLmodelsforlayoutde-\ntection,characterrecognition,andmanyotherdocumentprocessingtasks.\nTo promote extensibility, LayoutParser also incorporates a community\nplatform for sharing both pre-trained models and full document digiti-\nzation pipelines. We demonstrate that LayoutParser is helpful for both\nlightweight and large-scale digitization pipelines in real-word use cases.\nThe library is publicly available at https://layout-parser.github.io.\nKeywords: DocumentImageAnalysis·DeepLearning·LayoutAnalysis\n· Character Recognition · Open Source library · Toolkit.\n1 Introduction\nDeep Learning(DL)-based approaches are the state-of-the-art for a wide range of\ndocumentimageanalysis(DIA)tasksincludingdocumentimageclassification[11,\n1202\nnuJ\n12\n]VC.sc[\n2v84351.3012:viXra\n')




```python
print(docs[0].metadata)
```

    {'source': './example_data/layout-parser-paper.pdf', 'file_path': './example_data/layout-parser-paper.pdf', 'page': 0, 'total_pages': 16, 'Author': '', 'CreationDate': 'D:20210622012710Z', 'Creator': 'LaTeX with hyperref', 'Keywords': '', 'ModDate': 'D:20210622012710Z', 'PTEX.Fullbanner': 'This is pdfTeX, Version 3.14159265-2.6-1.40.21 (TeX Live 2020) kpathsea version 6.3.2', 'Producer': 'pdfTeX-1.40.21', 'Subject': '', 'Title': '', 'Trapped': 'False'}
    

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

For detailed documentation of all PDFPlumberLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PDFPlumberLoader.html
