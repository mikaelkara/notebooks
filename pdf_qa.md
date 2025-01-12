---
keywords: [pdf, document loader]
---
# Build a PDF ingestion and Question/Answering system

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Document loaders](/docs/concepts/document_loaders)
- [Chat models](/docs/concepts/chat_models)
- [Embeddings](/docs/concepts/embedding_models)
- [Vector stores](/docs/concepts/vectorstores)
- [Retrieval-augmented generation](/docs/tutorials/rag/)

:::

PDF files often hold crucial unstructured data unavailable from other sources. They can be quite lengthy, and unlike plain text files, cannot generally be fed directly into the prompt of a language model.

In this tutorial, you'll create a system that can answer questions about PDF files. More specifically, you'll use a [Document Loader](/docs/concepts/document_loaders) to load text in a format usable by an LLM, then build a retrieval-augmented generation (RAG) pipeline to answer questions, including citations from the source material.

This tutorial will gloss over some concepts more deeply covered in our [RAG](/docs/tutorials/rag/) tutorial, so you may want to go through those first if you haven't already.

Let's dive in!

## Loading documents

First, you'll need to choose a PDF to load. We'll use a document from [Nike's annual public SEC report](https://s1.q4cdn.com/806093406/files/doc_downloads/2023/414759-1-_5_Nike-NPS-Combo_Form-10-K_WR.pdf). It's over 100 pages long, and contains some crucial data mixed with longer explanatory text. However, you can feel free to use a PDF of your choosing.

Once you've chosen your PDF, the next step is to load it into a format that an LLM can more easily handle, since LLMs generally require text inputs. LangChain has a few different [built-in document loaders](/docs/how_to/document_loader_pdf/) for this purpose which you can experiment with. Below, we'll use one powered by the [`pypdf`](https://pypi.org/project/pypdf/) package that reads from a filepath:


```python
%pip install -qU pypdf langchain_community
```


```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
```

    107
    


```python
print(docs[0].page_content[0:100])
print(docs[0].metadata)
```

    Table of Contents
    UNITED STATES
    SECURITIES AND EXCHANGE COMMISSION
    Washington, D.C. 20549
    FORM 10-K
    
    {'source': '../example_data/nke-10k-2023.pdf', 'page': 0}
    

So what just happened?

- The loader reads the PDF at the specified path into memory.
- It then extracts text data using the `pypdf` package.
- Finally, it creates a LangChain [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) for each page of the PDF with the page's content and some metadata about where in the document the text came from.

LangChain has [many other document loaders](/docs/integrations/document_loaders/) for other data sources, or you can create a [custom document loader](/docs/how_to/document_loader_custom/).

## Question answering with RAG

Next, you'll prepare the loaded documents for later retrieval. Using a [text splitter](/docs/concepts/text_splitters), you'll split your loaded documents into smaller documents that can more easily fit into an LLM's context window, then load them into a [vector store](/docs/concepts/vectorstores). You can then create a [retriever](/docs/concepts/retrievers) from the vector store for use in our RAG chain:

import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm" openaiParams={`model="gpt-4o"`} />



```python
# | output: false
# | echo: false

import getpass
import os

from langchain_anthropic import ChatAnthropic

if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Anthropic API Key:")

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
```


```python
%pip install langchain_openai
```


```python
# | output: false
# | echo: false

import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
```


```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()
```

Finally, you'll use some built-in helpers to construct the final `rag_chain`:


```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

results
```




    {'input': "What was Nike's revenue in 2023?",
     'context': [Document(page_content='Table of Contents\nFISCAL 2023 NIKE BRAND REVENUE HIGHLIGHTS\nThe following tables present NIKE Brand revenues disaggregated by reportable operating segment, distribution channel and major product line:\nFISCAL 2023 COMPARED TO FISCAL 2022\n•NIKE, Inc. Revenues were $51.2 billion in fiscal 2023, which increased 10% and 16% compared to fiscal 2022 on a reported and currency-neutral basis, respectively.\nThe increase was due to higher revenues in North America, Europe, Middle East & Africa ("EMEA"), APLA and Greater China, which contributed approximately 7, 6,\n2 and 1 percentage points to NIKE, Inc. Revenues, respectively.\n•NIKE Brand revenues, which represented over 90% of NIKE, Inc. Revenues, increased 10% and 16% on a reported and currency-neutral basis, respectively. This\nincrease was primarily due to higher revenues in Men\'s, the Jordan Brand, Women\'s and Kids\' which grew 17%, 35%,11% and 10%, respectively, on a wholesale\nequivalent basis.', metadata={'page': 35, 'source': '../example_data/nke-10k-2023.pdf'}),
      Document(page_content='Enterprise Resource Planning Platform, data and analytics, demand sensing, insight gathering, and other areas to create an end-to-end technology foundation, which we\nbelieve will further accelerate our digital transformation. We believe this unified approach will accelerate growth and unlock more efficiency for our business, while driving\nspeed and responsiveness as we serve consumers globally.\nFINANCIAL HIGHLIGHTS\n•In fiscal 2023, NIKE, Inc. achieved record Revenues of $51.2 billion, which increased 10% and 16% on a reported and currency-neutral basis, respectively\n•NIKE Direct revenues grew 14% from $18.7 billion in fiscal 2022 to $21.3 billion in fiscal 2023, and represented approximately 44% of total NIKE Brand revenues for\nfiscal 2023\n•Gross margin for the fiscal year decreased 250 basis points to 43.5% primarily driven by higher product costs, higher markdowns and unfavorable changes in foreign\ncurrency exchange rates, partially offset by strategic pricing actions', metadata={'page': 30, 'source': '../example_data/nke-10k-2023.pdf'}),
      Document(page_content="Table of Contents\nNORTH AMERICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 14,897 $ 12,228 22 % 22 %$ 11,644 5 % 5 %\nApparel 5,947 5,492 8 % 9 % 5,028 9 % 9 %\nEquipment 764 633 21 % 21 % 507 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nRevenues by:    \nSales to Wholesale Customers $ 11,273 $ 9,621 17 % 18 %$ 10,186 -6 % -6 %\nSales through NIKE Direct 10,335 8,732 18 % 18 % 6,993 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nEARNINGS BEFORE INTEREST AND TAXES $ 5,454 $ 5,114 7 % $ 5,089 0 %\nFISCAL 2023 COMPARED TO FISCAL 2022\n•North America revenues increased 18% on a currency-neutral basis, primarily due to higher revenues in Men's and the Jordan Brand. NIKE Direct revenues\nincreased 18%, driven by strong digital sales growth of 23%, comparable store sales growth of 9% and the addition of new stores.", metadata={'page': 39, 'source': '../example_data/nke-10k-2023.pdf'}),
      Document(page_content="Table of Contents\nEUROPE, MIDDLE EAST & AFRICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 8,260 $ 7,388 12 % 25 %$ 6,970 6 % 9 %\nApparel 4,566 4,527 1 % 14 % 3,996 13 % 16 %\nEquipment 592 564 5 % 18 % 490 15 % 17 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nRevenues by:    \nSales to Wholesale Customers $ 8,522 $ 8,377 2 % 15 %$ 7,812 7 % 10 %\nSales through NIKE Direct 4,896 4,102 19 % 33 % 3,644 13 % 15 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nEARNINGS BEFORE INTEREST AND TAXES $ 3,531 $ 3,293 7 % $ 2,435 35 % \nFISCAL 2023 COMPARED TO FISCAL 2022\n•EMEA revenues increased 21% on a currency-neutral basis, due to higher revenues in Men's, the Jordan Brand, Women's and Kids'. NIKE Direct revenues\nincreased 33%, driven primarily by strong digital sales growth of 43% and comparable store sales growth of 22%.", metadata={'page': 40, 'source': '../example_data/nke-10k-2023.pdf'})],
     'answer': 'According to the financial highlights, Nike, Inc. achieved record revenues of $51.2 billion in fiscal 2023, which increased 10% on a reported basis and 16% on a currency-neutral basis compared to fiscal 2022.'}



You can see that you get both a final answer in the `answer` key of the results dict, and the `context` the LLM used to generate an answer.

Examining the values under the `context` further, you can see that they are documents that each contain a chunk of the ingested page content. Usefully, these documents also preserve the original metadata from way back when you first loaded them:


```python
print(results["context"][0].page_content)
```

    Table of Contents
    FISCAL 2023 NIKE BRAND REVENUE HIGHLIGHTS
    The following tables present NIKE Brand revenues disaggregated by reportable operating segment, distribution channel and major product line:
    FISCAL 2023 COMPARED TO FISCAL 2022
    •NIKE, Inc. Revenues were $51.2 billion in fiscal 2023, which increased 10% and 16% compared to fiscal 2022 on a reported and currency-neutral basis, respectively.
    The increase was due to higher revenues in North America, Europe, Middle East & Africa ("EMEA"), APLA and Greater China, which contributed approximately 7, 6,
    2 and 1 percentage points to NIKE, Inc. Revenues, respectively.
    •NIKE Brand revenues, which represented over 90% of NIKE, Inc. Revenues, increased 10% and 16% on a reported and currency-neutral basis, respectively. This
    increase was primarily due to higher revenues in Men's, the Jordan Brand, Women's and Kids' which grew 17%, 35%,11% and 10%, respectively, on a wholesale
    equivalent basis.
    


```python
print(results["context"][0].metadata)
```

    {'page': 35, 'source': '../example_data/nke-10k-2023.pdf'}
    

This particular chunk came from page 35 in the original PDF. You can use this data to show which page in the PDF the answer came from, allowing users to quickly verify that answers are based on the source material.

:::info
For a deeper dive into RAG, see [this more focused tutorial](/docs/tutorials/rag/) or [our how-to guides](/docs/how_to/#qa-with-rag).
:::

## Next steps

You've now seen how to load documents from a PDF file with a Document Loader and some techniques you can use to prepare that loaded data for RAG.

For more on document loaders, you can check out:

- [The entry in the conceptual guide](/docs/concepts/document_loaders)
- [Related how-to guides](/docs/how_to/#document-loaders)
- [Available integrations](/docs/integrations/document_loaders/)
- [How to create a custom document loader](/docs/how_to/document_loader_custom/)

For more on RAG, see:

- [Build a Retrieval Augmented Generation (RAG) App](/docs/tutorials/rag/)
- [Related how-to guides](/docs/how_to/#qa-with-rag)
