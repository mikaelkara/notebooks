# Jina Reranker

This notebook shows how to use Jina Reranker for document compression and retrieval.


```python
%pip install -qU langchain langchain-openai langchain-community langchain-text-splitters langchainhub

%pip install --upgrade --quiet  faiss

# OR  (depending on Python version)

%pip install --upgrade --quiet  faiss_cpu
```


```python
# Helper function for printing docs


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
```

## Set up the base vector store retriever

Let's start by initializing a simple vector store retriever and storing the 2023 State of the Union speech (in chunks). We can set up the retriever to retrieve a high number (20) of docs.

##### Set the Jina and OpenAI API keys


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["JINA_API_KEY"] = getpass.getpass()
```


```python
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = TextLoader(
    "../../how_to/state_of_the_union.txt",
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding = JinaEmbeddings(model_name="jina-embeddings-v2-base-en")
retriever = FAISS.from_documents(texts, embedding).as_retriever(search_kwargs={"k": 20})

query = "What did the president say about Ketanji Brown Jackson"
docs = retriever.get_relevant_documents(query)
pretty_print_docs(docs)
```

## Doing reranking with JinaRerank

Now let's wrap our base retriever with a ContextualCompressionRetriever, using Jina Reranker as a compressor.


```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import JinaRerank

compressor = JinaRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.get_relevant_documents(
    "What did the president say about Ketanji Jackson Brown"
)
```


```python
pretty_print_docs(compressed_docs)
```

## QA reranking with Jina Reranker


```python
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retrieval_qa_chat_prompt.pretty_print()
```

    ================================[1m System Message [0m================================
    
    Answer any use questions based solely on the context below:
    
    <context>
    [33;1m[1;3m{context}[0m
    </context>
    
    =============================[1m Messages Placeholder [0m=============================
    
    [33;1m[1;3m{chat_history}[0m
    
    ================================[1m Human Message [0m=================================
    
    [33;1m[1;3m{input}[0m
    


```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
chain = create_retrieval_chain(compression_retriever, combine_docs_chain)
```


```python
chain.invoke({"input": query})
```
