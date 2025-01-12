# How to add scores to retriever results

Retrievers will return sequences of [Document](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html) objects, which by default include no information about the process that retrieved them (e.g., a similarity score against a query). Here we demonstrate how to add retrieval scores to the `.metadata` of documents:
1. From [vectorstore retrievers](/docs/how_to/vectorstore_retriever);
2. From higher-order LangChain retrievers, such as [SelfQueryRetriever](/docs/how_to/self_query) or [MultiVectorRetriever](/docs/how_to/multi_vector).

For (1), we will implement a short wrapper function around the corresponding vector store. For (2), we will update a method of the corresponding class.

## Create vector store

First we populate a vector store with some data. We will use a [PineconeVectorStore](https://python.langchain.com/api_reference/pinecone/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html), but this guide is compatible with any LangChain vector store that implements a `.similarity_search_with_score` method.


```python
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]

vectorstore = PineconeVectorStore.from_documents(
    docs, index_name="sample", embedding=OpenAIEmbeddings()
)
```

## Retriever

To obtain scores from a vector store retriever, we wrap the underlying vector store's `.similarity_search_with_score` method in a short function that packages scores into the associated document's metadata.

We add a `@chain` decorator to the function to create a [Runnable](/docs/concepts/lcel) that can be used similarly to a typical retriever.


```python
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs
```


```python
result = retriever.invoke("dinosaur")
result
```




    (Document(page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose', metadata={'genre': 'science fiction', 'rating': 7.7, 'year': 1993.0, 'score': 0.84429127}),
     Document(page_content='Toys come alive and have a blast doing so', metadata={'genre': 'animated', 'year': 1995.0, 'score': 0.792038262}),
     Document(page_content='Three men walk into the Zone, three men walk out of the Zone', metadata={'director': 'Andrei Tarkovsky', 'genre': 'thriller', 'rating': 9.9, 'year': 1979.0, 'score': 0.751571238}),
     Document(page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea', metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006.0, 'score': 0.747471571}))



Note that similarity scores from the retrieval step are included in the metadata of the above documents.

## SelfQueryRetriever

`SelfQueryRetriever` will use a LLM to generate a query that is potentially structured-- for example, it can construct filters for the retrieval on top of the usual semantic-similarity driven selection. See [this guide](/docs/how_to/self_query) for more detail.

`SelfQueryRetriever` includes a short (1 - 2 line) method `_get_docs_with_query` that executes the `vectorstore` search. We can subclass `SelfQueryRetriever` and override this method to propagate similarity scores.

First, following the [how-to guide](/docs/how_to/self_query), we will need to establish some metadata on which to filter:


```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0)
```

We then override the `_get_docs_with_query` to use the `similarity_search_with_score` method of the underlying vector store: 


```python
from typing import Any, Dict


class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        """Get docs, adding score information."""
        docs, scores = zip(
            *self.vectorstore.similarity_search_with_score(query, **search_kwargs)
        )
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs
```

Invoking this retriever will now include similarity scores in the document metadata. Note that the underlying structured-query capabilities of `SelfQueryRetriever` are retained.


```python
retriever = CustomSelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)


result = retriever.invoke("dinosaur movie with rating less than 8")
result
```




    (Document(page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose', metadata={'genre': 'science fiction', 'rating': 7.7, 'year': 1993.0, 'score': 0.84429127}),)



## MultiVectorRetriever

`MultiVectorRetriever` allows you to associate multiple vectors with a single document. This can be useful in a number of applications. For example, we can index small chunks of a larger document and run the retrieval on the chunks, but return the larger "parent" document when invoking the retriever. [ParentDocumentRetriever](/docs/how_to/parent_document_retriever/), a subclass of `MultiVectorRetriever`, includes convenience methods for populating a vector store to support this. Further applications are detailed in this [how-to guide](/docs/how_to/multi_vector/).

To propagate similarity scores through this retriever, we can again subclass `MultiVectorRetriever` and override a method. This time we will override `_get_relevant_documents`.

First, we prepare some fake data. We generate fake "whole documents" and store them in a document store; here we will use a simple [InMemoryStore](https://python.langchain.com/api_reference/core/stores/langchain_core.stores.InMemoryBaseStore.html).


```python
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The storage layer for the parent documents
docstore = InMemoryStore()
fake_whole_documents = [
    ("fake_id_1", Document(page_content="fake whole document 1")),
    ("fake_id_2", Document(page_content="fake whole document 2")),
]
docstore.mset(fake_whole_documents)
```

Next we will add some fake "sub-documents" to our vector store. We can link these sub-documents to the parent documents by populating the `"doc_id"` key in its metadata.


```python
docs = [
    Document(
        page_content="A snippet from a larger document discussing cats.",
        metadata={"doc_id": "fake_id_1"},
    ),
    Document(
        page_content="A snippet from a larger document discussing discourse.",
        metadata={"doc_id": "fake_id_1"},
    ),
    Document(
        page_content="A snippet from a larger document discussing chocolate.",
        metadata={"doc_id": "fake_id_2"},
    ),
]

vectorstore.add_documents(docs)
```




    ['62a85353-41ff-4346-bff7-be6c8ec2ed89',
     '5d4a0e83-4cc5-40f1-bc73-ed9cbad0ee15',
     '8c1d9a56-120f-45e4-ba70-a19cd19a38f4']



To propagate the scores, we subclass `MultiVectorRetriever` and override its `_get_relevant_documents` method. Here we will make two changes:

1. We will add similarity scores to the metadata of the corresponding "sub-documents" using the `similarity_search_with_score` method of the underlying vector store as above;
2. We will include a list of these sub-documents in the metadata of the retrieved parent document. This surfaces what snippets of text were identified by the retrieval, together with their corresponding similarity scores.


```python
from collections import defaultdict

from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class CustomMultiVectorRetriever(MultiVectorRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        results = self.vectorstore.similarity_search_with_score(
            query, **self.search_kwargs
        )

        # Map doc_ids to list of sub-documents, adding scores to metadata
        id_to_doc = defaultdict(list)
        for doc, score in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                doc.metadata["score"] = score
                id_to_doc[doc_id].append(doc)

        # Fetch documents corresponding to doc_ids, retaining sub_docs in metadata
        docs = []
        for _id, sub_docs in id_to_doc.items():
            docstore_docs = self.docstore.mget([_id])
            if docstore_docs:
                if doc := docstore_docs[0]:
                    doc.metadata["sub_docs"] = sub_docs
                    docs.append(doc)

        return docs
```

Invoking this retriever, we can see that it identifies the correct parent document, including the relevant snippet from the sub-document with similarity score.


```python
retriever = CustomMultiVectorRetriever(vectorstore=vectorstore, docstore=docstore)

retriever.invoke("cat")
```




    [Document(page_content='fake whole document 1', metadata={'sub_docs': [Document(page_content='A snippet from a larger document discussing cats.', metadata={'doc_id': 'fake_id_1', 'score': 0.831276655})]})]


