# Faiss (Async)

>[Facebook AI Similarity Search (Faiss)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also includes supporting code for evaluation and parameter tuning.
>
>See [The FAISS Library](https://arxiv.org/pdf/2401.08281) paper.

[Faiss documentation](https://faiss.ai/).

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use functionality related to the `FAISS` vector database using `asyncio`.
LangChain implemented the synchronous and asynchronous vector store functions.

See `synchronous` version [here](/docs/integrations/vectorstores/faiss).


```python
%pip install --upgrade --quiet  faiss-gpu # For CUDA 7.5+ Supported GPU's.
# OR
%pip install --upgrade --quiet  faiss-cpu # For CPU Installation
```

We want to use OpenAIEmbeddings so we have to get the OpenAI API Key. 


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")

# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../../extras/modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db = await FAISS.afrom_documents(docs, embeddings)

query = "What did the president say about Ketanji Brown Jackson"
docs = await db.asimilarity_search(query)

print(docs[0].page_content)
```

## Similarity Search with score
There are some FAISS specific methods. One of them is `similarity_search_with_score`, which allows you to return not only the documents but also the distance score of the query to them. The returned distance score is L2 distance. Therefore, a lower score is better.


```python
docs_and_scores = await db.asimilarity_search_with_score(query)

docs_and_scores[0]
```

It is also possible to do a search for documents similar to a given embedding vector using `similarity_search_by_vector` which accepts an embedding vector as a parameter instead of a string.


```python
embedding_vector = await embeddings.aembed_query(query)
docs_and_scores = await db.asimilarity_search_by_vector(embedding_vector)
```

## Saving and loading
You can also save and load a FAISS index. This is useful so you don't have to recreate it everytime you use it.


```python
db.save_local("faiss_index")

new_db = FAISS.load_local("faiss_index", embeddings, asynchronous=True)

docs = await new_db.asimilarity_search(query)

docs[0]
```

# Serializing and De-Serializing to bytes

you can pickle the FAISS Index by these functions. If you use embeddings model which is of 90 mb (sentence-transformers/all-MiniLM-L6-v2 or any other model), the resultant pickle size would be more than 90 mb. the size of the model is also included in the overall size. To overcome this, use the below functions. These functions only serializes FAISS index and size would be much lesser. this can be helpful if you wish to store the index in database like sql.


```python
from langchain_huggingface import HuggingFaceEmbeddings

pkl = db.serialize_to_bytes()  # serializes the faiss index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.deserialize_from_bytes(
    embeddings=embeddings, serialized=pkl, asynchronous=True
)  # Load the index
```

## Merging
You can also merge two FAISS vectorstores


```python
db1 = await FAISS.afrom_texts(["foo"], embeddings)
db2 = await FAISS.afrom_texts(["bar"], embeddings)
```


```python
db1.docstore._dict
```




    {'8164a453-9643-4959-87f7-9ba79f9e8fb0': Document(page_content='foo')}




```python
db2.docstore._dict
```




    {'4fbcf8a2-e80f-4f65-9308-2f4cb27cb6e7': Document(page_content='bar')}




```python
db1.merge_from(db2)
```


```python
db1.docstore._dict
```




    {'8164a453-9643-4959-87f7-9ba79f9e8fb0': Document(page_content='foo'),
     '4fbcf8a2-e80f-4f65-9308-2f4cb27cb6e7': Document(page_content='bar')}



## Similarity Search with filtering
FAISS vectorstore can also support filtering, since the FAISS does not natively support filtering we have to do it manually. This is done by first fetching more results than `k` and then filtering them. You can filter the documents based on metadata. You can also set the `fetch_k` parameter when calling any search method to set how many documents you want to fetch before filtering. Here is a small example:


```python
from langchain_core.documents import Document

list_of_documents = [
    Document(page_content="foo", metadata=dict(page=1)),
    Document(page_content="bar", metadata=dict(page=1)),
    Document(page_content="foo", metadata=dict(page=2)),
    Document(page_content="barbar", metadata=dict(page=2)),
    Document(page_content="foo", metadata=dict(page=3)),
    Document(page_content="bar burr", metadata=dict(page=3)),
    Document(page_content="foo", metadata=dict(page=4)),
    Document(page_content="bar bruh", metadata=dict(page=4)),
]
db = FAISS.from_documents(list_of_documents, embeddings)
results_with_scores = db.similarity_search_with_score("foo")
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
```

    Content: foo, Metadata: {'page': 1}, Score: 5.159960813797904e-15
    Content: foo, Metadata: {'page': 2}, Score: 5.159960813797904e-15
    Content: foo, Metadata: {'page': 3}, Score: 5.159960813797904e-15
    Content: foo, Metadata: {'page': 4}, Score: 5.159960813797904e-15
    

Now we make the same query call but we filter for only `page = 1` 


```python
results_with_scores = await db.asimilarity_search_with_score("foo", filter=dict(page=1))
for doc, score in results_with_scores:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
```

    Content: foo, Metadata: {'page': 1}, Score: 5.159960813797904e-15
    Content: bar, Metadata: {'page': 1}, Score: 0.3131446838378906
    

Same thing can be done with the `max_marginal_relevance_search` as well.


```python
results = await db.amax_marginal_relevance_search("foo", filter=dict(page=1))
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
```

    Content: foo, Metadata: {'page': 1}
    Content: bar, Metadata: {'page': 1}
    

Here is an example of how to set `fetch_k` parameter when calling `similarity_search`. Usually you would want the `fetch_k` parameter >> `k` parameter. This is because the `fetch_k` parameter is the number of documents that will be fetched before filtering. If you set `fetch_k` to a low number, you might not get enough documents to filter from.


```python
results = await db.asimilarity_search("foo", filter=dict(page=1), k=1, fetch_k=4)
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
```

    Content: foo, Metadata: {'page': 1}
    

## Delete

You can also delete ids. Note that the ids to delete should be the ids in the docstore.


```python
db.delete([db.index_to_docstore_id[0]])
```




    True




```python
# Is now missing
0 in db.index_to_docstore_id
```




    False




```python

```
