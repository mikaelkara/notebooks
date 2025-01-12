# Hypothetical Document Embedding (HyDE) in Document Retrieval

## Overview

This code implements a Hypothetical Document Embedding (HyDE) system for document retrieval. HyDE is an innovative approach that transforms query questions into hypothetical documents containing the answer, aiming to bridge the gap between query and document distributions in vector space.

## Motivation

Traditional retrieval methods often struggle with the semantic gap between short queries and longer, more detailed documents. HyDE addresses this by expanding the query into a full hypothetical document, potentially improving retrieval relevance by making the query representation more similar to the document representations in the vector space.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and OpenAI embeddings
3. Language model for generating hypothetical documents
4. Custom HyDERetriever class implementing the HyDE technique

## Method Details

### Document Preprocessing and Vector Store Creation

1. The PDF is processed and split into chunks.
2. A FAISS vector store is created using OpenAI embeddings for efficient similarity search.

### Hypothetical Document Generation

1. A language model (GPT-4) is used to generate a hypothetical document that answers the given query.
2. The generation is guided by a prompt template that ensures the hypothetical document is detailed and matches the chunk size used in the vector store.

### Retrieval Process

The `HyDERetriever` class implements the following steps:

1. Generate a hypothetical document from the query using the language model.
2. Use the hypothetical document as the search query in the vector store.
3. Retrieve the most similar documents to this hypothetical document.

## Key Features

1. Query Expansion: Transforms short queries into detailed hypothetical documents.
2. Flexible Configuration: Allows adjustment of chunk size, overlap, and number of retrieved documents.
3. Integration with OpenAI Models: Uses GPT-4 for hypothetical document generation and OpenAI embeddings for vector representation.

## Benefits of this Approach

1. Improved Relevance: By expanding queries into full documents, HyDE can potentially capture more nuanced and relevant matches.
2. Handling Complex Queries: Particularly useful for complex or multi-faceted queries that might be difficult to match directly.
3. Adaptability: The hypothetical document generation can adapt to different types of queries and document domains.
4. Potential for Better Context Understanding: The expanded query might better capture the context and intent behind the original question.

## Implementation Details

1. Uses OpenAI's ChatGPT model for hypothetical document generation.
2. Employs FAISS for efficient similarity search in the vector space.
3. Allows for easy visualization of both the hypothetical document and retrieved results.

## Conclusion

Hypothetical Document Embedding (HyDE) represents an innovative approach to document retrieval, addressing the semantic gap between queries and documents. By leveraging advanced language models to expand queries into hypothetical documents, HyDE has the potential to significantly improve retrieval relevance, especially for complex or nuanced queries. This technique could be particularly valuable in domains where understanding query intent and context is crucial, such as legal research, academic literature review, or advanced information retrieval systems.

<div style="text-align: center;">

<img src="../images/HyDe.svg" alt="HyDe" style="width:40%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/hyde-advantages.svg" alt="HyDe" style="width:100%; height:auto;">
</div>

### Import libraries


```python
import os
import sys
from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### Define document(s) path


```python
path = "../data/Understanding_Climate_Change.pdf"
```

### Define the HyDe retriever class - creating vector store, generating hypothetical document, and retrieving


```python
class HyDERetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        self.embeddings = OpenAIEmbeddings()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    
        
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            the document size has be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc

```

### Create a HyDe retriever instance


```python
retriever = HyDERetriever(path)
```

### Demonstrate on a use case


```python
test_query = "What is the main cause of climate change?"
results, hypothetical_doc = retriever.retrieve(test_query)
```

### Plot the hypothetical document and the retrieved documnets 


```python
docs_content = [doc.page_content for doc in results]

print("hypothetical_doc:\n")
print(text_wrap(hypothetical_doc)+"\n")
show_context(docs_content)
```
