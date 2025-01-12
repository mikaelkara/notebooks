# Explainable Retrieval in Document Search

## Overview

This code implements an Explainable Retriever, a system that not only retrieves relevant documents based on a query but also provides explanations for why each retrieved document is relevant. It combines vector-based similarity search with natural language explanations, enhancing the transparency and interpretability of the retrieval process.

## Motivation

Traditional document retrieval systems often work as black boxes, providing results without explaining why they were chosen. This lack of transparency can be problematic in scenarios where understanding the reasoning behind the results is crucial. The Explainable Retriever addresses this by offering insights into the relevance of each retrieved document.

## Key Components

1. Vector store creation from input texts
2. Base retriever using FAISS for efficient similarity search
3. Language model (LLM) for generating explanations
4. Custom ExplainableRetriever class that combines retrieval and explanation generation

## Method Details

### Document Preprocessing and Vector Store Creation

1. Input texts are converted into embeddings using OpenAI's embedding model.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### Retriever Setup

1. A base retriever is created from the vector store, configured to return the top 5 most similar documents.

### Explanation Generation

1. An LLM (GPT-4) is used to generate explanations.
2. A custom prompt template is defined to guide the LLM in explaining the relevance of retrieved documents.

### ExplainableRetriever Class

1. Combines the base retriever and explanation generation into a single interface.
2. The `retrieve_and_explain` method:
   - Retrieves relevant documents using the base retriever.
   - For each retrieved document, generates an explanation of its relevance to the query.
   - Returns a list of dictionaries containing both the document content and its explanation.

## Benefits of this Approach

1. Transparency: Users can understand why specific documents were retrieved.
2. Trust: Explanations build user confidence in the system's results.
3. Learning: Users can gain insights into the relationships between queries and documents.
4. Debugging: Easier to identify and correct issues in the retrieval process.
5. Customization: The explanation prompt can be tailored for different use cases or domains.

## Conclusion

The Explainable Retriever represents a significant step towards more interpretable and trustworthy information retrieval systems. By providing natural language explanations alongside retrieved documents, it bridges the gap between powerful vector-based search techniques and human understanding. This approach has potential applications in various fields where the reasoning behind information retrieval is as important as the retrieved information itself, such as legal research, medical information systems, and educational tools.

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

### Define the explainable retriever class 


```python
class ExplainableRetriever:
    def __init__(self, texts):
        self.embeddings = OpenAIEmbeddings()

        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)

        
        # Create a base retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create an explanation chain
        explain_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Analyze the relationship between the following query and the retrieved context.
            Explain why this context is relevant to the query and how it might help answer the query.
            
            Query: {query}
            
            Context: {context}
            
            Explanation:
            """
        )
        self.explain_chain = explain_prompt | self.llm

    def retrieve_and_explain(self, query):
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        explained_results = []
        
        for doc in docs:
            # Generate explanation
            input_data = {"query": query, "context": doc.page_content}
            explanation = self.explain_chain.invoke(input_data).content
            
            explained_results.append({
                "content": doc.page_content,
                "explanation": explanation
            })
        
        return explained_results


```

### Create a mock example and explainable retriever instance


```python

# Usage
texts = [
    "The sky is blue because of the way sunlight interacts with the atmosphere.",
    "Photosynthesis is the process by which plants use sunlight to produce energy.",
    "Global warming is caused by the increase of greenhouse gases in Earth's atmosphere."
]

explainable_retriever = ExplainableRetriever(texts)

```

### Show the results


```python
query = "Why is the sky blue?"
results = explainable_retriever.retrieve_and_explain(query)

for i, result in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Content: {result['content']}")
    print(f"Explanation: {result['explanation']}")
    print()
```
