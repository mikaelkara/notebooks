# Context Enrichment Window for Document Retrieval

## Overview

This code implements a context enrichment window technique for document retrieval in a vector database. It enhances the standard retrieval process by adding surrounding context to each retrieved chunk, improving the coherence and completeness of the returned information.

## Motivation

Traditional vector search often returns isolated chunks of text, which may lack necessary context for full understanding. This approach aims to provide a more comprehensive view of the retrieved information by including neighboring text chunks.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and OpenAI embeddings
3. Custom retrieval function with context window
4. Comparison between standard and context-enriched retrieval

## Method Details

### Document Preprocessing

1. The PDF is read and converted to a string.
2. The text is split into chunks with surrounding sentences

### Vector Store Creation

1. OpenAI embeddings are used to create vector representations of the chunks.
2. A FAISS vector store is created from these embeddings.

### Context-Enriched Retrieval

LlamaIndex has a special parser for such task. [SentenceWindowNodeParser](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#sentencewindownodeparser) this parser splits documents into sentences. But the resulting nodes inculde the surronding senteces with a relation structure. Then, on the query [MetadataReplacementPostProcessor](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor) helps connecting back these related sentences.

### Retrieval Comparison

The notebook includes a section to compare standard retrieval with the context-enriched approach.

## Benefits of this Approach

1. Provides more coherent and contextually rich results
2. Maintains the advantages of vector search while mitigating its tendency to return isolated text fragments
3. Allows for flexible adjustment of the context window size

## Conclusion

This context enrichment window technique offers a promising way to improve the quality of retrieved information in vector-based document search systems. By providing surrounding context, it helps maintain the coherence and completeness of the retrieved information, potentially leading to better understanding and more accurate responses in downstream tasks such as question answering.

<div style="text-align: center;">

<img src="../images/vector-search-comparison_context_enrichment.svg" alt="context enrichment window" style="width:70%; height:auto;">
</div>

### Import libraries and environment variables


```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
import faiss
import os
import sys
from dotenv import load_dotenv
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Llamaindex global settings for llm and embeddings
EMBED_DIMENSION=512
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=EMBED_DIMENSION)
```

### Read docs


```python
path = "../data/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()
print(documents[0])
```

### Create vector store and retriever


```python
# Create FaisVectorStore to store embeddings
fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)
```

## Ingestion Pipelines

### Ingestion Pipeline with Sentence Splitter


```python
base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],
    vector_store=vector_store
)

base_nodes = base_pipeline.run(documents=documents)
```

### Ingestion Pipeline with Sentence Window


```python
node_parser = SentenceWindowNodeParser(
    # How many sentences on both sides to capture. 
    # Setting this to 3 results in 7 sentences.
    window_size=3,
    # the metadata key for to be used in MetadataReplacementPostProcessor
    window_metadata_key="window",
    # the metadata key that holds the original sentence
    original_text_metadata_key="original_sentence"
)

# Create a pipeline with defined document transformations and vectorstore
pipeline = IngestionPipeline(
    transformations=[node_parser],
    vector_store=vector_store,
)

windowed_nodes = pipeline.run(documents=documents)
```

## Querying


```python
query = "Explain the role of deforestation and fossil fuels in climate change"
```

### Querying *without* Metadata Replacement 


```python
# Create vector index from base nodes
base_index = VectorStoreIndex(base_nodes)

# Instantiate query engine from vector index
base_query_engine = base_index.as_query_engine(
    similarity_top_k=1,
)

# Send query to the engine to get related node(s)
base_response = base_query_engine.query(query)

print(base_response)
```

#### Print Metadata of the Retrieved Node


```python
pprint(base_response.source_nodes[0].node.metadata)
```

### Querying with Metadata Replacement
"Metadata replacement" intutively might sound a little off topic since we're working on the base sentences. But LlamaIndex stores these "before/after sentences" in the metadata data of the nodes. Therefore to build back up these windows of sentences we need Metadata replacement post processor.


```python
# Create window index from nodes created from SentenceWindowNodeParser
windowed_index = VectorStoreIndex(windowed_nodes)

# Instantiate query enine with MetadataReplacementPostProcessor
windowed_query_engine = windowed_index.as_query_engine(
    similarity_top_k=1,
    node_postprocessors=[
        MetadataReplacementPostProcessor(
            target_metadata_key="window" # `window_metadata_key` key defined in SentenceWindowNodeParser
            )
        ],
)

# Send query to the engine to get related node(s)
windowed_response = windowed_query_engine.query(query)

print(windowed_response)
```

#### Print Metadata of the Retrieved Node


```python
# Window and original sentence are added to the metadata
pprint(windowed_response.source_nodes[0].node.metadata)
```
