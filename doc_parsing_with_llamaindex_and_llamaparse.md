# Comparing LlamaIndex and LlamaParse for Dense Document Questioning Answering on Vertex AI
<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/doc_parsing_with_llamaindex_and_llamaparse.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fdocument-processing%2Fdoc_parsing_with_llamaindex_and_llamaparse.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/document-processing/doc_parsing_with_llamaindex_and_llamaparse.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/document-processing/doc_parsing_with_llamaindex_and_llamaparse.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Noa Ben-Efraim](https://github.com/noabenefraim/) |

## Overview
There are many ways to customize RAG pipelines by choosing how to ingest, parse, chunk, and retrieve your data. This notebook focuses on comparing different document parsing capabilities offered by LlamaIndex.

We will compare document parsing with LlamaIndex and LlamaParse on a 10-Q financial document, which is heavily populated with complex tables.

### Objectives
This notebook compare using LlamaIndex and LlamaParse for ingesting and indexing a complex document. 

You will complete the following tasks:
1. Ingest and parse document using LlamaIndex SimpleDataReader, LlamaIndex LangchainNodeParser, and LlamaParse Parser using Gemini models.
2. Index your parsed document in a VectorStore.
3. Create a a query agent for each parsing technique that can answer questions against the input document.
4. Compare results across LlamaIndex and LlamaParse.

### LlamaIndex
LlamaIndex is a foundational data framework for building LLM applications. A few of its main capabilities are:

+ Data Ingestion: Loads your data from various sources (documents, databases, APIs).   
+ Indexing: Structures your data into efficient formats for LLM retrieval (e.g., vector stores, tree structures).   
+ Querying: Enables you to ask questions or give instructions to the LLM, referencing your indexed data for answers.   
+ Integration: Connects with various LLMs, vector databases, and other tools.   
  

### LlamaParse
LlamaParse is a tool within the LlamaIndex ecosystem, focused on parsing complex documents:

+ PDFs: Handles PDFs with tables, charts, and other embedded elements that can be challenging for standard parsing.  
+ Semi-structured Data: Extracts structured information from documents that aren't fully formatted databases.   
+ Enhanced Retrieval: Works seamlessly with LlamaIndex to improve retrieval accuracy for complex documents.

## Getting Started

### Authenticate your notebook environment

This notebook expects the following resources to exists:
+ Initialized Google Cloud project 
+ Vertex AI API enabled
+ GCS Bucket and Vertex AI Search Index and Endpoint
+ A LlamaParse API Key [request a key here](https://docs.cloud.llamaindex.ai/llamacloud/getting_started/api_key)


```
PROJECT_ID = ""  # @param {type:"string"}
REGION = ""  # @param {type:"string"}
GCS_BUCKET = ""  # @param {type:"string"}
VS_INDEX_NAME = ""  # @param {type:"string"}
VS_INDEX_ENDPOINT_NAME = ""  # @param {type:"string"}
DATA_FOLDER = "./data"  # @param {type:"string"}
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = ""  # @param {type:"string"}
LOCATION = ""  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Setting up the Environment
Install dependencies


```
%pip install google-cloud-aiplatform \
  llama-index \
  langchain-community \
  llama-index-embeddings-vertex \
  llama-index-llms-vertex \
  termcolor \
  llama-index-core -q
```

Set up imports


```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.response.schema import Response
from llama_index.core.extractors import KeywordExtractor, QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_parse import LlamaParse
from termcolor import colored
```

Generate credentials


```
import google.auth
import google.auth.transport.requests

# credentials will now have an api token
credentials = google.auth.default(quota_project_id="genai-noabe")[0]
request = google.auth.transport.requests.Request()
credentials.refresh(request)
```


```
embedding_model = VertexTextEmbedding("text-embedding-004", credentials=credentials)
llm = Vertex(model="gemini-pro", temperature=0.0, max_tokens=5000)

Settings.embed_model = embedding_model
Settings.llm = llm
```

Set up LlamaIndex settings to point to Gemini models.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Download sample data

For the remainder of the notebook we will examine Alphabet Inc. 10Q document. A 10Q is a financial document that is dense with tables with financial figures. This document is a great candidate to to investigate document parsing capabilities.


```
!mkdir {DATA_FOLDER}
!wget "https://abc.xyz/assets/ae/e9/753110054014b6de4d620a2853f6/goog-10-q-q2-2024.pdf" -P {DATA_FOLDER}
```

##  Document Parsing with LlamaIndex

This section will ingest and parse the 10Q using LlamaIndex tools, specifically focusing on SimpleDirectoryReader and LangChainNodeParser.

### Option 1: `SimpleDirectoryReader`
The SimpleDirectoryReader is the core data ingestion tool in LlamaIndex. It's designed to load data from a variety of sources and convert it into a format suitable for further processing and indexing by LlamaIndex.


```
reader = SimpleDirectoryReader("./data")
documents = reader.load_data(show_progress=True)
print(documents[0])
```


```
# Index the parsed document
simpledirectory_index = VectorStoreIndex.from_documents(documents)

# Generate a query engine based on the SimpleDataReader
simple_query_engine = simpledirectory_index.as_query_engine(similarity_top_k=2)
```

### Option 2: `LangChainNodeParser` with LlamaIndex
The LangChainNodeParser is a part of LlamaIndex. It is a specialized parser designed to extract structured information from text documents using the power of LangChain.

Key Features:
+ LangChain Integration: Leverages LangChain's powerful language models and tools to parse text.
+ Node-Based Output: Converts unstructured text into a structured format based on a defined schema, represented as a hierarchy of nodes. This enables more sophisticated querying and analysis of the extracted information.
+ Customization: Supports defining custom parsing schemas to match the structure of your specific documents.
+ Flexibility: Can be used in combination with other LlamaIndex components, such as the SimpleDataReader, to process and index the extracted structured data.


```
parser = LangchainNodeParser(RecursiveCharacterTextSplitter())
langchain_nodes = parser.get_nodes_from_documents(documents)
```


```
# An example node that was generated using the LangChainNodeParser and the associated metadata
langchain_nodes[0]
```


```
# Index the document based on the LangChain nodes generated above
langchainparser_index = VectorStoreIndex(nodes=langchain_nodes)

# Create a query engine based off the LangChainNodeParser
lg_query_engine = langchainparser_index.as_query_engine(similarity_top_k=2)
```

## LlamaParse

LlamaParse Parser is a powerful tool for extracting structured data from unstructured or semi-structured text, offering flexibility, customization, and seamless integration within the LlamaIndex framework.It can take an unstructured or semi-structured text document and, using a defined schema, extract structured information from it. This structured output is represented as a nested hierarchy of nodes, facilitating further processing and analysis.

A few key features include:

+ JSON Schema: Leverages the standardized JSON Schema format for more complex schemas.
+ Prompt Templates: Allows you to craft custom prompts to guide the language model's parsing behavior, offering greater control and adaptability.
+ LLM Selection: You have the flexibility to choose the specific LLM you want to use for parsing, enabling you to tailor the performance to your specific needs and budget.
+ Node-Based Output:
    + Structured Representation: The parsed output is organized into a hierarchy of nodes, each representing a piece of extracted information.
    + Nested Structure: Nodes can contain other nodes, allowing for the representation of complex relationships and nested data structures within the document.
    + Metadata: Nodes can also include additional metadata, such as confidence scores or source information, enriching the extracted data.
+ Integration with LlamaIndex: The structured output from parser() seamlessly integrates with other LlamaIndex components, such as indexing and querying, facilitating efficient retrieval and analysis of the extracted information.

#### Define a Parser

Here we will define a LlamaParse() parser with specific parsing instructions, and ingest the data.


```
parser = LlamaParse(
    parsing_instruction="You are a financial analyst working specifically with 10Q documents. Not all pages have titles. Try to reconstruct the dialogue spoken in a cohesive way.",
    api_key="",
    result_type="text",  # "markdown" and "text" are available
    language="en",
    invalidate_cache=True,
)
```

### Option 1 - LlamaParse with SimpleDirectoryReader

This is the apples to apples comparison with LlamaIndex. We are using the SimpleDirectoryReader with the LlamaParse file extractor, and then loading the data directly from documents to a Vector Store for retrieval.


```
import nest_asyncio

nest_asyncio.apply()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    input_files=["./data/goog-10-q-q2-2024.pdf"], file_extractor=file_extractor  # type: ignore
).load_data()
```


```
lp_simple = VectorStoreIndex.from_documents(documents)
lp_simple_engine = lp_simple.as_query_engine(similarity_top_k=2)
```

### Option 2 - LlamaParse and Vertex AI Vector Search

This approach is a more customized approach by defining the Vector Search mechanism through Vertex AI and extracting metadata that will be embedded and stored in the search index. 

Using metadata in Retrieval Augmented Generation (RAG) improves accuracy and context by focusing searches and providing additional information. This leads to efficient filtering, ranking, and personalized responses tailored to user needs and history. Metadata also facilitates handling complex multi-criteria queries, making RAG systems more versatile and effective.

The following section will:
+ Parse the documents using LlamaParse
+ Extract metadata from documents returned from LlamaParse
+ Create metadata embeddings attached to each document
+ Create index in Vertex AI Vector Store
+ Query against the Vector Store

#### Parse data using LlamaParse


```
documents = parser.load_data("./data/goog-10-q-q2-2024.pdf")
```

#### Create Metadata from Nodes

Using extractors we will generate meta-data for each node. The metadata is generated using Gemini-Pro and focuses on what questions can this text answer and what key words are meaningful in this section. Each metadata piece will be embedded with Gemini text-embedding model. 

Creating metadata can be useful for another lookup criteria during RAG based search.


```
extractors = [
    QuestionsAnsweredExtractor(questions=3, llm=llm),
    KeywordExtractor(keywords=10, llm=llm),
]
```


```
# Run metadata transformation pipeline.
pipeline = IngestionPipeline(
    transformations=extractors,  # type: ignore
)
nodes = await pipeline.arun(documents=documents, in_place=False)
```

Example metadata that was generated:


```
print(nodes[1].metadata)
```


```
# Generate embeddings for each metadata node
for node in nodes:
    node_embedding = embedding_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding
```

#### Load Nodes into Predefined Vector Store

This following section required a preexisting Vertex AI Vector Store. Vector stores contain embedding vectors of ingested document chunks.

For information to create a vector store, refer to this link https://docs.llamaindex.ai/en/stable/examples/vector_stores/VertexAIVectorSearchDemo/


```
vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    index_id="",  # Add in your Vertex AI Vector Search Index ID
    endpoint_id="",  # Add in your Vertex AI Vector Search Deployed Index ID
    gcs_bucket_name=GCS_BUCKET,
)

# Only need to run once
vector_store.add(nodes)
```

#### Create a search index and search and query the Vector Store


```
lp_index = VectorStoreIndex.from_vector_store(vector_store)
lp_query_engine = lp_index.as_query_engine(similarity_top_k=2)
```

## Query Comparison between LlamaIndex and LlamaParse
Below are queries that responses can be found in the 10Q document within complex tables. Let's see how each approach compares.


```
queries = [
    "What are the total cash, cash equivalents, and marketable securities as of Dec 23 2023",
    "Total investments with fair value change reflected in other comprehensive income as of Dec 23 2023",
    "What is the corporate debt securities unrealized loss as of Dec 31 2023 for 12 months or greater?",
    "What is the coupon rate for total outstanding debt",
    "Provide the table of share repurchases",
]
```


```
def print_output(response: Response):
    print("Response:")
    print("-" * 80)
    print(colored(response.response, color="red"))
    print("-" * 80)
    print("Source Documents:")
    print("-" * 80)
    for source in response.source_nodes:
        print(f"Sample Text: {source.text[:100]}")
        print(f"Relevance score: {source.get_score():.3f}")
        print(f"File Name: {source.metadata.get('file_name')}")
        print(f"Page #: {source.metadata.get('page_label')}")
        print(f"File Path: {source.metadata.get('file_path')}")
        print("-" * 80)


def run_query(query_idx: int):
    query = queries[query_idx]
    print("Query: " + query)
    print(colored("LlamaIndex SimpleDirectoryReader response....\n", color="blue"))
    print_output(simple_query_engine.query(query))

    print(
        colored(
            "LlamaIndex LangChainNodeParser on LlamaIndex response....\n", color="blue"
        )
    )
    print_output(lg_query_engine.query(query))

    print(colored("LlamaParse Simple response....\n", color="blue"))
    print_output(lp_simple_engine.query(query))

    print(colored("LlamaParse on Vertex AI response....\n", color="blue"))
    print_output(lp_query_engine.query(query))
    print("###################################################\n\n")
```


```
run_query(query_idx=0)
```


```
run_query(query_idx=1)
```


```
run_query(query_idx=2)
```


```
run_query(query_idx=3)
```


```
run_query(query_idx=4)
```

## Observations

### Answer Key
| Query                                                                                                | Answer           | Citation page |
|------------------------------------------------------------------------------------------------------|------------------|---------------|
| "What are the total cash, cash equivalents, and marketable securities as of Dec 23 2023"             | $110,916 million | 5             |
| "Total investments with fair value change reflected in other comprehensive income as of Dec 23 2023" | $78,917 million  | 13            |
| "What is the corporate debt securities unrealized loss as of Dec 31 2023 for 12 months or greater?   | 592 million      | 15            |
| "What is the coupon rate for total outstanding debt"                                                 | 0.45-2.25%       | 22            |
| "Provide the table of share repurchases"                                                             | Table            | 27 or 49      |

### Generated Answers
| Document Parsing Technique               | Query 1 | Query 2 | Query 3 | Query 4 | Query 5 |
|------------------------------------------|---------|---------|---------|---------|---------|
| LlamaIndex - SimpleDirectoryReader       | (✓)     | (✓)     | (✓)     | (✓)     | (✓)     |
| LlamaIndex - LangChainNodeParser         | (✓)     | (✓)     | (✓)     | (✓)     | (✓)     |
| LlamaParse - SimpleDirectoryReader       |  (✓)    |  (✓)    |      (✓)|    (✓)  |     (✓) |
| LlamaParse - Vertex AI Vector Search       |  (✓)    |  (✓)    |      (✓)|    (✓)  |     (✓) |

## Conclusion

There are many ways to customize your data ingestion and retrieval pipelines for custom RAG applications. This notebook was an overview to a handful of options that work in combination with Google Gemini models. 
