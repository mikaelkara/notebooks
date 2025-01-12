```
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Multimodal Retrieval Augmented Generation (RAG) with Gemini, Vertex AI Vector Search, and LangChain

<table align="left">
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fretrieval-augmented-generation%2Fmultimodal_rag_langchain.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
    <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/multimodal_rag_langchain.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | | 
|-|-|
|Author(s) | [Holt Skinner](https://github.com/holtskinner) |

## Overview

Retrieval augmented generation (RAG) has become a popular paradigm for enabling LLMs to access external data and also as a mechanism for grounding to mitigate against hallucinations.

In this notebook, you will learn how to perform multimodal RAG where you will perform Q&A over a financial document filled with both text and images.

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases. The Gemini API gives you access to the Gemini 1.0 Pro Vision and Gemini 1.0 Pro models.

### Comparing text-based and multimodal RAG

Multimodal RAG offers several advantages over text-based RAG:

1. **Enhanced knowledge access:** Multimodal RAG can access and process both textual and visual information, providing a richer and more comprehensive knowledge base for the LLM.
2. **Improved reasoning capabilities:** By incorporating visual cues, multimodal RAG can make better informed inferences across different types of data modalities.

This notebook shows you how to use RAG with Gemini API in Vertex AI, and [multimodal embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/multimodal-embeddings), to build a document search engine.

Through hands-on examples, you will discover how to construct a multimedia-rich metadata repository of your document sources, enabling search, comparison, and reasoning across diverse information streams.

### Objectives

This notebook provides a guide to building a document search engine using multimodal retrieval augmented generation (RAG), step by step:

1. Extract and store metadata of documents containing both text and images, and generate embeddings the documents
2. Search the metadata with text queries to find similar text or images
3. Search the metadata with image queries to find similar images
4. Using a text query as input, search for contextual answers using both text and images

### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK for Python and other dependencies


```
%pip install -U -q google-cloud-aiplatform langchain-core langchain-google-vertexai langchain-text-splitters langchain-community "unstructured[all-docs]" pypdf pydantic lxml pillow matplotlib opencv-python tiktoken
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Define Google Cloud project information


```
PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# For Vector Search Staging
GCS_BUCKET = "YOUR_BUCKET_NAME"  # @param {type:"string"}
GCS_BUCKET_URI = f"gs://{GCS_BUCKET}"
```

### Initialize the Vertex AI SDK


```
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=GCS_BUCKET_URI)
```

### Import libraries


```
import base64
import os
import re
import uuid

from IPython.display import Image, Markdown, display
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_vertexai import (
    ChatVertexAI,
    VectorSearchVectorStore,
    VertexAI,
    VertexAIEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# from langchain_community.vectorstores import Chroma  # Optional
```

### Define model information

- [Vertex AI - Model Information](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)


```
MODEL_NAME = "gemini-1.5-flash"
GEMINI_OUTPUT_TOKEN_LIMIT = 8192

EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_TOKEN_LIMIT = 2048

TOKEN_LIMIT = min(GEMINI_OUTPUT_TOKEN_LIMIT, EMBEDDING_TOKEN_LIMIT)
```

## Data Loading

#### Get documents and images from GCS


```
# Download documents and images used in this notebook
!gsutil -m rsync -r gs://github-repo/rag/intro_multimodal_rag/ .
print("Download completed")
```

## Partition PDF tables, text, and images

### The data

The source data that you will use in this notebook is a modified version of [Google-10K](https://abc.xyz/assets/investor/static/pdf/20220202_alphabet_10K.pdf) which provides a comprehensive overview of the company's financial performance, business operations, management, and risk factors. As the original document is rather large, you will be using [a modified version with only 14 pages](https://storage.googleapis.com/github-repo/rag/multimodal_rag_langchain/google-10k-sample-14pages.pdf) instead. Although it's truncated, the sample document still contains text along with images such as tables, charts, and graphs.


```
pdf_folder_path = "/content/data/" if "google.colab" in sys.modules else "data/"
pdf_file_name = "google-10k-sample-14pages.pdf"

# Extract images, tables, and chunk text from a PDF file.
raw_pdf_elements = partition_pdf(
    filename=pdf_file_name,
    extract_images_in_pdf=False,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=pdf_folder_path,
)

# Categorize extracted elements from a PDF into tables and texts.
tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

# Optional: Enforce a specific token size for texts
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=10000, chunk_overlap=0
)
joined_texts = " ".join(texts)
texts_4k_token = text_splitter.split_text(joined_texts)
```


```
# Generate summaries of text elements


def generate_text_summaries(
    texts: list[str], tables: list[str], summarize_texts: bool = False
) -> tuple[list, list]:
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = PromptTemplate.from_template(prompt_text)
    empty_response = RunnableLambda(
        lambda x: AIMessage(content="Error processing document")
    )
    # Text summary chain
    model = VertexAI(
        temperature=0, model_name=MODEL_NAME, max_output_tokens=TOKEN_LIMIT
    ).with_fallbacks([empty_response])
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts:
        if summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
        else:
            text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries


# Get text, table summaries
text_summaries, table_summaries = generate_text_summaries(
    texts_4k_token, tables, summarize_texts=True
)
```


```
def encode_image(image_path: str) -> str:
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(model: ChatVertexAI, base64_image: str, prompt: str) -> str:
    """Make image summary"""
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path: str) -> tuple[list[str], list[str]]:
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval.
    If it's a table, extract all elements of the table.
    If it's a graph, explain the findings in the graph.
    Do not include any numbers that are not mentioned in the image.
    """

    model = ChatVertexAI(model_name=MODEL_NAME, max_output_tokens=TOKEN_LIMIT)

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".png"):
            base64_image = encode_image(os.path.join(path, img_file))
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(model, base64_image, prompt))

    return img_base64_list, image_summaries


# Image summaries
img_base64_list, image_summaries = generate_img_summaries(".")
```

## Create & Deploy Vertex AI Vector Search Index & Endpoint

Skip this step if you already have Vector Search set up.

- https://console.cloud.google.com/vertex-ai/matching-engine/indexes

- Create [`MatchingEngineIndex`](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.MatchingEngineIndex)
  - https://cloud.google.com/vertex-ai/docs/vector-search/create-manage-index


```
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings
DIMENSIONS = 768  # Dimensions output from textembedding-gecko

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="mm_rag_langchain_index",
    dimensions=DIMENSIONS,
    approximate_neighbors_count=150,
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=7,
    description="Multimodal RAG LangChain Index",
    index_update_method="STREAM_UPDATE",
)
```

- Create [`MatchingEngineIndexEndpoint`](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.MatchingEngineIndexEndpoint)
  - https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-public


```
DEPLOYED_INDEX_ID = "mm_rag_langchain_index_endpoint"

index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=DEPLOYED_INDEX_ID,
    description="Multimodal RAG LangChain Index Endpoint",
    public_endpoint_enabled=True,
)
```

- Deploy Index to Index Endpoint
  - NOTE: This will take a while to run.
  - You can stop this cell after starting it instead of waiting for deployment.
  - You can check the status at https://console.cloud.google.com/vertex-ai/matching-engine/indexes


```
index_endpoint = index_endpoint.deploy_index(
    index=index, deployed_index_id="mm_rag_langchain_deployed_index"
)
index_endpoint.deployed_indexes
```

## Create retriever & load documents

- Create [`VectorSearchVectorStore`](https://api.python.langchain.com/en/latest/vectorstores/langchain_google_vertexai.vectorstores.vectorstores.VectorSearchVectorStore.html) with Vector Search Index ID and Endpoint ID.
- Use [`textembedding-gecko`](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings) as embedding model.


```
# The vectorstore to use to index the summaries
vectorstore = VectorSearchVectorStore.from_components(
    project_id=PROJECT_ID,
    region=LOCATION,
    gcs_bucket_name=GCS_BUCKET,
    index_id=index.name,
    endpoint_id=index_endpoint.name,
    embedding=VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME),
    stream_update=True,
)
```

- Alternatively, use Chroma for a local vector store.


```
# vectorstore = Chroma(
#     collection_name="mm_rag_test",
#     embedding_function=VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME),
# )
```

- Create Multi-Vector Retriever using the vector store you created.
- Since vector stores only contain the embedding and an ID, you'll also need to create a document store indexed by ID to get the original source documents after searching for embeddings.


```
docstore = InMemoryStore()

id_key = "doc_id"
# Create the multi-vector retriever
retriever_multi_vector_img = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)
```

- Load data into Document Store and Vector Store


```
# Raw Document Contents
doc_contents = texts + tables + img_base64_list

doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries + table_summaries + image_summaries)
]

retriever_multi_vector_img.docstore.mset(list(zip(doc_ids, doc_contents)))

# If using Vertex AI Vector Search, this will take a while to complete.
# You can cancel this cell and continue later.
retriever_multi_vector_img.vectorstore.add_documents(summary_docs)
```

## Create Chain with Retriever and Gemini LLM


```
def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xFF\xD8\xFF": "jpg",
        b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = [
        {
            "type": "text",
            "text": (
                "You are financial analyst tasking with providing investment advice.\n"
                "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
                "Use this information to provide investment advice related to the user's question. \n"
                f"User-provided question: {data_dict['question']}\n\n"
                "Text and / or tables:\n"
                f"{formatted_texts}"
            ),
        }
    ]

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
    return [HumanMessage(content=messages)]


# Create RAG chain
chain_multimodal_rag = (
    {
        "context": retriever_multi_vector_img | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(img_prompt_func)
    | ChatVertexAI(
        temperature=0,
        model_name=MODEL_NAME,
        max_output_tokens=TOKEN_LIMIT,
    )  # Multi-modal LLM
    | StrOutputParser()
)
```

## Process user query


```
query = """
 - What are the critical difference between various graphs for Class A Share?
 - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
 - Identify key chart patterns for Google Class A shares.
 - What is cost of revenues, operating expenses and net income for 2020. Do mention the percentage change
 - What was the effect of Covid in the 2020 financial year?
 - What are the total revenues for APAC and USA for 2021?
 - What is deferred income taxes?
 - How do you compute net income per share?
 - What drove percentage change in the consolidated revenue and cost of revenue for the year 2021 and was there any effect of Covid?
 - What is the cause of 41% increase in revenue from 2020 to 2021 and how much is dollar change?
"""
```

### Get Retrieved documents


```
# List of source documents
docs = retriever_multi_vector_img.get_relevant_documents(query, limit=10)

source_docs = split_image_text_types(docs)

print(source_docs["texts"])

for i in source_docs["images"]:
    display(Image(base64.b64decode(i)))
```

### Get generative response


```
result = chain_multimodal_rag.invoke(query)

Markdown(result)
```

## Conclusions

Congratulations on making it through this multimodal RAG notebook!

While multimodal RAG can be quite powerful, note that it can face some limitations:

* **Data dependency:** Needs high-accuracy data from the text and visuals.
* **Computationally demanding:** Generating embeddings from multimodal data is resource-intensive.
* **Domain specific:** Models trained on general data may not shine in specialized fields like medicine.
* **Black box:** Understanding how these models work can be tricky, hindering trust and adoption.


Despite these challenges, multimodal RAG represents a significant step towards search and retrieval systems that can handle diverse, multimodal data.
