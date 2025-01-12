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

# Intro to Building a Scalable and Modular RAG System with RAG Engine in Vertex AI 

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Frag-engine%2Fintro_rag_engine.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/rag-engine/intro_rag_engine.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/rag-engine/intro_rag_engine.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Holt Skinner](https://github.com/holtskinner) |

## Overview

Retrieval Augmented Generation (RAG) improves large language models by allowing them to access and process external information sources during generation. This ensures the model's responses are grounded in factual data and avoids hallucinations.

A common problem with LLMs is that they don't understand private knowledge, that
is, your organization's data. With RAG Engine, you can enrich the
LLM context with additional private information, because the model can reduce
hallucination and answer questions more accurately.

By combining additional knowledge sources with the existing knowledge that LLMs
have, a better context is provided. The improved context along with the query
enhances the quality of the LLM's response.

The following concepts are key to understanding Vertex AI RAG Engine. These concepts are listed in the order of the
retrieval-augmented generation (RAG) process.

1. **Data ingestion**: Intake data from different data sources. For example,
  local files, Google Cloud Storage, and Google Drive.

1. **Data transformation**: Conversion of the data in preparation for indexing. For example, data is split into chunks.

1. **Embedding**: Numerical representations of words or pieces of text. These numbers capture the
   semantic meaning and context of the text. Similar or related words or text
   tend to have similar embeddings, which means they are closer together in the
   high-dimensional vector space.

1. **Data indexing**: RAG Engine creates an index called a corpus.
   The index structures the knowledge base so it's optimized for searching. For
   example, the index is like a detailed table of contents for a massive
   reference book.

1. **Retrieval**: When a user asks a question or provides a prompt, the retrieval
  component in RAG Engine searches through its knowledge
  base to find information that is relevant to the query.

1. **Generation**: The retrieved information becomes the context added to the
  original user query as a guide for the generative AI model to generate
  factually grounded and relevant responses.

For more information, refer to the public documentation for [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-overview).

## Get started

### Install Vertex AI SDK and other required packages



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
from IPython.display import Markdown
from vertexai.preview import rag
from vertexai.preview.generative_models import GenerativeModel, Tool
```

### Create a RAG Corpus


```
# Currently supports Google first-party embedding models
EMBEDDING_MODEL = "publishers/google/models/text-embedding-004"  # @param {type:"string", isTemplate: true}
embedding_model_config = rag.EmbeddingModelConfig(publisher_model=EMBEDDING_MODEL)

rag_corpus = rag.create_corpus(
    display_name="my-rag-corpus", embedding_model_config=embedding_model_config
)
```

### Check the corpus just created


```
rag.list_corpora()
```

### Upload a local file to the corpus


```
%%writefile test.txt

Here's a demo using RAG Engine on Vertex AI.
```


```
rag_file = rag.upload_file(
    corpus_name=rag_corpus.name,
    path="test.txt",
    display_name="test.txt",
    description="my test file",
)
```

### Import files from Google Cloud Storage

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Google Cloud Storage bucket.

For this example, we'll use a public GCS bucket containing earning reports from Alphabet.


```
INPUT_GCS_BUCKET = (
    "gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs/"
)

response = rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[INPUT_GCS_BUCKET],
    chunk_size=1024,  # Optional
    chunk_overlap=100,  # Optional
    max_embedding_requests_per_min=900,  # Optional
)
```

### Import files from Google Drive

Eligible paths can be formatted as:

- `https://drive.google.com/drive/folders/{folder_id}`
- `https://drive.google.com/file/d/{file_id}`.

Remember to grant "Viewer" access to the "Vertex RAG Data Service Agent" (with the format of `service-{project_number}@gcp-sa-vertex-rag.iam.gserviceaccount.com`) for your Drive folder/files.



```
response = rag.import_files(
    corpus_name=rag_corpus.name,
    paths=["https://drive.google.com/drive/folders/{folder_id}"],
    chunk_size=512,
    chunk_overlap=50,
)
```

### Optional: Perform direct context retrieval


```
# Direct context retrieval
response = rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,
            # Optional: supply IDs from `rag.list_files()`.
            # rag_file_ids=["rag-file-1", "rag-file-2", ...],
        )
    ],
    text="What is RAG and why it is helpful?",
    similarity_top_k=10,  # Optional
    vector_distance_threshold=0.5,  # Optional
)
print(response)

# Optional: The retrieved context can be passed to any SDK or model generation API to generate final results.
# context = " ".join([context.text for context in response.contexts.contexts]).replace("\n", "")
```

### Create RAG Retrieval Tool


```
# Create a tool for the RAG Corpus
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_corpora=[rag_corpus.name],
            similarity_top_k=10,
            vector_distance_threshold=0.5,
        ),
    )
)
```

### Generate Content with Gemini using Rag Retrieval Tool


```
# Load tool into Gemini model
rag_gemini_model = GenerativeModel(
    "gemini-1.5-flash-001",  # your self-deployed endpoint
    tools=[rag_retrieval_tool],
)
```


```
response = rag_gemini_model.generate_content("What is RAG?")

display(Markdown(response.text))
```

### Generate Content with Llama3 using Rag Retrieval Tool


```
# Load tool into Llama model
llama_gemini_model = GenerativeModel(
    # your self-deployed endpoint for Llama3
    "projects/{project}/locations/{location}/endpoints/{endpoint_resource_id}",
    tools=[rag_retrieval_tool],
)
```


```
response = rag_gemini_model.generate_content("What is RAG?")

display(Markdown(response.text))
```
