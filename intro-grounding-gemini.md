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

# Getting Started with Grounding with Gemini in Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/grounding/intro-grounding-gemini.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgrounding%2Fintro-grounding-gemini.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/grounding/intro-grounding-gemini.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/grounding/intro-grounding-gemini.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Holt Skinner](https://github.com/holtskinner), [Kristopher Overholt](https://github.com/koverholt) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.11

## Overview

[Grounding in Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini) lets you use generative text models to generate content grounded in your own documents and data. This capability lets the model access information at runtime that goes beyond its training data. By grounding model responses in Google Search results or data stores within [Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction), LLMs that are grounded in data can produce more accurate, up-to-date, and relevant responses.

Grounding provides the following benefits:

- Reduces model hallucinations (instances where the model generates content that isn't factual)
- Anchors model responses to specific information, documents, and data sources
- Enhances the trustworthiness, accuracy, and applicability of the generated content

In the context of grounding in Vertex AI, you can configure two different sources of grounding:

1. Google Search results for data that is publicly available and indexed
2. [Data stores in Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest), which can include your own data in the form of website data, unstructured data, or structured data

### Allowlisting

Some of the features in this sample notebook require access to certain features via an allowlist. [Grounding with Vertex AI Search](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini) is available in Public Preview, whereas Grounding with Google Search results is generally available.

If you use this service in a production application, you will also need to [use a Google Search entry point](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/grounding-search-entry-points).

### Objective

In this tutorial, you learn how to:

- Generate LLM text and chat model responses grounded in Google Search results
- Compare the results of ungrounded LLM responses with grounded LLM responses
- Create and use a data store in Vertex AI Search to ground responses in custom documents and data
- Generate LLM text and chat model responses grounded in Vertex AI Search results

This tutorial uses the following Google Cloud AI services and resources:

- Vertex AI
- Vertex AI Search and Conversation

The steps performed include:

- Configuring the LLM and prompt for various examples
- Sending example prompts to generative text and chat models in Vertex AI
- Setting up a data store in Vertex AI Search with your own data
- Sending example prompts with various levels of grounding (no grounding, web grounding, data store grounding)

## Before you begin

### Set up your Google Cloud project

**The following steps are required, regardless of your notebook environment.**

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.
1. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).
1. Enable the [Vertex AI and Vertex AI Agent Builder APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,discoveryengine.googleapis.com).
1. If you are running this notebook locally, you need to install the [Cloud SDK](https://cloud.google.com/sdk).

### Installation

Install the following packages required to execute this notebook.


```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

Restart the kernel after installing packages:


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your Google Cloud account

If you are running this notebook on Google Colab, you will need to authenticate your environment. To do this, run the new cell below. This step is not required if you are using Vertex AI Workbench.


```
import sys

if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

**If you don't know your project ID**, try the following:
* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113)

You can also change the `REGION` variable used by Vertex AI. Learn more about [Vertex AI regions](https://cloud.google.com/vertex-ai/docs/general/locations).


```
PROJECT_ID = "your-project-id"  # @param {type:"string"}
REGION = "us-central1"  # @param {type: "string"}
```


```
import vertexai

vertexai.init(project=PROJECT_ID, location=REGION)
```

### Import libraries


```
from IPython.display import Markdown, display
from vertexai.generative_models import (
    GenerationResponse,
    GenerativeModel,
    Tool,
    grounding,
)
from vertexai.preview.generative_models import grounding as preview_grounding
```

Initialize the Gemini model from Vertex AI:


```
model = GenerativeModel("gemini-1.5-flash")
```

## Helper functions


```
def print_grounding_response(response: GenerationResponse):
    """Prints Gemini response with grounding citations."""
    grounding_metadata = response.candidates[0].grounding_metadata

    # Citation indices are in byte units
    ENCODING = "utf-8"
    text_bytes = response.text.encode(ENCODING)

    prev_index = 0
    markdown_text = ""

    for grounding_support in grounding_metadata.grounding_supports:
        text_segment = text_bytes[
            prev_index : grounding_support.segment.end_index
        ].decode(ENCODING)

        footnotes_text = ""
        for grounding_chunk_index in grounding_support.grounding_chunk_indices:
            footnotes_text += f"[{grounding_chunk_index + 1}]"

        markdown_text += f"{text_segment} {footnotes_text}\n"
        prev_index = grounding_support.segment.end_index

    if prev_index < len(text_bytes):
        markdown_text += str(text_bytes[prev_index:], encoding=ENCODING)

    markdown_text += "\n----\n## Grounding Sources\n"

    if grounding_metadata.web_search_queries:
        markdown_text += (
            f"\n**Web Search Queries:** {grounding_metadata.web_search_queries}\n"
        )
        if grounding_metadata.search_entry_point:
            markdown_text += f"\n**Search Entry Point:**\n {grounding_metadata.search_entry_point.rendered_content}\n"
    elif grounding_metadata.retrieval_queries:
        markdown_text += (
            f"\n**Retrieval Queries:** {grounding_metadata.retrieval_queries}\n"
        )

    markdown_text += "### Grounding Chunks\n"

    for index, grounding_chunk in enumerate(
        grounding_metadata.grounding_chunks, start=1
    ):
        context = grounding_chunk.web or grounding_chunk.retrieved_context
        if not context:
            print(f"Skipping Grounding Chunk {grounding_chunk}")
            continue

        markdown_text += f"{index}. [{context.title}]({context.uri})\n"

    display(Markdown(markdown_text))
```

## Example: Grounding with Google Search results

In this example, you'll compare LLM responses with no grounding with responses that are grounded in the results of a Google Search. You'll ask a question about a recent hardware release from the Google Store.


```
PROMPT = "You are an expert in astronomy. When is the next solar eclipse in the US?"
```

### Text generation without grounding

Make a prediction request to the LLM with no grounding:


```
response = model.generate_content(PROMPT)

display(Markdown(response.text))
```

### Text generation grounded in Google Search results

Now you can add the `tools` keyword arg with a `grounding` tool of `grounding.GoogleSearchRetrieval()` to instruct the LLM to first perform a Google Search with the prompt, then construct an answer based on the web search results.

The search queries and [Search Entry Point](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/grounding-search-entry-points) are available for each `Candidate` in the response. The helper function `print_grounding_response()` prints the response text with citations.


```
tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

response = model.generate_content(PROMPT, tools=[tool])

print_grounding_response(response)
```

Note that the response without grounding only has limited information from the LLM about solar eclipses. Whereas the response that was grounded in web search results contains the most up to date information from web search results that are returned as part of the LLM with grounding request.

## Example: Grounding with custom documents and data

In this example, you'll compare LLM responses with no grounding with responses that are grounded in the [results of a data store in Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest).

The data store will contain internal documents from a fictional bank, Cymbal Bank. These documents aren't available on the public internet, so the Gemini model won't have any information about them by default.

### Creating a data store in Vertex AI Search

In this example, you'll use a Google Cloud Storage bucket with a few sample internal documents for our bank. There's some docs about booking business travel, strategic plan for this Fiscal Year and HR docs describing the different jobs available in the company.

Follow the tutorial steps in the Vertex AI Search documentation to:

1. [Create a data store with unstructured data](https://cloud.google.com/generative-ai-app-builder/docs/try-enterprise-search#unstructured-data) that loads in documents from the GCS folder `gs://cloud-samples-data/gen-app-builder/search/cymbal-bank-employee`.
2. [Create a search app](https://cloud.google.com/generative-ai-app-builder/docs/try-enterprise-search#create_a_search_app) that is attached to that data store. You should also enable the **Enterprise edition features** so that you can search indexed records within the data store.

Note: The data store must be in the same project that you are using for Gemini.

Once you've created a data store, obtain the Data Store ID and input it below.

Note: You will need to wait for data ingestion to finish before using a data store with grounding. For more information, see [create a data store](https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es).


```
DATA_STORE_PROJECT_ID = PROJECT_ID  # @param {type:"string"}
DATA_STORE_REGION = "global"  # @param {type:"string"}
# Replace this with your data store ID from Vertex AI Search
DATA_STORE_ID = "your-data-store-id_1234567890123"  # @param {type:"string"}
```

Now you can ask a question about the company culture:


```
PROMPT = "What is the company culture like?"
```

### Text generation without grounding

Make a prediction request to the LLM with no grounding:


```
response = model.generate_content(PROMPT)

display(Markdown(response.text))
```

### Text generation grounded in Vertex AI Search results

Now we can add the `tools` keyword arg with a grounding tool of `grounding.VertexAISearch()` to instruct the LLM to first perform a search within your custom data store, then construct an answer based on the relevant documents:


```
tool = Tool.from_retrieval(
    preview_grounding.Retrieval(
        preview_grounding.VertexAISearch(
            datastore=DATA_STORE_ID,
            project=DATA_STORE_PROJECT_ID,
            location=DATA_STORE_REGION,
        )
    )
)

response = model.generate_content(PROMPT, tools=[tool])
print_grounding_response(response)
```

Note that the response without grounding doesn't have any context about what company we are asking about. Whereas the response that was grounded in Vertex AI Search results contains information from the documents provided, along with citations of the information.

<div class="alert alert-block alert-warning">
<b>⚠️ Important notes:</b><br>
<br>
<b>If you get an error when running the previous cell:</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;In order for this sample notebook to work with data store in Vertex AI Search,<br>
&nbsp;&nbsp;&nbsp;&nbsp;you'll need to create a <a href="https://cloud.google.com/generative-ai-app-builder/docs/try-enterprise-search#create_a_data_store">data store</a> <b>and</b> a <a href="https://cloud.google.com/generative-ai-app-builder/docs/try-enterprise-search#create_a_search_app">search app</a> associated with it in Vertex AI Search.<br>
&nbsp;&nbsp;&nbsp;&nbsp;If you only create a data store, the previous request will return errors when making queries against the data store.
<br><br>
<b>If you get an empty response when running the previous cell:</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;You will need to wait for data ingestion to finish before using a data store with grounding.<br>
&nbsp;&nbsp;&nbsp;&nbsp;For more information, see <a href="https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es">create a data store</a>.
</div>
</div>

## Example: Grounded chat responses

You can also use grounding when working with chat models in Vertex AI. In this example, you'll compare LLM responses with no grounding with responses that are grounded in the results of a Google Search and a data store in Vertex AI Search.


```
PROMPT = "What are managed datasets in Vertex AI?"
PROMPT_FOLLOWUP = "What types of data can I use?"
```

### Chat session grounded in Google Search results

Now you can add the `tools` keyword arg with a grounding tool of `grounding.GoogleSearchRetrieval()` to instruct the chat model to first perform a Google Search with the prompt, then construct an answer based on the web search results:


```
chat = model.start_chat()
tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

response = chat.send_message(PROMPT, tools=[tool])
print_grounding_response(response)

response = chat.send_message(PROMPT_FOLLOWUP, tools=[tool])
print_grounding_response(response)
```

### Chat session grounded in Vertex AI Search results

Now we can add the `tools` keyword arg with a grounding tool of `grounding.VertexAISearch()` to instruct the chat model to first perform a search within your custom data store, then construct an answer based on the relevant documents:


```
PROMPT = "How do I book business travel?"
PROMPT_FOLLOWUP = "Give me more details."
```


```
chat = model.start_chat()
tool = Tool.from_retrieval(
    preview_grounding.Retrieval(
        preview_grounding.VertexAISearch(
            datastore=DATA_STORE_ID,
            project=DATA_STORE_PROJECT_ID,
            location=DATA_STORE_REGION,
        )
    )
)
response = chat.send_message(PROMPT, tools=[tool])
print_grounding_response(response)

response = chat.send_message(PROMPT_FOLLOWUP, tools=[tool])
print_grounding_response(response)
```

## Cleaning up

To avoid incurring charges to your Google Cloud account for the resources used in this notebook, follow these steps:

1. To avoid unnecessary Google Cloud charges, use the [Google Cloud console](https://console.cloud.google.com/) to delete your project if you do not need it. Learn more in the Google Cloud documentation for [managing and deleting your project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1. If you used an existing Google Cloud project, delete the resources you created to avoid incurring charges to your account. For more information, refer to the documentation to [Delete data from a data store in Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/delete-datastores), then delete your data store.
1. Disable the [Vertex AI Search and Conversation API](https://console.cloud.google.com/apis/api/discoveryengine.googleapis.com) and [Vertex AI API](https://console.cloud.google.com/apis/api/aiplatform.googleapis.com) in the Google Cloud Console.
