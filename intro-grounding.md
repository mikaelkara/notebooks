```
# Copyright 2023 Google LLC
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

# Getting Started with Grounding in Vertex AI

> **NOTE:** This notebook uses the PaLM generative model, which will reach its [discontinuation date in October 2024](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text#model_versions). Please refer to [this updated notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/grounding/intro-grounding-gemini.ipynb) for a version which uses the latest Gemini model.

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/grounding/intro-grounding.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/grounding/intro-grounding.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/grounding/intro-grounding.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Kristopher Overholt](https://github.com/koverholt) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.11

## Overview

[Grounding in Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/grounding/ground-language-models) lets you use language models (e.g., [`text-bison` and `chat-bison`](https://cloud.google.com/vertex-ai/docs/generative-ai/language-model-overview)) to generate content grounded in your own documents and data. This capability lets the model access information at runtime that goes beyond its training data. By grounding model responses in Google Search results or data stores within [Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/enterprise-search-introduction), LLMs that are grounded in data can produce more accurate, up-to-date, and relevant responses.

Grounding provides the following benefits:

- Reduces model hallucinations (instances where the model generates content that isn't factual)
- Anchors model responses to specific information, documents, and data sources
- Enhances the trustworthiness, accuracy, and applicability of the generated content

In the context of grounding in Vertex AI, you can configure two different sources of grounding:

1. Google Search results for data that is publicly available and indexed
1. [Data stores in Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest), which can include your own data in the form of website data, unstructured data, or structured data

**NOTE:** Some of the features in this sample notebook require early access to certain features via an allowlist. [Grounding with Vertex AI Search](https://cloud.google.com/vertex-ai/docs/generative-ai/grounding/ground-language-models) is available in Public Preview, whereas Grounding with Google Web Search results is available in Private Preview. To request early access to features in Private Preview, contact your account representative or [Google Cloud Support](https://cloud.google.com/contact).

### Objective

In this tutorial, you learn how to:

- Generate LLM text and chat model responses grounded in Google Search results
- Compare the results of ungrounded LLM responses with grounded LLM responses
- Create and use a data store in Vertex AI Search to ground responses in custom documents and data
- Generate LLM text and chat model responses grounded in Vertex AI Search results
- Use the asynchronous text and chat models APIs with grounding

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
1. Enable the [Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com) and [Vertex AI Search and Conversation API](https://console.cloud.google.com/flows/enableapi?apiid=discoveryengine.googleapis.com).
1. If you want to use Grounding with Google Web Search results, your project must also be allowlisted for this feature while it is in the Private Preview stage.
1. If you are running this notebook locally, you need to install the [Cloud SDK](https://cloud.google.com/sdk).

### Installation

Install the following packages required to execute this notebook.


```
%pip install --upgrade --quiet google-cloud-aiplatform==1.38.1
```

Restart the kernel after installing packages:


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Configure your project ID

**If you don't know your project ID**, try the following:
* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113)


```
PROJECT_ID = "your-project-id"  # @param {type:"string"}

# Set the project ID
!gcloud config set project {PROJECT_ID}
```

    Updated property [core/project].
    

### Configure your region

You can also change the `REGION` variable used by Vertex AI. Learn more about [Vertex AI regions](https://cloud.google.com/vertex-ai/docs/general/locations).


```
REGION = "us-central1"  # @param {type: "string"}
```

### Authenticate your Google Cloud account

If you are running this notebook on Google Colab, you will need to authenticate your environment. To do this, run the new cell below. This step is not required if you are using Vertex AI Workbench.


```
import sys

if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Import libraries


```
import vertexai
from vertexai.language_models import ChatModel, GroundingSource, TextGenerationModel
```

### Initialize Vertex AI SDK for Python

Initialize the Vertex AI SDK for Python for your project:


```
vertexai.init(project=PROJECT_ID, location=REGION)
```

Initialize the generative text and chat models from Vertex AI:


```
text_model = TextGenerationModel.from_pretrained("text-bison")
chat_model = ChatModel.from_pretrained("chat-bison")
```

## Example: Grounding with Google Search results

In this example, you'll compare LLM responses with no grounding with responses that are grounded in the results of a Google Search. You'll ask a question about a recent hardware release from the Google Store.


```
PROMPT = (
    "What are the price, available colors, and storage size options of a Pixel Tablet?"
)
```

### Text generation without grounding

Make a prediction request to the LLM with no grounding:


```
response = text_model.predict(PROMPT)
response
```




     **Price:**
    
    * Starting at $399 for the Wi-Fi-only model with 128GB of storage
    * $499 for the Wi-Fi + 5G model with 128GB of storage
    * $599 for the Wi-Fi + 5G model with 256GB of storage
    
    **Available Colors:**
    
    * Chalk (white)
    * Charcoal (black)
    * Sage (green)
    
    **Storage Size Options:**
    
    * 128GB
    * 256GB



### Text generation grounded in Google Search results

Now you can add the `grounding_source` keyword arg with a grounding source of `GroundingSource.WebSearch()` to instruct the LLM to first perform a Google Search with the prompt, then construct an answer based on the web search results:


```
grounding_source = GroundingSource.WebSearch()

response = text_model.predict(
    PROMPT,
    grounding_source=grounding_source,
)

response, response.grounding_metadata
```




    ( The Pixel Tablet starts at $499 in the US, £599 in the UK, €679 throughout selected European regions, and CAD $699 in Canada. It comes in three colors: Porcelain (white), Rose (pink), and Hazel (green). The storage size options are 128GB and 256GB.,
     GroundingMetadata(citations=[GroundingCitation(start_index=1, end_index=129, url='https://www.androidauthority.com/google-pixel-tablet-3163922/', title=None, license=None, publication_date=None), GroundingCitation(start_index=130, end_index=206, url='https://www.androidpolice.com/google-pixel-tablet/', title=None, license=None, publication_date=None)], search_queries=['Pixel Tablet price, colors, and storage size options?']))



Note that the response without grounding only has limited information from the LLM about the Pixel tablet. Whereas the response that was grounded in web search results contains the most up to date information from web search results that are returned as part of the LLM with grounding request.

## Example: Grounding with custom documents and data

In this example, you'll compare LLM responses with no grounding with responses that are grounded in the [results of a data store in Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest). You'll ask a question about a GoogleSQL query to create an [object table in BigQuery](https://cloud.google.com/bigquery/docs/object-table-introduction).

### Creating a data store in Vertex AI Search

Follow the steps in the [Vertex AI Search getting started documentation](https://cloud.google.com/generative-ai-app-builder/docs/try-enterprise-search#create_a_search_app_for_website_data) to create a data store in Vertex AI Search with sample data. In this example, you'll use a website-based data store that contains content from the Google Cloud website, including documentation.

Once you've created a data store, obtain the Data Store ID and input it below.


```
DATA_STORE_ID = "your-data-store-id_1234567890123"  # Replace this with your data store ID from Vertex AI Search
DATA_STORE_REGION = "global"
```

Now you can ask a question about object tables in BigQuery and when to use them:


```
PROMPT = "When should I use an object table in BigQuery? And how does it store data?"
```

### Text generation without grounding

Make a prediction request to the LLM with no grounding:


```
response = text_model.predict(PROMPT)

response, response.grounding_metadata
```




    ( **When to use an object table in BigQuery**
    
    Object tables are a specialized type of table in BigQuery that is designed for storing and querying semi-structured data. Semi-structured data is data that does not conform to a fixed schema, such as JSON, XML, or Avro.
    
    Object tables are useful for storing data that is:
    
    * **Complex and hierarchical:** Object tables can store data that is nested or has a complex structure. For example, you could store a JSON object that represents a customer record, which includes the customer's name, address, and order history.
    * **Changing frequently:** Object,
     GroundingMetadata(citations=[], search_queries=[]))



### Text generation grounded in Vertex AI Search results

Now we can add the `grounding_source` keyword arg with a grounding source of `GroundingSource.VertexAISearch()` to instruct the LLM to first perform a search within your custom data store, then construct an answer based on the relevant documents:


```
grounding_source = GroundingSource.VertexAISearch(
    data_store_id=DATA_STORE_ID, location=DATA_STORE_REGION
)

response = text_model.predict(
    PROMPT,
    grounding_source=grounding_source,
)

response, response.grounding_metadata
```




    ( **When to use an object table in BigQuery**
    
    Object tables are useful for storing and analyzing unstructured data, such as images, videos, and audio files. They can also be used to store semi-structured data, such as JSON or XML files.
    
    Object tables are particularly useful when you need to:
    
    * Store large amounts of unstructured data
    * Perform complex analysis on unstructured data
    * Share unstructured data with others
    * Access unstructured data from multiple locations
    
    **How object tables store data**
    
    Object tables store data in a columnar format, which makes it efficient to query and analyze large amounts of data. Each column,
     GroundingMetadata(citations=[], search_queries=['When should I use an object table in BigQuery?']))



Note that the response without grounding only has limited information from the LLM about object tables in BigQuery that might not be accurate. Whereas the response that was grounded in Vertex AI Search results contains the most up to date information from the Google Cloud documentation about BigQuery.

## Example: Grounded chat responses

You can also use grounding when working with chat models in Vertex AI. In this example, you'll compare LLM responses with no grounding with responses that are grounded in the results of a Google Search and a data store in Vertex AI Search.

You'll ask a question about Vertex AI and a follow up question about managed datasets in Vertex AI:


```
PROMPT = "What are managed datasets in Vertex AI?"
PROMPT_FOLLOWUP = "What types of data can I use"
```

### Chat session without grounding

Start a chat session and send messages to the LLM with no grounding:


```
chat = chat_model.start_chat()

response = chat.send_message(PROMPT)
print(response.text)

response = chat.send_message(PROMPT_FOLLOWUP)
print(response.text)
```

     Managed datasets are a feature of Vertex AI that allows you to easily create, manage, and version your datasets. With managed datasets, you can:
    
    * **Easily create datasets:** You can create datasets from a variety of sources, including Cloud Storage, BigQuery, and CSV files.
    * **Manage datasets:** You can view, edit, and delete datasets. You can also add and remove columns from datasets.
    * **Version datasets:** You can create new versions of datasets. This allows you to track changes to your datasets over time.
    * **Share datasets:** You can share datasets with other users in your organization.
    * **Use datasets in Vertex AI models:** You can use managed datasets to train and evaluate Vertex AI models.
    
    Managed datasets are a powerful tool that can help you to improve the performance of your Vertex AI models.
     You can use a variety of data types with managed datasets, including:
    
    * **Structured data:** Structured data is data that is organized in a tabular format. Examples of structured data include CSV files, JSON files, and BigQuery tables.
    * **Unstructured data:** Unstructured data is data that is not organized in a tabular format. Examples of unstructured data include images, videos, and text files.
    * **Semi-structured data:** Semi-structured data is data that has some structure, but not as much as structured data. Examples of semi-structured data include XML files and HTML files.
    
    You can also use managed datasets to combine different types of data. For example, you could create a dataset that includes both structured data and unstructured data.
    

### Chat session grounded in Google Search results

Now you can add the `grounding_source` keyword arg with a grounding source of `GroundingSource.WebSearch()` to instruct the chat model to first perform a Google Search with the prompt, then construct an answer based on the web search results:


```
chat = chat_model.start_chat()
grounding_source = GroundingSource.WebSearch()

response = chat.send_message(
    PROMPT,
    grounding_source=grounding_source,
)
print(response.text)
print(response.grounding_metadata)

response = chat.send_message(
    PROMPT_FOLLOWUP,
    grounding_source=grounding_source,
)
print(response.text)
print(response.grounding_metadata)
```

     Managed datasets in Vertex AI are a way to store and manage your data for use in machine learning models. They provide a number of benefits, including:
    
    - **Centralized storage:** Managed datasets are stored in a central location, making them easy to access and manage.
    - **Data versioning:** Managed datasets support data versioning, so you can easily track changes to your data over time.
    - **Data security:** Managed datasets are encrypted at rest and in transit, so you can be sure that your data is safe.
    - **Data processing:** Managed datasets can be processed using a variety of tools, including Vertex AI's built-in data processing tools.
    GroundingMetadata(citations=[], search_queries=['Vertex AI managed datasets?'])
     You can use a variety of data types in managed datasets, including:
    
    - **Structured data:** Structured data is data that is organized in a tabular format, such as CSV files or SQL tables.
    - **Unstructured data:** Unstructured data is data that is not organized in a tabular format, such as images, videos, and text files.
    - **Semi-structured data:** Semi-structured data is data that is partially structured, such as JSON files or XML files.
    GroundingMetadata(citations=[], search_queries=['What types of data can I use in a managed dataset in Vertex AI?'])
    

### Chat session grounded in Vertex AI Search results

Now you can add the `grounding_source` keyword arg with a grounding source of `GroundingSource.VertexAISearch()` to instruct the chat model to first perform a search within your custom data store, then construct an answer based on the relevant documents:


```
chat = chat_model.start_chat()
grounding_source = GroundingSource.VertexAISearch(
    data_store_id=DATA_STORE_ID, location=DATA_STORE_REGION
)

response = chat.send_message(
    PROMPT,
    grounding_source=grounding_source,
)
print(response.text)
print(response.grounding_metadata)

response = chat.send_message(
    PROMPT_FOLLOWUP,
    grounding_source=grounding_source,
)
print(response.text)
print(response.grounding_metadata)
```

     Managed datasets in Vertex AI are used to provide the source data for training AutoML and custom models.
    GroundingMetadata(citations=[GroundingCitation(start_index=1, end_index=105, url='https://cloud.google.com/vertex-ai/docs/datasets/overview', title=None, license=None, publication_date=None)], search_queries=['Vertex AI managed datasets?'])
     Managed datasets in Vertex AI are used to provide the source data for training AutoML and custom models.
    GroundingMetadata(citations=[GroundingCitation(start_index=1, end_index=105, url='https://cloud.google.com/vertex-ai/docs/datasets/overview', title=None, license=None, publication_date=None)], search_queries=['Vertex AI managed datasets'])
    

## Example: Grounded async text and chat responses

You can also use grounding in Vertex AI when working with the asynchronous APIs for the text and chat models. In this example, you'll compare LLM responses with no grounding with responses that are grounded in the results of a data store in Vertex AI Search.

You'll ask a question about different services available in Google Cloud.


```
PROMPT = "What are the different types of databases available in Google Cloud?"
```

### Async text generation grounded in Google Search results


```
grounding_source = GroundingSource.WebSearch()

response = await text_model.predict_async(
    PROMPT,
    grounding_source=grounding_source,
)

response, response.grounding_metadata
```




    ( The different types of databases available in Google Cloud are:
    
    1. Cloud Spanner: It provides all the relational database capabilities of Cloud SQL along with horizontal scalability which usually comes with NoSQL databases.
    
    2. Cloud Bigtable: Users can store different types of data, including time-series, marketing, financial, IoT, and graph data. Cloud Bigtable also integrates with popular big data.
    
    3. Cloud SQL: Provides managed MySQL, PostgreSQL, and SQL Server databases on Google Cloud.
    
    4. AlloyDB: It is a fully managed PostgreSQL-compatible database service that offers high performance and scalability for PostgreSQL workloads.,
     GroundingMetadata(citations=[GroundingCitation(start_index=69, end_index=226, url='https://medium.com/google-cloud/choose-the-right-database-service-in-gcp-8e3803245e1d', title=None, license=None, publication_date=None), GroundingCitation(start_index=230, end_index=352, url='https://www.techtarget.com/searchcloudcomputing/feature/7-Google-Cloud-database-options-to-free-up-your-IT-team', title=None, license=None, publication_date=None), GroundingCitation(start_index=353, end_index=407, url='https://www.techtarget.com/searchcloudcomputing/feature/7-Google-Cloud-database-options-to-free-up-your-IT-team', title=None, license=None, publication_date=None), GroundingCitation(start_index=411, end_index=500, url='https://cloud.google.com/blog/topics/developers-practitioners/your-google-cloud-database-options-explained', title=None, license=None, publication_date=None)], search_queries=['What are the different types of databases available in Google Cloud?']))



### Async text generation grounded in Vertex AI Search results


```
grounding_source = GroundingSource.VertexAISearch(
    data_store_id=DATA_STORE_ID, location=DATA_STORE_REGION
)

response = await text_model.predict_async(
    PROMPT,
    grounding_source=grounding_source,
)

response, response.grounding_metadata
```




    ( The different types of databases available in Google Cloud are:
    
    1. **Cloud SQL**: A fully-managed database service that supports MySQL, PostgreSQL, and SQL Server.
    2. **BigQuery**: A serverless, highly scalable data warehouse that can handle petabytes of data.
    3. **Spanner**: A globally distributed, highly scalable relational database that supports ACID transactions.
    4. **Firestore**: A NoSQL document database that is ideal for real-time applications.
    5. **Memorystore**: An in-memory data store that is ideal for applications that require fast access to data.,
     GroundingMetadata(citations=[GroundingCitation(start_index=1, end_index=65, url='https://cloud.google.com/learn/what-is-a-cloud-database', title=None, license=None, publication_date=None)], search_queries=['What are the different types of databases available in Google Cloud?']))



### Async chat session grounded in Google Search results


```
chat = chat_model.start_chat()

grounding_source = GroundingSource.WebSearch()
response = await chat.send_message_async(
    PROMPT,
    grounding_source=grounding_source,
)

response, response.grounding_metadata
```




    ( - Cloud Spanner
    - Cloud Bigtable
    - Cloud SQL
    - AlloyDB,
     GroundingMetadata(citations=[], search_queries=['What are the different types of databases available in Google Cloud?']))



### Async chat session grounded in Vertex AI Search results


```
chat = chat_model.start_chat()

grounding_source = GroundingSource.VertexAISearch(
    data_store_id=DATA_STORE_ID, location=DATA_STORE_REGION
)
response = await chat.send_message_async(
    PROMPT,
    grounding_source=grounding_source,
)

response, response.grounding_metadata
```




    ( The different types of databases available in Google Cloud are:
    
    1. [2] **Cloud SQL**: A fully-managed database service that supports MySQL, PostgreSQL, and SQL Server.
    2. [2] **BigQuery**: A serverless, highly scalable data warehouse that can handle large amounts of data.
    3. [2] **Spanner**: A globally-distributed, highly available relational database.
    4. [2] **Firestore**: A NoSQL document database that is ideal for real-time applications.
    5. [2] **Memorystore**: An in-memory data store that is optimized for high-performance applications.,
     GroundingMetadata(citations=[GroundingCitation(start_index=1, end_index=65, url='https://cloud.google.com/learn/what-is-a-cloud-database', title=None, license=None, publication_date=None)], search_queries=['What are the different types of databases available in Google Cloud?']))



## Cleaning up

To avoid incurring charges to your Google Cloud account for the resources used in this notebook, follow these steps:

1. To avoid unnecessary Google Cloud charges, use the [Google Cloud console](https://console.cloud.google.com/) to delete your project if you do not need it. Learn more in the Google Cloud documentation for [managing and deleting your project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1. If you used an existing Google Cloud project, delete the resources you created to avoid incurring charges to your account. For more information, refer to the documentation to [Delete data from a data store in Vertex AI Search](https://cloud.google.com/generative-ai-app-builder/docs/delete-datastores), then delete your data store.
1. Disable the [Vertex AI Search and Conversation API](https://console.cloud.google.com/apis/api/discoveryengine.googleapis.com) and [Vertex AI API](https://console.cloud.google.com/apis/api/aiplatform.googleapis.com) in the Google Cloud Console.
