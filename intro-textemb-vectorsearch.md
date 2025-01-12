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

# Getting Started with Text Embeddings + Vertex AI Vector Search


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro-textemb-vectorsearch.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fembeddings%2Fintro-textemb-vectorsearch.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/embeddings/intro-textemb-vectorsearch.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro-textemb-vectorsearch.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro-textemb-vectorsearch.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Smitha Venkat](https://github.com/smitha-google), [Kaz Sato](https://github.com/kazunori279)|

## Introduction

In this tutorial, you learn how to use Google Cloud AI tools to quickly bring the power of Large Language Models to enterprise systems.  

This tutorial covers the following -

*   What are embeddings - what business challenges do they help solve ?
*   Understanding Text with Vertex AI Text Embeddings
*   Find Embeddings fast with Vertex AI Vector Search
*   Grounding LLM outputs with Vector Search

This tutorial is based on [the blog post](https://cloud.google.com/blog/products/ai-machine-learning/how-to-use-grounding-for-your-llms-with-text-embeddings), combined with sample code.


### Prerequisites

This tutorial is designed for developers who has basic knowledge and experience with Python programming and machine learning.

If you are not reading this tutorial in Qwiklab, then you need to have a Google Cloud project that is linked to a billing account to run this. Please go through [this document](https://cloud.google.com/vertex-ai/docs/start/cloud-environment) to create a project and setup a billing account for it.

### Choose the runtime environment

The notebook can be run on either Google Colab or [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).

- To use Colab: Click [this link](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro-textemb-vectorsearch.ipynb) to open the tutorial in Colab.

- To use Workbench: If it is the first time to use Workbench in your Google Cloud project, open [the Workbench console](https://console.cloud.google.com/vertex-ai/workbench) and click ENABLE button to enable Notebooks API. Then click [this link](https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/embeddings/intro-textemb-vectorsearch.ipynb),  and select an existing notebook or create a new notebook.


### How much will this cost?

In case you are using your own Cloud project, not a temporary project on Qwiklab, you need to spend roughly a few US dollars to finish this tutorial.

The pricing of the Cloud services we will use in this tutorial are available in the following pages:

- [Vertex AI Embeddings for Text](https://cloud.google.com/vertex-ai/pricing#generative_ai_models)
- [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/pricing#matchingengine)
- [BigQuery](https://cloud.google.com/bigquery/pricing)
- [Cloud Storage](https://cloud.google.com/storage/pricing)
- [Vertex AI Workbench](https://cloud.google.com/vertex-ai/pricing#notebooks) if you use one

You can use the [Pricing Calculator](https://cloud.google.com/products/calculator) to generate a cost estimate based on your projected usage. The following is an example of rough cost estimation with the calculator, assuming you will go through this tutorial a couple of time.

<img src="https://storage.googleapis.com/github-repo/img/embeddings/vs-quickstart/pricing.png" width="50%"/>

### **Warning: delete your objects after the tutorial**

In case you are using your own Cloud project, please make sure to delete all the Indexes, Index Endpoints and Cloud Storage buckets (and the Workbench instance if you use one) after finishing this tutorial. Otherwise the remaining assets would incur unexpected costs.


# Bringing Gen AI and LLMs to production services

Many people are now starting to think about how to bring Gen AI and LLMs to production services, and facing with several challenges.

- "How to integrate LLMs or AI chatbots with existing IT systems, databases and business data?"
- "We have thousands of products. How can I let LLM memorize them all precisely?"
- "How to handle the hallucination issues in AI chatbots to build a reliable service?"

Here is a quick solution: **grounding** with **embeddings** and **vector search**.

What is grounding? What are embedding and vector search? In this tutorial, we will learn these crucial concepts to build reliable Gen AI services for enterprise use. But before we dive deeper, let's try the demo below.

![](https://storage.googleapis.com/gweb-cloudblog-publish/original_images/1._demo_animation.gif)

**Exercise: Try the Stack Overflow semantic search demo:**

This demo is available as a [public live demo](https://ai-demos.dev/). Select "STACKOVERFLOW" and enter any coding question as a query, so it runs a text search on **8 million** questions posted on [Stack Overflow](https://stackoverflow.com/). Try the text semantic search with some queries like 'How to shuffle rows in SQL?' or arbitrary programming questions.

In this tutorial, we are going to see how to build a similar search experience - what is involved in building solutions like this using Vertex AI Embeddings API and Vector Search.

# What is Embeddings?

With the rise of LLMs, why is it becoming important for IT engineers and ITDMs to understand how they work?

In traditional IT systems, most data is organized as structured or tabular data, using simple keywords, labels, and categories in databases and search engines.

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/1.png)

In contrast, AI-powered services arrange data into a simple data structure known as "embeddings."

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/2.png)

Once trained with specific content like text, images, or any content, AI creates a space called "embedding space", which is essentially a map of the content's meaning.

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/3.png)

AI can identify the location of each content on the map, that's what embedding is.

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/4.png)

Let's take an example where a text discusses movies, music, and actors, with a distribution of 10%, 2%, and 30%, respectively. In this case, the AI can create an embedding with three values: 0.1, 0.02, and 0.3, in 3 dimensional space.

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/5.png)

AI can put content with similar meanings closely together in the space.

This is how Google organizes data across various services like Google Search, YouTube, Play, and many others, to provide search results and recommendations with relevant content.

Embeddings can also be used to represent different types of things in businesses, such as products, users, user activities, conversations, music & videos, signals from IoT sensors, and so on.

AI and Embeddings are now playing a crucial role in creating a new way of human-computer interaction.

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/6.png)

AI organizes data into embeddings, which represent what the user is looking for, the meaning of contents, or many other things you have in your business. This creates a new level of user experience that is becoming the new standard.

To learn more about embeddings, [Foundational courses: Embeddings on Google Machine Learning Crush Course](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) and [Meet AI's multitool: Vector embeddings by Dale Markowitz](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings) are great materials.


# Vertex AI Embeddings for Text

With the [Vertex AI Embeddings for Text](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings), you can easily create a text embedding with LLM. The product is also available on [Vertex AI Model Garden](https://cloud.google.com/model-garden)

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/7.png)

This API is designed to extract embeddings from texts. It can take text input up to 2048 input tokens, and outputs 768 dimensional text embeddings.

## LLM text embedding business use cases

With the embedding API, you can apply the innovation of embeddings, combined with the LLM capability, to various text processing tasks, such as:

**LLM-enabled Semantic Search**: text embeddings can be used to represent both the meaning and intent of a user's query and documents in the embedding space. Documents that have similar meaning to the user's query intent will be found fast with vector search technology. The model is capable of generating text embeddings that capture the subtle nuances of each sentence and paragraphs in the document.

**LLM-enabled Text Classification**: LLM text embeddings can be used for text classification with a deep understanding of different contexts without any training or fine-tuning (so-called zero-shot learning). This wasn't possible with the past language models without task-specific training.

**LLM-enabled Recommendation**: The text embedding can be used for recommendation systems as a strong feature for training recommendation models such as Two-Tower model. The model learns the relationship between the query and candidate embeddings, resulting in next-gen user experience with semantic product recommendation.

LLM-enabled Clustering, Anomaly Detection, Sentiment Analysis, and more, can be also handled with the LLM-level deep semantics understanding.


## Sorting 8 million texts at "librarian-level" precision

Vertex AI Embeddings for Text has an embedding space with 768 dimensions. As explained earlier, the space represents a huge map of a wide variety of texts in the world, organized by their meanings. With each input text, the model can find a location (embedding) in the map.

By visualizing the embedding space, you can actually observe how the model sorts the texts at the "librarian-level" precision.

**Exercise: Try the Nomic AI Atlas**

[Nomic AI](http://nomic.ai/) provides a platform called Atlas for storing, visualizing and interacting with embedding spaces with high scalability and in a smooth UI, and they worked with Google for visualizing the embedding space of the 8 million Stack Overflow questions. You can try exploring around the space, zooming in and out to each data point on your browser on this page, courtesy of Nomic AI.

The embedding space represents a huge map of texts, organized by their meanings
With each input text, the model can find a location (embedding) in the map
Like a librarian reading through millions of texts, sorting them with millions of nano-categories

Try exploring it [here](https://atlas.nomic.ai/map/edaff028-12b5-42a0-8e8b-6430c9b8222b/bcb42818-3581-4fb5-ac30-9883d01f98ec). Zoom into a few categories, point each dots, and see how the LLM is sorting similar questions close together in the space.

![](https://storage.googleapis.com/gweb-cloudblog-publish/images/4._Nomic_AI_Atlas.max-2200x2200.png)

### The librarian-level semantic understanding

Here are the examples of the librarian-level semantic understanding by Embeddings API with Stack Overflow questions.

![](https://storage.googleapis.com/gweb-cloudblog-publish/images/5._semantic_understanding.max-2200x2200.png)

For example, the model thinks the question "Does moving the request line to a header frame require an app change?" is similar to the question "Does an application developed on HTTP1x require modifications to run on HTTP2?". That is because The model knows both questions talk about what's the change required to support the HTTP2 header frame.

Note that this demo didn't require any training or fine-tuning with computer programming specific datasets. This is the innovative part of the zero-shot learning capability of the LLM. It can be applied to a wide variety of industries, including finance, healthcare, retail, manufacturing, construction, media, and more, for deep semantic search on the industry-focused business documents without spending time and cost for collecting industry specific datasets and training models.

# Text Embeddings in Action

Lets try using Text Embeddings in action with actual sample code.

## Setup

Before get started with the Vertex AI services, we need to setup the following.

* Install Python SDK
* Environment variables
* Authentication (Colab only)
* Enable APIs
* Set IAM permissions

### Install Python SDK

Vertex AI, Cloud Storage and BigQuery APIs can be accessed with multiple ways including REST API and Python SDK. In this tutorial we will use the SDK.


```
%pip install --upgrade --user google-cloud-aiplatform google-cloud-storage 'google-cloud-bigquery[pandas]'
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


### Environment variables

Sets environment variables. If asked, please replace the following `[your-project-id]` with your project ID and run it.


```
# get project ID
PROJECT_ID = ! gcloud config get project
PROJECT_ID = PROJECT_ID[0]
LOCATION = "us-central1"
if PROJECT_ID == "(unset)":
    print(f"Please set the project ID manually below")
```


```
# define project information
if PROJECT_ID == "(unset)":
    PROJECT_ID = "[your-project-id]"  # @param {type:"string"}

# generate an unique id for this session
from datetime import datetime

UID = datetime.now().strftime("%m%d%H%M")
```

### Authentication (Colab only)

If you are running this notebook on Colab, you will need to run the following cell authentication. This step is not required if you are using Vertex AI Workbench as it is pre-authenticated.


```
import sys

# if it's Colab runtime, authenticate the user with Google Cloud
if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Enable APIs

Run the following to enable APIs for Compute Engine, Vertex AI, Cloud Storage and BigQuery with this Google Cloud project.


```
! gcloud services enable compute.googleapis.com aiplatform.googleapis.com storage.googleapis.com bigquery.googleapis.com --project {PROJECT_ID}
```

### Set IAM permissions

Also, we need to add access permissions to the default service account for using those services.

- Go to [the IAM page](https://console.cloud.google.com/iam-admin/) in the Console
- Look for the principal for default compute service account. It should look like: `<project-number>-compute@developer.gserviceaccount.com`
- Click the edit button at right and click `ADD ANOTHER ROLE` to add `Vertex AI User`, `BigQuery User` and `Storage Admin` to the account.

This will look like this:

![](https://storage.googleapis.com/github-repo/img/embeddings/vs-quickstart/iam-setting.png)

## Getting Started with Vertex AI Embeddings for Text

Now it's ready to get started with embeddings!

### Data Preparation

We will be using [the Stack Overflow public dataset](https://console.cloud.google.com/marketplace/product/stack-exchange/stack-overflow) hosted on BigQuery table `bigquery-public-data.stackoverflow.posts_questions`. This is a very big dataset with 23 million rows that doesn't fit into the memory. We are going to limit it to 1000 rows for this tutorial.


```
# load the BQ Table into a Pandas DataFrame
from google.cloud import bigquery

QUESTIONS_SIZE = 1000

bq_client = bigquery.Client(project=PROJECT_ID)
QUERY_TEMPLATE = """
        SELECT distinct q.id, q.title
        FROM (SELECT * FROM `bigquery-public-data.stackoverflow.posts_questions`
        where Score > 0 ORDER BY View_Count desc) AS q
        LIMIT {limit} ;
        """
query = QUERY_TEMPLATE.format(limit=QUESTIONS_SIZE)
query_job = bq_client.query(query)
rows = query_job.result()
df = rows.to_dataframe()

# examine the data
df.head()
```

### Call the API to generate embeddings

With the Stack Overflow dataset, we will use the `title` column (the question title) and generate embedding for it with Embeddings for Text API. The API is available under the [vertexai](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai) package of the SDK.

You may see some warning messages from the TensorFlow library but you can ignore them.


```
# init the vertexai package
import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

From the package, import [TextEmbeddingModel](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.language_models.TextEmbeddingModel) and get a model.


```
# Load the text embeddings model
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("text-embedding-004")
```

In this tutorial we will use `text-embedding-004` model for getting text embeddings. Please take a look at [Supported models](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings#supported_models) on the doc to see the list of supported models.

Once you get the model, you can call its [get_embeddings](https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.language_models.TextEmbeddingModel#vertexai_language_models_TextEmbeddingModel_get_embeddings) function to get embeddings. You can pass up to 5 texts at once in a call. But there is a caveat. By default, the text embeddings API has a "request per minute" quota set to 60 for new Cloud projects and 600 for projects with usage history (see [Quotas and limits](https://cloud.google.com/vertex-ai/docs/quotas#request_quotas) to check the latest quota value for `base_model:textembedding-gecko`). So, rather than using the function directly, you may want to define a wrapper like below to limit under 10 calls per second, and pass 5 texts each time.


```
import time

import tqdm  # to show a progress bar

# get embeddings for a list of texts
BATCH_SIZE = 5


def get_embeddings_wrapper(texts):
    embs = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH_SIZE)):
        time.sleep(1)  # to avoid the quota error
        result = model.get_embeddings(texts[i : i + BATCH_SIZE])
        embs = embs + [e.values for e in result]
    return embs
```

The following code will get embedding for the question titles and add them as a new column `embedding` to the DataFrame. This will take a few minutes.


```
# get embeddings for the question titles and add them as "embedding" column
df = df.assign(embedding=get_embeddings_wrapper(list(df.title)))
df.head()
```

## Look at the embedding similarities

Let's see how these embeddings are organized in the embedding space with their meanings by quickly calculating the similarities between them and sorting them.

As embeddings are vectors, you can calculate similarity between two embeddings by using one of the popular metrics like the followings:

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/8.png)

Which metric should we use? Usually it depends on how each model is trained. In case of the model `text-embedding-004`, we need to use inner product (dot product).

In the following code, it picks up one question randomly and uses the numpy `np.dot` function to calculate the similarities between the question and other questions.


```
import random

import numpy as np

# pick one of them as a key question
key = random.randint(0, len(df))

# calc dot product between the key and other questions
embs = np.array(df.embedding.to_list())
similarities = np.dot(embs[key], embs.T)

# print similarities for the first 5 questions
similarities[:5]
```

Finally, sort the questions with the similarities and print the list.


```
# print the question
print(f"Key question: {df.title[key]}\n")

# sort and print the questions by similarities
sorted_questions = sorted(
    zip(df.title, similarities), key=lambda x: x[1], reverse=True
)[:20]
for i, (question, similarity) in enumerate(sorted_questions):
    print(f"{similarity:.4f} {question}")
```

# Find embeddings fast with Vertex AI Vector Search

As we have explained above, you can find similar embeddings by calculating the distance or similarity between the embeddings.

But this isn't easy when you have millions or billions of embeddings. For example, if you have 1 million embeddings with 768 dimensions, you need to repeat the distance calculations for 1 million x 768 times. This would take some seconds - too slow.

So the researchers have been studying a technique called [Approximate Nearest Neighbor (ANN)](https://en.wikipedia.org/wiki/Nearest_neighbor_search) for faster search. ANN uses "vector quantization" for separating the space into multiple spaces with a tree structure. This is similar to the index in relational databases for improving the query performance, enabling very fast and scalable search with billions of embeddings.

With the rise of LLMs, the ANN is getting popular quite rapidly, known as the Vector Search technology.

![](https://storage.googleapis.com/gweb-cloudblog-publish/images/7._ANN.1143068821171228.max-2200x2200.png)

In 2020, Google Research published a new ANN algorithm called [ScaNN](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html). It is considered one of the best ANN algorithms in the industry, also the most important foundation for search and recommendation in major Google services such as Google Search, YouTube and many others.


## What is Vertex AI Vector Search?

Google Cloud developers can take the full advantage of Google's vector search technology with [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) (previously called Matching Engine). With this fully managed service, developers can just add the embeddings to its index and issue a search query with a key embedding for the blazingly fast vector search. In the case of the Stack Overflow demo, Vector Search can find relevant questions from 8 million embeddings in tens of milliseconds.

![](https://storage.googleapis.com/github-repo/img/embeddings/textemb-vs-notebook/9.png)

With Vector Search, you don't need to spend much time and money building your own vector search service from scratch or using open source tools if your goal is high scalability, availability and maintainability for production systems.

## Get Started with Vector Search

When you already have the embeddings, then getting started with Vector Search is pretty easy. In this section, we will follow the steps below.

### Setting up Vector Search
- Save the embeddings in JSON files on Cloud Storage
- Build an Index
- Create an Index Endpoint
- Deploy the Index to the endpoint

### Use Vector Search

- Query with the endpoint

### **Tip for Colab users**

If you use Colab for this tutorial, you may lose your runtime while you are waiting for the Index building and deployment in the later sections as it takes tens of minutes. In that case, run the following sections again with the new instance to recover the runtime: [Install Python SDK, Environment variables and Authentication](https://colab.research.google.com/drive/1xJhLFEyPqW0qvKiERD6aYgeTHa6_U50N?resourcekey=0-2qUkxckCjt6W03AsqvZHhw#scrollTo=AtXnXhF8U-8R&line=9&uniqifier=1).

Then, use the [Utilities](https://colab.research.google.com/drive/1xJhLFEyPqW0qvKiERD6aYgeTHa6_U50N?resourcekey=0-2qUkxckCjt6W03AsqvZHhw#scrollTo=BE1tELsH-u8N&line=1&uniqifier=1) to recover the Index and Index Endpoint and continue with the rest.

### Save the embeddings in a JSON file
To load the embeddings to Vector Search, we need to save them in JSON files with JSONL format. See more information in the docs at [Input data format and structure](https://cloud.google.com/vertex-ai/docs/matching-engine/match-eng-setup/format-structure#data-file-formats).

First, export the `id` and `embedding` columns from the DataFrame in JSONL format, and save it.


```
# save id and embedding as a json file
jsonl_string = df[["id", "embedding"]].to_json(orient="records", lines=True)
with open("questions.json", "w") as f:
    f.write(jsonl_string)

# show the first few lines of the json file
! head -n 3 questions.json
```

Then, create a new Cloud Storage bucket and copy the file to it.


```
BUCKET_URI = f"gs://{PROJECT_ID}-embvs-tutorial-{UID}"
! gsutil mb -l $LOCATION -p {PROJECT_ID} {BUCKET_URI}
! gsutil cp questions.json {BUCKET_URI}
```

### Create an Index

Now it's ready to load the embeddings to Vector Search. Its APIs are available under the [aiplatform](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform) package of the SDK.


```
# init the aiplatform package
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)
```

Create an [MatchingEngineIndex](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.MatchingEngineIndex) with its `create_tree_ah_index` function (Matching Engine is the previous name of Vector Search).


```
# create index
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=f"embvs-tutorial-index-{UID}",
    contents_delta_uri=BUCKET_URI,
    dimensions=768,
    approximate_neighbors_count=20,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)
```

By calling the `create_tree_ah_index` function, it starts building an Index. This will take under a few minutes if the dataset is small, otherwise about 50 minutes or more depending on the size of the dataset. You can check status of the index creation on [the Vector Search Console > INDEXES tab](https://console.cloud.google.com/vertex-ai/matching-engine/indexes).

![](https://storage.googleapis.com/github-repo/img/embeddings/vs-quickstart/creating-index.png)

#### The parameters for creating index

- `contents_delta_uri`: The URI of Cloud Storage directory where you stored the embedding JSON files
- `dimensions`: Dimension size of each embedding. In this case, it is 768 as we are using the embeddings from the Text Embeddings API.
- `approximate_neighbors_count`: how many similar items we want to retrieve in typical cases
- `distance_measure_type`: what metrics to measure distance/similarity between embeddings. In this case it's `DOT_PRODUCT_DISTANCE`

See [the document](https://cloud.google.com/vertex-ai/docs/vector-search/create-manage-index) for more details on creating Index and the parameters.

#### Batch Update or Streaming Update?
There are two types of index: Index for *Batch Update* (used in this tutorial) and Index for *Streaming Updates*. The Batch Update index can be updated with a batch process whereas the Streaming Update index can be updated in real-time. The latter one is more suited for use cases where you want to add or update each embeddings in the index more often, and crucial to serve with the latest embeddings, such as e-commerce product search.


### Create Index Endpoint and deploy the Index

To use the Index, you need to create an [Index Endpoint](https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-public). It works as a server instance accepting query requests for your Index.


```
# create IndexEndpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"embvs-tutorial-index-endpoint-{UID}",
    public_endpoint_enabled=True,
)
```

This tutorial utilizes a [Public Endpoint](https://cloud.google.com/vertex-ai/docs/vector-search/setup/setup#choose-endpoint) and does not support [Virtual Private Cloud (VPC)](https://cloud.google.com/vpc/docs/private-services-access). Unless you have a specific requirement for VPC, we recommend using a Public Endpoint. Despite the term "public" in its name, it does not imply open access to the public internet. Rather, it functions like other endpoints in Vertex AI services, which are secured by default through IAM. Without explicit IAM permissions, as we have previously established, no one can access the endpoint.

With the Index Endpoint, deploy the Index by specifying an unique deployed index ID.


```
DEPLOYED_INDEX_ID = f"embvs_tutorial_deployed_{UID}"
```


```
# deploy the Index to the Index Endpoint
my_index_endpoint.deploy_index(index=my_index, deployed_index_id=DEPLOYED_INDEX_ID)
```

If it is the first time to deploy an Index to an Index Endpoint, it will take around 25 minutes to automatically build and initiate the backend for it. After the first deployment, it will finish in seconds. To see the status of the index deployment, open [the Vector Search Console > INDEX ENDPOINTS tab](https://console.cloud.google.com/vertex-ai/matching-engine/index-endpoints) and click the Index Endpoint.

<img src="https://storage.googleapis.com/github-repo/img/embeddings/vs-quickstart/deploying-index.png" width="70%">

### Run Query

Finally it's ready to use Vector Search. In the following code, it creates an embedding for a test question, and find similar question with the Vector Search.


```
test_embeddings = get_embeddings_wrapper(["How to read JSON with Python?"])
```


```
# Test query
response = my_index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_INDEX_ID,
    queries=test_embeddings,
    num_neighbors=20,
)

# show the result
import numpy as np

for idx, neighbor in enumerate(response[0]):
    id = np.int64(neighbor.id)
    similar = df.query("id == @id", engine="python")
    print(f"{neighbor.distance:.4f} {similar.title.values[0]}")
```

The `find_neighbors` function only takes milliseconds to fetch the similar items even when you have billions of items on the Index, thanks to the ScaNN algorithm. Vector Search also supports [autoscaling](https://cloud.google.com/vertex-ai/docs/vector-search/deploy-index-public#autoscaling) which can automatically resize the number of nodes based on the demands of your workloads.

# IMPORTANT: Cleaning Up

In case you are using your own Cloud project, not a temporary project on Qwiklab, please make sure to delete all the Indexes, Index Endpoints and Cloud Storage buckets after finishing this tutorial. Otherwise the remaining objects would **incur unexpected costs**.

If you used Workbench, you may also need to delete the Notebooks from [the console](https://console.cloud.google.com/vertex-ai/workbench).


```
# wait for a confirmation
input("Press Enter to delete Index Endpoint, Index and Cloud Storage bucket:")

# delete Index Endpoint
my_index_endpoint.undeploy_all()
my_index_endpoint.delete(force=True)

# delete Index
my_index.delete()

# delete Cloud Storage bucket
! gsutil rm -r {BUCKET_URI}
```

# Summary

## Grounding LLM outputs with Vertex AI Vector Search

As we have seen, by combining the Embeddings API and Vector Search, you can use the embeddings to "ground" LLM outputs to real business data with low latency.

For example, if an user asks a question, Embeddings API can convert it to an embedding, and issue a query on Vector Search to find similar embeddings in its index. Those embeddings represent the actual business data in the databases. As we are just retrieving the business data and not generating any artificial texts, there is no risk of having hallucinations in the result.

![](https://storage.googleapis.com/gweb-cloudblog-publish/original_images/10._grounding.png)

### The difference between the questions and answers

In this tutorial, we have used the Stack Overflow dataset. There is a reason why we had to use it; As the dataset has many pairs of **questions and answers**, so you can just find questions similar to your question to find answers to it.

In many business use cases, the semantics (meaning) of questions and answers are different. Also, there could be cases where you would want to add variety of recommended or personalized items to the results, like product search on e-commerce sites.

In these cases, the simple semantics search don't work well. It's more like a recommendation system problem where you may want to train a model (e.g. Two-Tower model) to learn the relationship between the question embedding space and answer embedding space. Also, many production systems adds reranking phase after the semantic search to achieve higher search quality. Please see [Scaling deep retrieval with TensorFlow Recommenders and Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture) to learn more.

### Hybrid of semantic + keyword search

Another typical challenge you will face in production system is to support keyword search combined with the semantic search. For example, for e-commerce product search, you may want to let users find product by entering its product name or model number. As LLM doesn't memorize those product names or model numbers, semantic search can't handle those "usual" search functionalities.

[Vertex AI Search](https://cloud.google.com/blog/products/ai-machine-learning/vertex-ai-search-and-conversation-is-now-generally-available) is another product you may consider for those requirements. While Vector Search provides a simple semantic search capability only, Search provides a integrated search solution that combines semantic search, keyword search, reranking and filtering, available as an out-of-the-box tool.

### What about Retrieval Augmented Generation (RAG)?

In this tutorial, we have looked at the simple combination of LLM embeddings and vector search. From this starting point, you may also extend the design to [Retrieval Augmented Generation (RAG)](https://www.google.com/search?q=Retrieval+Augmented+Generation+(RAG)&oq=Retrieval+Augmented+Generation+(RAG)).

RAG is a popular architecture pattern of implementing grounding with LLM with text chat UI. The idea is to have the LLM text chat UI as a frontend for the document retrieval with vector search and summarization of the result.

![](https://storage.googleapis.com/gweb-cloudblog-publish/images/Figure-7-Ask_Your_Documents_Flow.max-529x434.png)

There are some pros and cons between the two solutions.

| | Embeddings + vector search | RAG |
|---|---|---|
| Design | simple | complex |
| UI | Text search UI | Text chat UI |
| Summarization of result | No | Yes |
| Multi-turn (Context aware) | No | Yes |
| Latency | milliseconds | seconds |
| Cost | lower | higher |
| Hallucinations | No risk | Some risk |

The Embedding + vector search pattern we have looked at with this tutorial provides simple, fast and low cost semantic search functionality with the LLM intelligence. RAG adds context-aware text chat experience and result summarization to it. While RAG provides the more "Gen AI-ish" experience, it also adds a risk of hallucination and higher cost and time for the text generation.

To learn more about how to build a RAG solution, you may look at [Building Generative AI applications made easy with Vertex AI PaLM API and LangChain](https://cloud.google.com/blog/products/ai-machine-learning/generative-ai-applications-with-vertex-ai-palm-2-models-and-langchain).

## Resources

To learn more, please check out the following resources:

### Documentations

[Vertex AI Embeddings for Text API documentation
](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)

[Vector Search documentation](https://cloud.google.com/vertex-ai/docs/matching-engine/overview)

### Vector Search blog posts

[Vertex Matching Engine: Blazing fast and massively scalable nearest neighbor search](https://cloud.google.com/blog/products/ai-machine-learning/vertex-matching-engine-blazing-fast-and-massively-scalable-nearest-neighbor-search)

[Find anything blazingly fast with Google's vector search technology](https://cloud.google.com/blog/topics/developers-practitioners/find-anything-blazingly-fast-googles-vector-search-technology)

[Enabling real-time AI with Streaming Ingestion in Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/real-time-ai-with-google-cloud-vertex-ai)

[Mercari leverages Google's vector search technology to create a new marketplace](https://cloud.google.com/blog/topics/developers-practitioners/mercari-leverages-googles-vector-search-technology-create-new-marketplace)

[Recommending news articles using Vertex AI Matching Engine](https://cloud.google.com/blog/products/ai-machine-learning/recommending-articles-using-vertex-ai-matching-engine)

[What is Multimodal Search: "LLMs with vision" change businesses](https://cloud.google.com/blog/products/ai-machine-learning/multimodal-generative-ai-search)

# Utilities

Sometimes it takes tens of minutes to create or deploy Indexes and you would lose connection with the Colab runtime. In that case, instead of creating or deploying new Index again, you can check [the Vector Search Console](https://console.cloud.google.com/vertex-ai/matching-engine/index-endpoints) and get the existing ones to continue.

## Get an existing Index

To get an Index object that already exists, replace the following `[your-index-id]` with the index ID and run the cell. You can check the ID on [the Vector Search Console > INDEXES tab](https://console.cloud.google.com/vertex-ai/matching-engine/indexes).


```
my_index_id = "[your-index-id]"  # @param {type:"string"}
my_index = aiplatform.MatchingEngineIndex(my_index_id)
```

## Get an existing Index Endpoint

To get an Index Endpoint object that already exists, replace the following `[your-index-endpoint-id]` with the Index Endpoint ID and run the cell. You can check the ID on [the Vector Search Console > INDEX ENDPOINTS tab](https://console.cloud.google.com/vertex-ai/matching-engine/index-endpoints).


```
my_index_endpoint_id = "[your-index-endpoint-id]"  # @param {type:"string"}
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(my_index_endpoint_id)
```
