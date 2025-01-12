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

# Get started with embeddings tuning on Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro_embeddings_tuning.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fembeddings%2Fintro_embeddings_tuning.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/embeddings/intro_embeddings_tuning.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/intro_embeddings_tuning.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Ivan Nardini](https://github.com/inardini)|

## Overview

This notebook guides you through the process of tuning the text embedding model on Vertex AI. Tuning an embeddings model for specific domains/tasks enhances understanding and improves retrieval performance.

Learn more about [Tune text embeddings](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-embeddings).

### Objective

Large Language Models (LLMs) face challenges in information retrieval due to hallucination, where they generate potentially inaccurate information. Retrieval-Augmented Generation (RAG) addresses this issue by using a retrieval component to identify relevant information in a knowledge base before passing it to the LLM for generation. To improve retrieval effectiveness, meaningful representation of queries and content is crucial, which can be achieved by fine-tuning the embedding model with retrieval-specific domain data. 

In this tutorial, you learn how to tune the text embedding model, `textembedding-gecko`.

This tutorial uses the following Google Cloud ML services and resources:

- Document AI
- Vertex AI
- Google Cloud Storage

The steps include:

- Prepare your model tuning dataset using Document AI, Gemini API, and LangChain on Vertex AI.  
- Run an embedding tuning job on Vertex AI Pipelines.
- Evaluate the embedding tuned model.
- Deploy the embedding tuned model on Vertex AI Prediction.
- Retrieve similar items using the tuned embedding model.

### Dataset

During the tutorial, you will create a set of synthetic query-chunk pairs using the [2023 Q3 Alphabet Earnings Release](https://www.abc.xyz/assets/95/eb/9cef90184e09bac553796896c633/2023q4-alphabet-earnings-release.pdf).

### Costs

This tutorial uses billable components of Google Cloud:

* Document AI
* Vertex AI
* Cloud Storage

Learn about [Document AI pricing](https://cloud.google.com/document-ai/pricing), [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing),
and [Cloud Storage pricing](https://cloud.google.com/storage/pricing),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Installation

Install the following packages required to execute this notebook.


```
%pip install --upgrade --user google-cloud-aiplatform google-cloud-documentai google-cloud-documentai-toolbox --quiet
%pip install --upgrade --user langchain langchain-core langchain-text-splitters langchain-google-community gcsfs etils --quiet
```

### Colab only: Uncomment the following cell to restart the kernel.


```
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

## Before you begin

### Set up your Google Cloud project

**The following steps are required, regardless of your notebook environment.**

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.

2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

3. [Enable APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,documentai.googleapis.com).

4. If you are running this notebook locally, you need to install the [Cloud SDK](https://cloud.google.com/sdk).

#### Set your project ID

**If you don't know your project ID**, try the following:
* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113)


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}

# Set the project id
! gcloud config set project {PROJECT_ID}
```

#### Region

You can also change the `REGION` variable used by Vertex AI. Learn more about [Vertex AI regions](https://cloud.google.com/vertex-ai/docs/general/locations).


```
REGION = "us-central1"  # @param {type: "string"}
```

#### Timestamp

If you are in a live tutorial session, you might be using a shared test account or project. To avoid name collisions between users on resources created, you create a timestamp for each instance session, and append the timestamp onto the name of resources you create in this tutorial.


```
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
```

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.


```
BUCKET_URI = f"gs://your-bucket-name-{PROJECT_ID}-unique"  # @param {type:"string"}
```

**Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.


```
! gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}
```

### Set up tutorial folder

Set up a folder for tutorial content including data, metadata and more.


```
from pathlib import Path as path

root_path = path.cwd()
tutorial_path = root_path / "tutorial"
data_path = tutorial_path / "data"

data_path.mkdir(parents=True, exist_ok=True)
```

### Import libraries

Import libraries to run the tutorial.


```
import random
import string
import time

from etils import epath
from google.api_core.client_options import ClientOptions
from google.cloud import aiplatform, documentai
from google.protobuf.json_format import MessageToDict
import langchain_core
from langchain_core.documents.base import Document
from langchain_google_community.docai import Blob, DocAIParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pandas as pd
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
import vertexai.preview.generative_models as generative_models
```

### Set Variables

Set variables to run the tutorial.


```
ID = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
```


```
# Dataset
PROCESSOR_ID = f"preprocess-docs-llm-{ID}"
LOCATION = REGION.split("-")[0]
RAW_DATA_URI = "gs://github-repo/embeddings/get_started_with_embedding_tuning"
PROCESSED_DATA_URI = f"{BUCKET_URI}/data/processed"
PREPARED_DATA_URI = f"{BUCKET_URI}/data/prepared"
PROCESSED_DATA_OCR_URI = f"{BUCKET_URI}/data/processed/ocr"
PROCESSED_DATA_TUNING_URI = f"{BUCKET_URI}/data/processed/tuning"

# Tuning
PIPELINE_ROOT = f"{BUCKET_URI}/pipelines"
BATCH_SIZE = 32  # @param {type:"integer"}
TRAINING_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"  # @param {type:"string"}
TRAINING_MACHINE_TYPE = "n1-standard-16"  # @param {type:"string"}

# Serving
PREDICTION_ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"  # @param {type:"string"}
PREDICTION_ACCELERATOR_COUNT = 1  # @param {type:"integer"}
PREDICTION_MACHINE_TYPE = "a2-highgpu-1g"  # @param {type:"string"}
```

### Helpers


```
def create_processor(project_id: str, location: str, processor_display_name: str):
    """Create a Document AI processor."""
    client_options = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=client_options)

    parent = client.common_location_path(project_id, location)

    return client.create_processor(
        parent=parent,
        processor=documentai.Processor(
            display_name=processor_display_name, type_="OCR_PROCESSOR"
        ),
    )


def generate_queries(
    chunk: str,
    num_questions: int = 3,
) -> langchain_core.documents.base.Document:
    """A function to generate contextual queries based on preprocessed chunk"""

    model = GenerativeModel("gemini-1.0-pro-001")

    generation_config = GenerationConfig(
        max_output_tokens=2048, temperature=0.9, top_p=1
    )

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    }

    prompt_template = """
    You are an examinator. Your task is to create one QUESTION for an exam using  only.

    
    {chunk}
    

    QUESTION:
    """

    query = prompt_template.format(
        chunk=chunk.page_content, num_questions=num_questions
    )

    for idx in range(num_questions):
        response = model.generate_content(
            [query],
            generation_config=generation_config,
            safety_settings=safety_settings,
        ).text

        return Document(
            page_content=response, metadata={"page": chunk.metadata["page"]}
        )


def get_task_by_name(job: aiplatform.PipelineJob, task_name: str):
    """Get a Vertex AI Pipeline job task by its name"""
    for task in job.task_details:
        if task.task_name == task_name:
            return task
    raise ValueError(f"Task {task_name} not found")


def get_metrics(
    job: aiplatform.PipelineJob, task_name: str = "text-embedding-evaluator"
):
    """Get metrics for the evaluation task"""
    evaluation_task = get_task_by_name(job, task_name)
    metrics = MessageToDict(evaluation_task.outputs["metrics"]._pb)["artifacts"][0][
        "metadata"
    ]
    metrics_df = pd.DataFrame([metrics])
    return metrics_df


def get_uploaded_model(
    job: aiplatform.PipelineJob, task_name: str = "text-embedding-model-uploader"
) -> aiplatform.Model:
    """Get uploaded model from the pipeline job"""
    evaluation_task = get_task_by_name(job, task_name)
    upload_metadata = MessageToDict(evaluation_task.execution._pb)["metadata"]
    return aiplatform.Model(upload_metadata["output:model_resource_name"])


def get_training_output_dir(
    job: aiplatform.PipelineJob, task_name: str = "text-embedding-trainer"
) -> str:
    """Get training output directory for the pipeline job"""
    trainer_task = get_task_by_name(job, task_name)
    output_artifacts = MessageToDict(trainer_task.outputs["training_output"]._pb)[
        "artifacts"
    ][0]
    return output_artifacts["uri"]


def get_top_k_scores(
    query_embedding: pd.DataFrame, corpus_embeddings: pd.DataFrame, k=10
) -> pd.DataFrame:
    """Get top k similar scores for each query"""
    similarity = corpus_embeddings.dot(query_embedding.T)
    topk_index = pd.DataFrame({c: v.nlargest(n=k).index for c, v in similarity.items()})
    return topk_index


def get_top_k_documents(
    query_text: list[str],
    corpus_text: pd.DataFrame,
    corpus_embeddings: pd.DataFrame,
    task_type: str = "RETRIEVAL_DOCUMENT",
    title: str = "",
    k: int = 10,
) -> pd.DataFrame:
    """Get top k similar documents for each query"""
    instances = []
    for text in query_text:
        instances.append(
            {
                "content": text,
                "task_type": task_type,
                "title": title,
            }
        )

    response = endpoint.predict(instances=instances)
    query_embedding = np.asarray(response.predictions)
    topk = get_top_k_scores(query_embedding, corpus_embeddings, k)
    return pd.DataFrame.from_dict(
        {
            query_text[c]: corpus_text.loc[v.values].values.ravel()
            for c, v in topk.items()
        },
        orient="columns",
    )
```

### Initialize Vertex AI SDK for Python

Initialize the Vertex AI SDK for Python for your project.


```
vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
```

## Tuning text embeddings

To tune the model, you should start by preparing your model tuning dataset and then upload it to a Cloud Storage bucket. Text embedding models support supervised tuning, which uses labeled examples to demonstrate the desired output from the model during inference.

Next, you create a model tuning job and deploy the tuned model to a Vertex AI endpoint.

Finally, you retrieve similar items using the tuned embedding model.

### Prepare your model tuning dataset using Document AI, Gemini API, and LangChain on Vertex AI

The tuning dataset consists of the following files:

- `corpus` file is a JSONL file where each line has the fields `_id`, `title` (optional), and `text` of each relevant chunk.

- `query` file is a JSONL file where each line has the fields `_id`, and `text` of each relevant query.

- `labels` files are TSV files (train, test and val) with three columns: `query-id`,`corpus-id`, and `score`. `query-id` represents the query id in the query file, `corpus-id` represents the corpus id in the corpus file, and `score` indicates relevance with higher scores meaning greater relevance. A default score of 1 is used if none is specified. The `train` file is required while `test` and `val` are optional.

#### Create a Document AI preprocessor

Create the OCR processor to identify and extract text in PDF document.


```
processor = create_processor(PROJECT_ID, LOCATION, PROCESSOR_ID)
```

#### Parse the document using Document AI Parser in LangChain

Initiate a LangChain parser.


```
blob = Blob.from_path(
    path=f"{RAW_DATA_URI}/goog-10-k-2023.pdf",
)

parser = DocAIParser(
    processor_name=processor.name,
    location=LOCATION,
    gcs_output_path=PROCESSED_DATA_OCR_URI,
)
```

Run a Google Document AI PDF Batch Processing job.


```
operations = parser.docai_parse([blob])
```


```
while True:
    if parser.is_running(operations):
        print("Waiting for Document AI to finish...")
        time.sleep(10)
    else:
        print("Document AI successfully processed!")
        break
```

Get the resulting LangChain Documents containing the extracted text and metadata.


```
results = parser.get_results(operations)
```


```
docs = list(parser.parse_from_results(results))
```


```
docs[0]
```

#### Create document chunks using `RecursiveCharacterTextSplitter`

You can create chunks using `RecursiveCharacterTextSplitter` in LangChain. The splitter divides text into smaller chunks of a chosen size based on a set of specified characters.

Initiate the splitter.


```
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False,
)
```

Create Text chunks.


```
document_content = [doc.page_content for doc in docs]
document_metadata = [{"page": idx} for idx, doc in enumerate(docs, 1)]
chunks = text_splitter.create_documents(document_content, metadatas=document_metadata)
```

#### Create queries

You can utilize Gemini on Vertex AI to produce hypothetical questions that are relevant to a given piece of context (chunk). 
This approach enables the generation of synthetic positive pairs of (query, relevant documents) in a scalable manner.

Running the query generation would require **some minutes** depending on the number of chunks you have. 


```
generated_queries = [generate_queries(chunk=chunk, num_questions=3) for chunk in chunks]
```

#### Create the tuning training and test dataset files.

Create the `corpus` file.


```
corpus_df = pd.DataFrame(
    {
        "_id": ["text_" + str(idx) for idx in range(len(generated_queries))],
        "text": [chunk.page_content for chunk in chunks],
        "doc_id": [chunk.metadata["page"] for chunk in chunks],
    }
)
```


```
corpus_df.head(10)
```

Create the `query` file.


```
query_df = pd.DataFrame(
    {
        "_id": ["query_" + str(idx) for idx in range(len(generated_queries))],
        "text": [query.page_content for query in generated_queries],
        "doc_id": [query.metadata["page"] for query in generated_queries],
    }
)
```


```
query_df.head(10)
```

Create the `score` file.


```
score_df = corpus_df.merge(query_df, on="doc_id")
score_df = score_df.rename(columns={"_id_x": "corpus-id", "_id_y": "query-id"})
score_df = score_df.drop(columns=["doc_id", "text_x", "text_y"])
score_df["score"] = 1
train_df = score_df.sample(frac=0.8)
test_df = score_df.drop(train_df.index)
```


```
train_df.head(10)
```

#### Save the tuning dataset

Upload the model tuning datasets to a Cloud Storage bucket.


```
corpus_df.to_json(
    f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/corpus.jsonl",
    orient="records",
    lines=True,
)
query_df.to_json(
    f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/query.jsonl", orient="records", lines=True
)
train_df.to_csv(
    f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/train.tsv",
    sep="\t",
    header=True,
    index=False,
)
test_df.to_csv(
    f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/test.tsv",
    sep="\t",
    header=True,
    index=False,
)
```

### Run an embedding tuning job on Vertex AI Pipelines

Next, set the tuning pipeline parameters including the Cloud Storage bucket paths with train and test datasets, the training batch size and the number of steps to perform model tuning. 

For more information about pipeline parameters, [check](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-embeddings#create-embedding-tuning-job) the official tuning documentation.


```
ITERATIONS = len(train_df) // BATCH_SIZE

params = {
    "batch_size": BATCH_SIZE,
    "iterations": ITERATIONS,
    "accelerator_type": TRAINING_ACCELERATOR_TYPE,
    "machine_type": TRAINING_MACHINE_TYPE,
    "base_model_version_id": "textembedding-gecko@003",
    "queries_path": f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/query.jsonl",
    "corpus_path": f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/corpus.jsonl",
    "train_label_path": f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/train.tsv",
    "test_label_path": f"{PROCESSED_DATA_TUNING_URI}/{TIMESTAMP}/test.tsv",
    "project": PROJECT_ID,
    "location": REGION,
}

template_uri = "https://us-kfp.pkg.dev/ml-pipeline/llm-text-embedding/tune-text-embedding-model/v1.1.1"
```

Run the model tuning pipeline job.


```
job = aiplatform.PipelineJob(
    display_name="tune-text-embedding",
    parameter_values=params,
    template_path=template_uri,
    pipeline_root=PIPELINE_ROOT,
    project=PROJECT_ID,
    location=REGION,
)
```


```
job.run()
```

### Evaluate the tuned model

Evaluate the tuned embedding model. The Vertex AI Pipeline automatically produces NDCG (Normalized Discounted Cumulative Gain) for both training and test datasets. 
Tuning the model should results in a NDCG@10 improvement compared with the base textembedding-gecko model which means that top 10 chunks that will be retrieved are now more likely to be exactly the relevant ones for answering the input query. In other words, the most relevant information is now easier to find with the new tuned embedding model. 


```
metric_df = get_metrics(job)
```


```
metric_df.to_dict()
```


```
metric_df
```

### Deploy the embedding tuned model on Vertex AI Prediction

To deploy the embedding tuned model, you need to create a Vertex AI Endpoint.

Then you deploy the tuned embeddings model to the endpoint.

#### Create the endpoint


```
endpoint = aiplatform.Endpoint.create(
    display_name="tuned_custom_embedding_endpoint",
    description="Endpoint for tuned model embeddings.",
    project=PROJECT_ID,
    location=REGION,
)
```

#### Deploy the tuned model

Get the tuned model.


```
model = get_uploaded_model(job)
```

Deploy the tuned model to the endpoint.


```
endpoint.deploy(
    model,
    accelerator_type=PREDICTION_ACCELERATOR_TYPE,
    accelerator_count=PREDICTION_ACCELERATOR_COUNT,
    machine_type=PREDICTION_MACHINE_TYPE,
)
```

### Retrieve similar items using the tuned embedding model

To retrieve similar items using the tuned embedding model, you need both the corpus text and the generated embeddings. Given a query, you will calculate the associated embeddings with the tuned model and you will apply a similarity function to find the most relevant document with respect the query.   

Read the corpus text and the generated embeddings.


```
training_output_dir = get_training_output_dir(job)
```


```
corpus_text = pd.read_json(
    epath.Path(training_output_dir) / "corpus_text.jsonl", lines=True
)
corpus_text.head()
```


```
corpus_embeddings = pd.read_json(
    epath.Path(training_output_dir) / "corpus_custom.jsonl", lines=True
)

corpus_embeddings.head()
```

Find the most relevant documents for each query.


```
queries = [
    """What about the revenues?""",
    """Who is Alphabet?""",
    """What about the costs?""",
]
output = get_top_k_documents(queries, corpus_text, corpus_embeddings, k=10)

with pd.option_context("display.max_colwidth", 200):
    display(output)
```

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial.


```
import shutil

delete_endpoint = False
delete_model = False
delete_job = False
delete_bucket = False
delete_tutorial = False

# Delete endpoint resource
if delete_endpoint:
    endpoint.delete(force=True)

# Delete model resource
if delete_model:
    model.delete()

# Delete pipeline job
if delete_job:
    job.delete()

# Delete Cloud Storage objects that were created
if delete_bucket:
    ! gsutil -m rm -r $BUCKET_URI

# Delete tutorial folder
if delete_tutorial:
    shutil.rmtree(str(tutorial_path))
```
