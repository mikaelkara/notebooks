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

# Intro to Batch Predictions with the Gemini API using BigQuery input


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction_using_bigquery_input.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fbatch-prediction%2Fintro_batch_prediction_using_bigquery_input.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/batch-prediction/intro_batch_prediction_using_bigquery_input.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction_using_bigquery_input.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction_using_bigquery_input.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong) |

## Overview

Different from getting online (synchronous) responses, where you are limited to one input request at a time, the batch predictions with the Gemini API in Vertex AI allow you to send a large number of multimodal requests to a Gemini model in a single batch request. Then, the model responses asynchronously populate to your storage output location in [Cloud Storage](https://cloud.google.com/storage/docs/introduction) or [BigQuery](https://cloud.google.com/bigquery/docs/storage_overview).

Batch predictions are generally more efficient and cost-effective than online predictions when processing a large number of inputs that are not latency sensitive.

To learn more, see the [Get batch predictions for Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) page.

### Objectives

In this tutorial, you learn how to make batch predictions with the Gemini API in Vertex AI. This tutorial uses **BigQuery** as an input source and an output location.

You will complete the following tasks:

- Preparing batch inputs and an output location
- Submitting a batch prediction job
- Retrieving batch prediction results


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

### Set Google Cloud project information

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
import os

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
```

## Code Examples

### Import libraries


```
from datetime import datetime
import time

from google.cloud import bigquery
import vertexai
from vertexai.batch_prediction import BatchPredictionJob
from vertexai.generative_models import GenerativeModel
```

### Initialize Vertex AI SDK


```
vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Load model

You can find a list of the Gemini models that support batch predictions in the [Multimodal models that support batch predictions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#multimodal_models_that_support_batch_predictions) page.

This tutorial uses the Gemini 1.5 Pro (`gemini-1.5-pro-002`) model.


```
MODEL_ID = "gemini-1.5-pro-002"  # @param {type:"string", isTemplate: true}

model = GenerativeModel(MODEL_ID)
```

### Prepare batch inputs

The input for batch requests specifies the items to send to your model for prediction.

Batch requests for Gemini accept BigQuery storage sources and Cloud Storage sources. You can learn more about the batch input formats for BigQuery and Cloud Storage sources in the [Batch text generation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#prepare_your_inputs) page.

This tutorial uses **BigQuery** as an example. To use a BigQuery table as the input, you must ensure the following:

- The BigQuery dataset must be created in a specific region (e.g. `us-central1`). Multi-region location (e.g. `US`) is not supported.
- The input table must have a column named `request` in `JSON` or `STRING` type.
- The content in the `request` column must be valid JSON. This JSON data represents your input for the model.
- The content in the JSON instructions must match the structure of a [GenerateContentRequest](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference).
- Your input table can have column data types other than `request`. These columns can have BigQuery data types except for the following: `array`, `struct`, `range`, `datetime`, and `geography`. These columns are ignored for content generation but included in the output table. The system reserves two column names for output: `response` and `status`. These are used to provide information about the outcome of the batch prediction job.
- Only public YouTube and Cloud Storage bucket URIs in the `fileData` or `file_data` field are supported in batch prediction.
- Each request that you send to a model can include parameters that control how the model generates a response. Learn more about Gemini parameters in the [Experiment with parameter values](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values) page.


This is an example BigQuery table:


```
INPUT_DATA = "bq://storage-samples.generative_ai.batch_requests_for_multimodal_input_2"  # @param {type:"string"}
```

You can query the BigQuery table to review the input data.


```
bq_client = bigquery.Client(project=PROJECT_ID)

bq_table_id = INPUT_DATA.replace("bq://", "")
sql = f"""
        SELECT *
        FROM {bq_table_id}
        """

query_result = bq_client.query(sql)

df = query_result.result().to_dataframe()
df.head()
```

### Prepare batch output location

When a batch prediction task completes, the output is stored in the location that you specified in your request.

- The location is in the form of a Cloud Storage or BigQuery URI prefix, for example:
`gs://path/to/output/data` or `bq://projectId.bqDatasetId`.

- If not specified, `STAGING_BUCKET/gen-ai-batch-prediction` will be used for Cloud Storage source and `bq://PROJECT_ID.gen_ai_batch_prediction.predictions_TIMESTAMP` will be used for BigQuery source.

This tutorial uses a **BigQuery** table as an example.

- You can specify the URI of your BigQuery table in `BQ_OUTPUT_URI`, or
- if it is not specified, this tutorial will create a new dataset `bq://PROJECT_ID.gen_ai_batch_prediction` for you.


```
BQ_OUTPUT_URI = "[your-bigquery-table]"  # @param {type:"string"}

if BQ_OUTPUT_URI == "[your-bigquery-table]":
    bq_dataset_id = "gen_ai_batch_prediction"

    # The output table will be created automatically if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    bq_table_id = f"prediction_result_{timestamp}"
    BQ_OUTPUT_URI = f"bq://{PROJECT_ID}.{bq_dataset_id}.{bq_table_id}"

    bq_dataset = bigquery.Dataset(f"{PROJECT_ID}.{bq_dataset_id}")
    bq_dataset.location = "us-central1"

    bq_dataset = bq_client.create_dataset(bq_dataset, exists_ok=True, timeout=30)
    print(
        f"Created BigQuery dataset {bq_client.project}.{bq_dataset.dataset_id} for batch prediction output."
    )

print(f"BigQuery output URI: {BQ_OUTPUT_URI}")
```

### Send a batch prediction request


You create a batch prediction job using the `BatchPredictionJob.submit()` method. To make a batch prediction request, you specify a source model ID, an input source and an output location, either Cloud Storage or BigQuery, where Vertex AI stores the batch prediction results.

To learn more, see the [Batch prediction API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api) page.



```
job = BatchPredictionJob.submit(
    source_model=MODEL_ID, input_dataset=INPUT_DATA, output_uri_prefix=BQ_OUTPUT_URI
)
```

Print out the job status and other properties.


```
print(f"Job resource name: {job.resource_name}")
print(f"Model resource name: {job.model_name}")
print(f"Job state: {job.state.name}")
```

### Wait for the batch prediction job to complete

Depending on the number of input items that you submitted, a batch generation task can take some time to complete. You can use the following code to check the job status and wait for the job to complete.


```
# Refresh the job until complete
while not job.has_ended:
    time.sleep(5)
    job.refresh()

# Check if the job succeeds
if job.has_succeeded:
    print("Job succeeded!")
else:
    print(f"Job failed: {job.error}")
```

### Retrieve batch prediction results

When a batch prediction task is complete, the output of the prediction is stored in the Cloud Storage bucket or BigQuery location that you specified in your request.

- When you are using BigQuery, the output of batch prediction is stored in an output dataset. If you had provided a dataset, the name of the dataset (`BQ_OUTPUT_URI`) is the name you had provided earlier. 
- If you did not provide an output dataset, a default dataset `bq://PROJECT_ID.gen_ai_batch_prediction` will be created for you. The name of the table is formed by appending `predictions_` with the timestamp of when the batch prediction job started.

You can print out the exact output location in the `job.output_location` property.


```
print(f"Job output location: {job.output_location}")
```

You can use the example code below to retrieve predictions and store them into a dataframe.



```
bq_table_id = job.output_location.replace("bq://", "")

sql = f"""
        SELECT *
        FROM {bq_table_id}
        """

query_result = bq_client.query(sql)

df = query_result.result().to_dataframe()
df.head()
```

## Cleaning up

Clean up resources created in this notebook.


```
# Delete the batch prediction job
job.delete()
```
