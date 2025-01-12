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

# Intro to Batch Predictions with the Gemini API


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fbatch-prediction%2Fintro_batch_prediction.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/batch-prediction/intro_batch_prediction.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
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

In this tutorial, you learn how to make batch predictions with the Gemini API in Vertex AI. This tutorial uses Cloud Storage as an input source and an output location.

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

## Code Examples

### Import libraries


```
from datetime import datetime
import json
import time

from google.cloud import storage
from vertexai.batch_prediction import BatchPredictionJob
from vertexai.generative_models import GenerativeModel
```

### Load model

You can find a list of the Gemini models that support batch predictions in the [Multimodal models that support batch predictions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#multimodal_models_that_support_batch_predictions) page.

This tutorial uses the Gemini 1.5 Flash (`gemini-1.5-flash-002`) model.


```
MODEL_ID = "gemini-1.5-flash-002"  # @param {type:"string", isTemplate: true}

model = GenerativeModel(MODEL_ID)
```

### Prepare batch inputs

The input for batch requests specifies the items to send to your model for prediction.

Batch requests for Gemini accept BigQuery storage sources and Cloud Storage sources. You can learn more about the batch input formats for BigQuery and Cloud Storage sources in the [Batch text generation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#prepare_your_inputs) page.

This tutorial uses Cloud Storage as an example. The requirements for Cloud Storage input are:

- File format: [JSON Lines (JSONL)](https://jsonlines.org/)
- Located in `us-central1`
- Appropriate read permissions for the service account

Each request that you send to a model can include parameters that control how the model generates a response. Learn more about Gemini parameters in the [Experiment with parameter values](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values) page.

This is one of the example requests in the input JSONL file `batch_requests_for_multimodal_input_2.jsonl`:
```
{"request":{"contents": [{"role": "user", "parts": [{"text": "List objects in this image."}, {"file_data": {"file_uri": "gs://cloud-samples-data/generative-ai/image/office-desk.jpeg", "mime_type": "image/jpeg"}}]}],"generationConfig":{"temperature": 0.4}}}
```



```
INPUT_DATA = "gs://cloud-samples-data/generative-ai/batch/batch_requests_for_multimodal_input_2.jsonl"  # @param {type:"string"}
```

### Prepare batch output location

When a batch prediction task completes, the output is stored in the location that you specified in your request.

- The location is in the form of a Cloud Storage or BigQuery URI prefix, for example:
`gs://path/to/output/data` or `bq://projectId.bqDatasetId`.

- If not specified, `STAGING_BUCKET/gen-ai-batch-prediction` will be used for Cloud Storage source and `bq://PROJECT_ID.gen_ai_batch_prediction.predictions_TIMESTAMP` will be used for BigQuery source.

This tutorial uses a Cloud Storage bucket as an example for the output location.
- You can specify the URI of your Cloud Storage bucket in `BUCKET_URI`, or
- if it is not specified, a new Cloud Storage bucket in the form of `gs://PROJECT_ID-TIMESTAMP` will be created for you.



```
BUCKET_URI = "[your-cloud-storage-bucket]"  # @param {type:"string"}

if BUCKET_URI == "[your-cloud-storage-bucket]":
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    BUCKET_URI = f"gs://{PROJECT_ID}-{TIMESTAMP}"

    ! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}
```

### Send a batch prediction request


You create a batch prediction job using the `BatchPredictionJob.submit()` method. To make a batch prediction request, you specify a source model ID, an input source and an output location, either Cloud Storage or BigQuery, where Vertex AI stores the batch prediction results.

To learn more, see the [Batch prediction API](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api) page.



```
job = BatchPredictionJob.submit(
    source_model=MODEL_ID, input_dataset=INPUT_DATA, output_uri_prefix=BUCKET_URI
)
```

Print out the job status and other properties.


```
print(f"Job resource name: {job.resource_name}")
print(f"Model resource name: {job.model_name}")
print(f"Job state: {job.state.name}")
```

Optionally, you can use `BatchPredictionJob.list()` to list all the batch prediction jobs in the project.


```
for batch_job in BatchPredictionJob.list():
    print(
        f"Job ID: '{batch_job.name}', Job state: {batch_job.state.name}, Job model: {batch_job.model_name}"
    )
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


You can print out the exact output location in the `job.output_location` property.


```
print(f"Job output location: {job.output_location}")
```

If you are using BigQuery, the output of batch prediction is stored in an output dataset.

This tutorial uses a Cloud Storage bucket as an example. For Cloud Storage,

- The output folder contains a set of JSONL files.
- The files are named `{gcs_path}/prediction-model-{timestamp}`
- Each line in the file corresponds to an instance.

Example output:
```
{'status': '', 'processed_time': '2024-11-13T14:04:28.376+00:00', 'request': {'contents': [{'parts': [{'file_data': None, 'text': 'List objects in this image.'}, {'file_data': {'file_uri': 'gs://cloud-samples-data/generative-ai/image/gardening-tools.jpeg', 'mime_type': 'image/jpeg'}, 'text': None}], 'role': 'user'}], 'generationConfig': {'temperature': 0.4}}, 'response': {'candidates': [{'avgLogprobs': -0.10394711927934126, 'content': {'parts': [{'text': "Here's a list of the objects in the image:\n\n* **Watering can:** A green plastic watering can with a white rose head.\n* **Plant:** A small plant (possibly oregano) in a terracotta pot.\n* **Terracotta pots:** Two terracotta pots, one containing the plant and another empty, stacked on top of each other.\n* **Gardening gloves:** A pair of striped gardening gloves.\n* **Gardening tools:** A small trowel and a hand cultivator (hoe).  Both are green with black handles."}], 'role': 'model'}, 'finishReason': 'STOP'}], 'modelVersion': 'gemini-1.5-flash-002@default', 'usageMetadata': {'candidatesTokenCount': 110, 'promptTokenCount': 264, 'totalTokenCount': 374}}}
```

The example code below shows how to read all the `.jsonl` files in the Cloud Storage output location and print each JSON object.


```
# Get the bucket name
bucket_name = job.output_location.split("/")[2]

# Get the path after the bucket name
directory_path = "/".join(job.output_location.split("/")[3:])

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)

blobs = bucket.list_blobs(prefix=directory_path)

for blob in blobs:
    if blob.name.endswith(".jsonl"):
        jsonl_string = blob.download_as_string().decode("utf-8")
        for line in jsonl_string.splitlines():
            try:
                json_object = json.loads(line)
                print(json_object)
            except json.JSONDecodeError as e:
                print(e)
```

## Cleaning up

Clean up resources created in this notebook.


```
# Delete the batch prediction job
job.delete()
```
