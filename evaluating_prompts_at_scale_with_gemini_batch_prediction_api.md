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

# Evaluating prompts at scale with Gemini Batch Prediction API

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluating_prompts_at_scale_with_gemini_batch_prediction_api.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fevaluating_prompts_at_scale_with_gemini_batch_prediction_api.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/evaluating_prompts_at_scale_with_gemini_batch_prediction_api.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluating_prompts_at_scale_with_gemini_batch_prediction_api.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluating_prompts_at_scale_with_gemini_batch_prediction_api.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Ariel Jassan](https://github.com/arieljassan) |

## Introduction

This tutorial guides you through the process of evaluating the effectiveness of your prompts at scale using the Gemini Batch Prediction API via Vertex AI. Even though in this tutorial we will do image classification, it can be extended to other cases as well. One of the benefits of using the Gemini Batch Prediction API is that you can evaluate your prompts and setup in Gemini using hundreds of examples with one single request.

You can find more information about the Gemini Batch Prediction API [here](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api).

For the purpose of this tutorial, we will execute a prompt to classify images into classes of sports. The data is based on an excerpt of the dataset that can be found in https://www.kaggle.com/datasets/gpiosenka/sports-classification.


## Steps

1. **Prepare the data in BigQuery and GCS**
    * Upload sample images to Google Cloud Storage and create ground truth table in BigQuery.
    
2. **Run Gemini Batch Prediction API**
    * Send prompts to Gemini for batch prediction and get results in BigQuery.

3. **Analyze results in BigQuery and Looker Studio**
    * Present findings, focusing on prompt/dataset strengths and weaknesses.

## Getting started

### Install dependencies


```
%pip install --upgrade -q google-cloud-aiplatform google-cloud-bigquery bigframes pandas pandas-gbq
```

### Restart Colab


```
# You will see a notification of Colab crashing. It is the expected behavior.
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Authenticate your notebook environment (Colab only)


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Define constants


```
PROJECT_ID = "your-project-id"
LOCATION = "us-central1"

# Generative model.
MODEL_ID = "gemini-1.5-flash-001"

# BigQuery tables.
BQ_DATASET_ID = "gemini_batch_predictions"
BQ_DATASET = f"{PROJECT_ID}.{BQ_DATASET_ID}"
FILES_TABLE = f"{BQ_DATASET_ID}.sports_files"
PROMPTS_TABLE = f"{BQ_DATASET}.temp_prompts"
TEXT_GENERATION_TABLE_PREFIX = f"{BQ_DATASET}.results"

# BigQuery views.
RESULTS_VIEW = f"{BQ_DATASET}.extraction_results"
EVALUATION_VIEW = f"{BQ_DATASET}.evaluation"

# File containing ground truth data in GCS.
BUCKET_NAME = "github-repo"
FOLDER = "generative-ai/gemini/evaluation/sports_files"
GCS_PREFIX = f"gs://{BUCKET_NAME}/{FOLDER}"
SPORTS_FILE = "sports_files.csv"
```

### Import libraries and initialize clients


```
import datetime
import json
import time

import bigframes.pandas as bpd
from google.cloud import bigquery, storage
import pandas as pd
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
from vertexai.preview.batch_prediction import BatchPredictionJob
```


```
# BigQuery client.
bq_client = bigquery.Client(project=PROJECT_ID)

# Google Cloud Storage client.
storage_client = storage.Client()

# Initialize Vertex AI SDK.
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Set BigQuery Pandas options.
bpd.options.bigquery.project = PROJECT_ID
bpd.options.bigquery.location = LOCATION
```

## Data preparation

In this section we will create the dataset in BigQuery, load the table with ground truth, and create the views that will serve for analysis of the results from Gemini and reporting in Looker Studio.

### Create BigQuery dataset and load table with ground truth


```
def create_dataset(dataset_id: str, location: str) -> None:
    """Creates a BigQuery dataset in a location."""
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = location

    dataset = bq_client.create_dataset(dataset, timeout=30)
    print(f"Created dataset {bq_client.project}.{dataset.dataset_id}")


def load_files_table_from_uri(files_table: str, uri: str) -> None:
    """Load ground truth into a BigQuery table from a GCS URI."""
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("path", "STRING"),
            bigquery.SchemaField("label", "STRING"),
        ],
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    load_job = bq_client.load_table_from_uri(uri, files_table, job_config=job_config)
    load_job.result()

    destination_table = bq_client.get_table(files_table)
    print(f"Loaded {destination_table.num_rows} rows.")


create_dataset(dataset_id=BQ_DATASET, location=LOCATION)
load_files_table_from_uri(files_table=FILES_TABLE, uri=f"{GCS_PREFIX}/{SPORTS_FILE}")
```

### Test image URIs are retrieved from BigQuery


```
ground_truth_df = bpd.read_gbq(FILES_TABLE)
ground_truth_df["path"][:2]
```

## Define prompt and execute it via Vertex AI Gemini Batch Prediction API

### Define the prompt


```
prompt = """\
- Classify the sport from the image below in one of the following categories:
* baseball
* basketball
* tennis
* volleyball

- Provide an answer in JSON format.

Example response:
{"sport": "baseball"}

- Image:
"""
```

### Classify one image using the Python SDK


```
def classify_image(model_id: str, prompt: str, gcs_prefix: str, blob_name: str) -> str:
    """Classifies an image."""
    model = GenerativeModel(
        model_id,
        generation_config=GenerationConfig(response_mime_type="application/json"),
    )
    image_content = Part.from_uri(
        uri=f"{gcs_prefix}/{blob_name}", mime_type="image/jpeg"
    )
    contents = [prompt, image_content]
    return model.generate_content(contents).text


blob_name = ground_truth_df.iloc[0]["path"]
response = classify_image(
    model_id=MODEL_ID,
    prompt=prompt,
    gcs_prefix=GCS_PREFIX,
    blob_name=blob_name,
)
print(f"blob_name: {blob_name}")
print(f"response: {response}")
```

### Create a BigQuery table applying the prompt to each of the images

In this section, an `evaluation_id` variable is created to identify the execution run.


```
# Use current time as identifier of the evaluation.
now = datetime.datetime.now()
evaluation_ts = str(now)
evaluation_id = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}"
json_file_name = f"/tmp/{evaluation_id}.json"

# Get URIs of the images from the ground truth table in BigQuery.
ground_truth_df = bpd.read_gbq(FILES_TABLE)

prompts_df = pd.DataFrame(
    [
        {
            "evaluation_ts": evaluation_ts,
            "evaluation_id": evaluation_id,
            "prompt_text": prompt,
            "gcs_uri": image_uri,
            "request": json.dumps(
                {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {
                                    "fileData": {
                                        "mimeType": "image/jpeg",
                                        "fileUri": f"{GCS_PREFIX}/{image_uri}",
                                    }
                                },
                            ],
                        }
                    ],
                    "generationConfig": {"responseMimeType": "application/json"},
                }
            ),
        }
        for image_uri in ground_truth_df["path"].values
    ]
)

# Save JSONL file
prompts_df.to_json(json_file_name, lines=True)

# Upload to BQ
prompts_df.to_gbq(PROMPTS_TABLE, PROJECT_ID)

table = bq_client.get_table(PROMPTS_TABLE)
print(
    f"Loaded {table.num_rows} rows and {len(table.schema)} columns to "
    f"{PROMPTS_TABLE}"
)
```

### Launch a Gemini Batch Prediction request


```
# Define table to store results from Gemini Batch Prediction.
text_generation_table = f"{TEXT_GENERATION_TABLE_PREFIX}_{evaluation_id}"

# Create batch prediction job.
batch_job = BatchPredictionJob.submit(
    source_model=MODEL_ID,
    input_dataset=f"bq://{PROMPTS_TABLE}",
    output_uri_prefix=f"bq://{text_generation_table}",
)
```

To check the status of the job, run this cell.


```
# Refresh the job until complete
while not batch_job.has_ended:
    time.sleep(10)
    batch_job.refresh()

# Check if the job succeeds
if batch_job.has_succeeded:
    print("Job succeeded!")
else:
    print(f"Job failed: {batch_job.error}")

# Check the location of the output
print(f"Job output location: {batch_job.output_location}")
```

### List sample of text generation results from BigQuery


```
text_generation_df = bpd.read_gbq(text_generation_table)
for row in text_generation_df["response"][:5]:
    print(json.loads(row)["candidates"][0]["content"]["parts"][0]["text"])
```

## Create Views in BigQuery

### Create view of text generation results

Run this only once to create the view


```
def create_text_generation_view(
    text_generation_table_prefix: str, results_view: str
) -> None:
    """Creates a view of text extraction results."""

    view = bigquery.Table(results_view)

    view.view_query = rf"""
    SELECT
        evaluation_id,
        evaluation_ts,
        prompt_text,
        gcs_uri,
        JSON_VALUE(JSON_VALUE(response, '$.candidates[0].content.parts[0].text'), "$.sport") AS label
    FROM `{text_generation_table_prefix}_*`
    """

    # Make an API request to create the view.
    view = bq_client.create_table(view, exists_ok=False)
    print(f"Created {view.table_type}: {str(view.reference)}")


create_text_generation_view(
    text_generation_table_prefix=TEXT_GENERATION_TABLE_PREFIX, results_view=RESULTS_VIEW
)
```

### Create view of experiment evaluation

Run this only once to create the view.


```
def create_evaluation_view(
    evaluation_view: str, files_table: str, results_view: str
) -> None:
    """Creates a view of experiment evaluation."""

    view = bigquery.Table(evaluation_view)

    view.view_query = f"""
      WITH t1 AS (
        SELECT
          e.evaluation_id,
          e.evaluation_ts,
          e.prompt_text,
          f.path,
          f.label,
          e.gcs_uri,
          f.label = e.label AS correct
        FROM `{files_table}` f
        JOIN `{results_view}` e
          ON f.path = e.gcs_uri
      )

      SELECT
        evaluation_id,
        evaluation_ts,
        prompt_text,
        path,
        label,
        correct
      FROM t1"""

    # Make an API request to create the view.
    view = bq_client.create_table(view, exists_ok=False)
    print(f"Created {view.table_type}: {str(view.reference)}")


create_evaluation_view(
    evaluation_view=EVALUATION_VIEW, files_table=FILES_TABLE, results_view=RESULTS_VIEW
)
```

## Analyze results in BigQuery and Looker Studio

### Copy a Looker Studio dashboard to analyze results

1. Make a copy of this [Looker Studio dashboard](https://lookerstudio.google.com/reporting/caba1b62-2820-467a-bbe7-bd852d538de8/preview)
1. Connect dashboard to your view
