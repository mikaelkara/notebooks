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

# Evaluate LLMs with AutoSxS Model Eval

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/legacy/evaluate_gemini_with_autosxs.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Flegacy%2Fevaluate_gemini_with_autosxs.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/legacy/evaluate_gemini_with_autosxs.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/legacy/evaluate_gemini_with_autosxs.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
</table>

| | |
|-|-|
|Author(s) | [Ivan Nardini](https://github.com/inardini) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.9

## Overview

Vertex AI Model Evaluation AutoSxS is an LLM evaluation tool, which enables users to compare the performance of Google-first-party and Third-party LLMs.

As part of preview release, AutoSxS only support comparing models for `summarization` and `question answering` tasks according to some predefined criteria. Check out the [documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/models/side-by-side-eval#autosxs) to know more.


This tutorial demonstrates how to use Vertex AI Model Evaluation AutoSxS for comparing models and check human alignment in a summarization task.

### Objective

In this tutorial, you learn how to use Vertex AI Model Evaluation AutoSxS to compare two LLMs predictions (one of the models is Gemini 1.0 Pro) in a summarization task.

This tutorial uses the following Google Cloud ML services and resources:

- Vertex AI Model Evaluation

The steps performed include:

- Read evaluation data
- Define the AutoSxS model evaluation pipeline
- Run the evaluation pipeline job
- Check the judgments, aggregated metrics and human alignment

### Dataset

The dataset is a modified sample of the [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) dataset for evaluation of abstractive single-document summarization systems.

### Costs

This tutorial uses billable components of Google Cloud:

* Vertex AI
* Cloud Storage

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and [Cloud Storage pricing](https://cloud.google.com/storage/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Installation

Install the following packages required to execute this notebook.


```
%pip install --user --upgrade --quiet google-cloud-aiplatform
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

## Before you begin

### Set up your Google Cloud project

**The following steps are required, regardless of your notebook environment.**

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.

2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

3. [Enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

4. If you are running this notebook locally, you need to install the [Cloud SDK](https://cloud.google.com/sdk).

### Authenticate your Google Cloud account

Depending on your Jupyter environment, you may have to manually authenticate. Follow the relevant instructions below.

**1. Vertex AI Workbench**
* Do nothing as you are already authenticated.

**2. Local JupyterLab instance, uncomment and run:**


```
# ! gcloud auth login
```

**3. Colab, uncomment and run:**


```
# from google.colab import auth
# auth.authenticate_user()
```

**4. Service account or other**
* See how to grant Cloud Storage permissions to your service account at https://cloud.google.com/storage/docs/gsutil/commands/iam#ch-examples.

#### Set your project ID

**If you don't know your project ID**, try the following:
* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113)


```
PROJECT_ID = "your-project-id"  # @param {type:"string"}

# Set the project id
! gcloud config set project {PROJECT_ID}
```

#### Region

You can also change the `REGION` variable used by Vertex AI. Learn more about [Vertex AI regions](https://cloud.google.com/vertex-ai/docs/general/locations).


```
REGION = "us-central1"  # @param {type: "string"}
```

### UUID

We define a UUID generation function to avoid resource name collisions on resources created within the notebook.


```
import random
import string


def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


UUID = generate_uuid()
```

### Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.


```
BUCKET_URI = f"gs://autosxs-demo-{UUID}"  # @param {type:"string"}
```

**Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.


```
! gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}
```

### Import libraries


```
import pprint

import pandas as pd
from google.cloud import aiplatform
from google.protobuf.json_format import MessageToDict
from IPython.display import HTML, display
```

### Define constant


```
EVALUATION_FILE_URI = (
    "gs://github-repo/evaluate-gemini/sum_eval_gemini_dataset_001.jsonl"
)
HUMAN_EVALUATION_FILE_URI = (
    "gs://github-repo/evaluate-gemini/sum_human_eval_gemini_dataset_001.jsonl"
)
TEMPLATE_URI = (
    "https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/autosxs-template/2.17.0"
)
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline"
```

### Helpers


```
def print_autosxs_judgments(df, n=3):
    """Print AutoSxS judgments in the notebook"""

    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"
    df = df.sample(n=n)

    for index, row in df.iterrows():
        if row["confidence"] >= 0.5:
            display(HTML(f"<h2>ID:</h2> <div style='{style}'>{row['id']}</div>"))
            display(
                HTML(f"<h2>Document:</h2> <div style='{style}'>{row['document']}</div>")
            )
            display(
                HTML(
                    f"<h2>Response A:</h2> <div style='{style}'>{row['response_a']}</div>"
                )
            )
            display(
                HTML(
                    f"<h2>Response B:</h2> <div style='{style}'>{row['response_b']}</div>"
                )
            )
            display(
                HTML(
                    f"<h2>Explanation:</h2> <div style='{style}'>{row['explanation']}</div>"
                )
            )
            display(
                HTML(
                    f"<h2>Confidence score:</h2> <div style='{style}'>{row['confidence']}</div>"
                )
            )
            display(HTML("<hr>"))


def print_aggregated_metrics(scores):
    """Print AutoSxS aggregated metrics"""

    score_b = round(win_rate_metrics["autosxs_model_b_win_rate"] * 100)
    display(
        HTML(
            f"<h3>AutoSxS Autorater prefers {score_b}% of time Model B over Model A </h3>"
        )
    )


def print_human_preference_metrics(metrics):
    """Print AutoSxS Human-preference alignment metrics"""
    display(
        HTML(
            f"<h3>AutoSxS Autorater prefers {score_b}% of time Model B over Model A </h3>"
        )
    )
```

### Initialize Vertex AI SDK for Python

Initialize the Vertex AI SDK for Python for your project.


```
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
```

## Evaluate LLMs using Vertex AI Model Evaluation AutoSxS

Suppose you've obtained your LLM-generated predictions in a summarization task. To evaluate LLMs such as Gemini 1.0 Pro on Vertex AI against another model using AutoSXS, you need to follow these steps for evaluation:

1.   **Prepare the Evaluation Dataset**: Gather your prompts, contexts, generated responses and human preference required for the evaluation.

2.   **Convert the Evaluation Dataset:** Convert the dataset either into the JSONL format and store it in a Cloud Storage bucket. Alternatively, you can save the dataset to a BigQuery table.

3.   **Run a Model Evaluation Job:** Use Vertex AI to run a model evaluation job to assess the performance of the LLM.

### Read the evaluation data

In this summarization use case, you use `sum_eval_gemini_dataset_001`, a JSONL-formatted evaluation datasets which contains content-response pairs without human preferences.

In the dataset, each row represents a single example. The dataset includes ID fields, such as "id" and "document," which are used to identify each unique example. The "document" field contains the newspaper articles to be summarized.

While the dataset does not have [data fields](https://cloud.google.com/vertex-ai/docs/generative-ai/models/side-by-side-eval#prep-eval-dataset) for prompts and contexts, it does include pre-generated predictions. These predictions contain the generated response according to the LLMs task, with "response_a" and "response_b" representing different article summaries.

**Note: For experimentation, you can provide only a few examples. The documentation recommends at least 400 examples to ensure high-quality aggregate metrics.**



```
evaluation_gemini_df = pd.read_json(EVALUATION_FILE_URI, lines=True)
evaluation_gemini_df.head()
```

### Run a model evaluation job

AutoSxS relays on Vertex AI pipelines to run model evaluation. And here you can see some of the required pipeline parameters:

*   `evaluation_dataset` to indicate where the evaluation dataset location. In this case, it is the JSONL Cloud Storage URI.

*   `id_columns` to distinguish evaluation examples that are unique. Here, as you can imagine, your have `id` and `document` fields.

*   `task` to indicate the task type you want to evaluate in `{task}@{version}` form. It can be `summarization` or `question_answer`. In this case you have `summarization`.

*   `autorater_prompt_parameters` to configure the autorater task behaviour. And you can specify inference instructions to guide task completion, as well as setting the inference context to refer during the task execution.

Lastly, you have to provide `response_column_a` and `response_column_b` with the names of columns containing predefined predictions in order to calculate the evaluation metrics. In this case, `response_a` and `response_b` respectively.

To learn more about all supported parameters, see the [official documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/models/side-by-side-eval#perform-eval).



```
display_name = f"autosxs-eval-{generate_uuid()}"
parameters = {
    "evaluation_dataset": EVALUATION_FILE_URI,
    "id_columns": ["id", "document"],
    "task": "summarization",
    "autorater_prompt_parameters": {
        "inference_context": {"column": "document"},
        "inference_instruction": {"template": "Summarize the following text: "},
    },
    "response_column_a": "response_a",
    "response_column_b": "response_b",
}
```

After you define the model evaluation parameters, you can run a model evaluation pipeline job using the predefined pipeline template with the Vertex AI Python SDK.


```
job = aiplatform.PipelineJob(
    job_id=display_name,
    display_name=display_name,
    pipeline_root=f"{BUCKET_URI}/{display_name}",
    template_path=TEMPLATE_URI,
    parameter_values=parameters,
    enable_caching=False,
)
job.run(sync=True)
```

### Evaluate the results

After the evaluation pipeline successfully runs, you can review the evaluation results by looking both at artifacts generated by the pipeline itself in Vertex AI Pipelines UI and in the notebook enviroment using the Vertex AI Python SDK.

AutoSXS produces three types of evaluation results: a judgments table, aggregated metrics, and alignment metrics (if human preferences are provided).


### AutoSxS Judgments

The judgments table contains metrics that offer insights on LLM performance for each example.

For each response pair, the judgments table includes a `choice` column indicating the better response based on the evaluation criteria used by the autorater.

Each choice has a `confidence score` column between 0 and 1, representing the autorater's level of confidence in the evaluation.

Last but not less important, AutoSXS provides an explanation for why the autorater preferred one response over the other.

Below you have an example of AutoSxS judgments output.


```
# To use an existing pipeline, override job using the line below.
# job = aiplatform.PipelineJob.get('projects/[PROJECT_NUMBER]/locations/[REGION]/pipelineJobs/[PIPELINE_RUN_NAME]')

for details in job.task_details:
    if details.task_name == "online-evaluation-pairwise":
        break

judgments_uri = MessageToDict(details.outputs["judgments"]._pb)["artifacts"][0]["uri"]
judgments_df = pd.read_json(judgments_uri, lines=True)
```


```
print_autosxs_judgments(judgments_df)
```

### AutoSxS Aggregate metrics

AutoSxS also provides aggregated metrics as an additional evaluation result. These win-rate metrics are calculated by utilizing the judgments table to determine the percentage of times the autorater preferred each model's response.  

These metrics are relevant to quickly find out which is the better model in the context of the evaluated task.

Below you have an example of AutoSxS Aggregate metrics.


```
for details in job.task_details:
    if details.task_name == "model-evaluation-text-generation-pairwise":
        break

win_rate_metrics = MessageToDict(details.outputs["autosxs_metrics"]._pb)["artifacts"][
    0
]["metadata"]
print_aggregated_metrics(win_rate_metrics)
```

### Human-preference alignment metrics

After reviewing the results of your initial AutoSxS evaluation, you may wonder about the alignment of the Autorater's assessment with human raters' views.

AutoSxS supports human preference to validate Autorater evaluation.

To check alignment with a human-preference dataset, you need to add the ground truths as a column to the `evaluation_dataset` and pass the column name to `human_preference_column`.

#### Read the evaluation data

With respect to the evaluation dataset, in this case the `sum_human_eval_gemini_dataset_001` dataset also includes human preferences.

Below is a sample of the dataset.


```
human_evaluation_gemini_df = pd.read_json(HUMAN_EVALUATION_FILE_URI, lines=True)
human_evaluation_gemini_df.head()
```

#### Run a model evaluation job

With respect to the AutoSXS pipeline, you must specify the human preference column in the pipeline parameters.

Then, you can run the evaluation pipeline job using the Vertex AI Python SDK as shown below.



```
display_name = f"autosxs-human-eval-{generate_uuid()}"
parameters = {
    "evaluation_dataset": HUMAN_EVALUATION_FILE_URI,
    "id_columns": ["id", "document"],
    "task": "summarization",
    "autorater_prompt_parameters": {
        "inference_context": {"column": "document"},
        "inference_instruction": {"template": "Summarize the following text: "},
    },
    "response_column_a": "response_a",
    "response_column_b": "response_b",
    "human_preference_column": "actual",
}
```


```
job = aiplatform.PipelineJob(
    job_id=display_name,
    display_name=display_name,
    pipeline_root=f"{BUCKET_URI}/{display_name}",
    template_path=TEMPLATE_URI,
    parameter_values=parameters,
    enable_caching=False,
)
job.run(sync=True)
```

### Get human-aligned aggregated metrics

Compared with the aggregated metrics from before, now the pipeline returns additional measurements that utilize the provided human-preference data provided.

In addition to well-known metrics like accuracy, precision, and recall, you will receive both the human preference scores and autorater preference scores. These scores indicate the level of agreement between the evaluations. And to simplify this comparison, Cohen's Kappa is provided. Cohen's Kappa measures the level of agreement between the autorater and human raters. It ranges from 0 to 1, where 0 represents agreement equivalent to a random choice and 1 indicates perfect agreement.

Below is a view of the resulting human-aligned aggregated metrics.



```
for details in job.task_details:
    if details.task_name == "model-evaluation-text-generation-pairwise":
        break

human_aligned_metrics = {
    k: round(v, 3)
    for k, v in MessageToDict(details.outputs["autosxs_metrics"]._pb)["artifacts"][0][
        "metadata"
    ].items()
}
pprint.pprint(human_aligned_metrics)
```

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial.


```
import os

# Delete Model Evaluation pipeline run
delete_pipeline = True
if delete_pipeline or os.getenv("IS_TESTING"):
    job.delete()

# Delete Cloud Storage objects that were created
delete_bucket = True
if delete_bucket or os.getenv("IS_TESTING"):
    ! gsutil -m rm -r $BUCKET_URI
```
