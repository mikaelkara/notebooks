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

 # Getting Started with Vertex AI Python SDK for Gen AI Evaluation Service 

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/intro_to_gen_ai_evaluation_service_sdk.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fintro_to_gen_ai_evaluation_service_sdk.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/intro_to_gen_ai_evaluation_service_sdk.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/intro_to_gen_ai_evaluation_service_sdk.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai) |

## Overview

In this tutorial, you will learn how to use the *Vertex AI Python SDK for Gen AI Evaluation Service*.


### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.


## Getting Started

### Install Vertex AI Python SDK for Gen AI Evaluation Service


```
%pip install --upgrade --user --quiet google-cloud-aiplatform[evaluation]
```

### Restart runtime
To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
EXPERIMENT_NAME = "my-eval-task-experiment"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
# General
import pandas as pd

# Main
from vertexai.evaluation import EvalTask, PointwiseMetric, PointwiseMetricPromptTemplate
```

### Helper functions


```
from IPython.display import Markdown, display


def display_eval_result(eval_result, metrics=None):
    """Display the evaluation results."""
    summary_metrics, metrics_table = (
        eval_result.summary_metrics,
        eval_result.metrics_table,
    )

    metrics_df = pd.DataFrame.from_dict(summary_metrics, orient="index").T
    if metrics:
        metrics_df = metrics_df.filter(
            [
                metric
                for metric in metrics_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
        metrics_table = metrics_table.filter(
            [
                metric
                for metric in metrics_table.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    # Display the summary metrics
    display(Markdown("### Summary Metrics"))
    display(metrics_df)
    # Display the metrics table
    display(Markdown("### Row-based Metrics"))
    display(metrics_table)
```

# Set up eval metrics based on your criteria

As a developer, you would like to evaluate the text quality generated from an LLM based on two criteria: Fluency and Entertaining. You define a metric called `text_quality` using those two criteria.


```
# Your own definition of text_quality.
metric_prompt_template = PointwiseMetricPromptTemplate(
    criteria={
        "fluency": "Sentences flow smoothly and are easy to read, avoiding awkward phrasing or run-on sentences. Ideas and sentences connect logically, using transitions effectively where needed.",
        "entertaining": "Short, amusing text that incorporates emojis, exclamations and questions to convey quick and spontaneous communication and diversion.",
    },
    rating_rubric={
        "1": "The response performs well on both criteria.",
        "0": "The response is somewhat aligned with both criteria",
        "-1": "The response falls short on both criteria",
    },
)

text_quality = PointwiseMetric(
    metric="text_quality",
    metric_prompt_template=metric_prompt_template,
)
```


```
print(text_quality.metric_prompt_template)
```

# Prepare your dataset

Evaluate stored generative AI model responses in an evaluation dataset.


```
responses = [
    # An example of good text_quality
    "Life is a rollercoaster, full of ups and downs, but it's the thrill that keeps us coming back for more!",
    # An example of medium text_quality
    "The weather is nice today, not too hot, not too cold.",
    # An example of poor text_quality
    "The weather is, you know, whatever.",
]

eval_dataset = pd.DataFrame(
    {
        "response": responses,
    }
)
```

# Run evaluation

With the evaluation dataset and metrics defined, you can run evaluation for an `EvalTask` on different models and applications, prompt templates, and many other use cases.


```
eval_task = EvalTask(
    dataset=eval_dataset, metrics=[text_quality], experiment=EXPERIMENT_NAME
)

eval_result = eval_task.evaluate()
```

You can view the summary metrics and row-based metrics for each response in the `EvalResult`.



```
display_eval_result(eval_result)
```

# Clean up

Delete ExperimentRun created by the evaluation.


```
from google.cloud import aiplatform

aiplatform.ExperimentRun(
    run_name=eval_result.metadata["experiment_run"],
    experiment=eval_result.metadata["experiment"],
).delete()
```
