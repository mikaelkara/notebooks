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

 # Evaluate Generative Model Tool Use | Gen AI Evaluation SDK Tutorial

 <table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_gemini_tool_use.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fevaluate_gemini_tool_use.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/evaluate_gemini_tool_use.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_gemini_tool_use.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai),  [Bo Zheng](https://github.com/coolalexzb) |

## Overview

* Define an API function and a Tool for Gemini model, and evaluate the Gemini model tool use quality with *Vertex AI Python SDK for Gen AI Evaluation Service*.

See also: 

- Learn more about [Vertex Gen AI Evaluation Service SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview).

## Getting Started

### Install Vertex AI Python SDK for Gen AI Evaluation Service


```
%pip install --upgrade --user --quiet google-cloud-aiplatform[evaluation]
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

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
# General
import json
import logging
import random
import string
import warnings

from IPython.display import Markdown, display
import pandas as pd

# Main
from vertexai.evaluation import EvalTask
from vertexai.generative_models import GenerativeModel
```

### Library settings


```
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
```

### Helper functions


```
def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def display_eval_report(eval_result, metrics=None):
    """Display the evaluation results."""

    title, summary_metrics, report_df = eval_result
    metrics_df = pd.DataFrame.from_dict(summary_metrics, orient="index").T
    if metrics:
        metrics_df = metrics_df.filter(
            [
                metric
                for metric in metrics_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
        report_df = report_df.filter(
            [
                metric
                for metric in report_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    # Display the title with Markdown for emphasis
    display(Markdown(f"## {title}"))

    # Display the metrics DataFrame
    display(Markdown("### Summary Metrics"))
    display(metrics_df)

    # Display the detailed report DataFrame
    display(Markdown("### Report Metrics"))
    display(report_df)
```

## Evaluate Tool use and Function Calling quality for Gemini

#### Tool evaluation metrics

* `tool_call_valid`
* `tool_name_match`
* `tool_parameter_key_match`
* `tool_parameter_kv_match`


```
tool_metrics = [
    "tool_call_valid",
    "tool_name_match",
    "tool_parameter_key_match",
    "tool_parameter_kv_match",
]
```

### 1. Evaluate a Bring-Your-Own-Prediction dataset

Generative model's tool use quality can be evaluated if the eval dataset contains saved model tool call responses, and expected references.


```
response = [
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Regal Edwards 14", "location": "Mountain View CA", "showtime": "7:30", "date": "2024-03-30", "num_tix": "2"}}]}',
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Regal Edwards 14", "location": "Mountain View CA", "showtime": "7:30", "date": "2024-03-30", "num_tix": "2"}}]}',
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Regal Edwards 14"}}]}',
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Cinemark", "location": "Mountain View CA", "showtime": "5:30", "date": "2024-03-30", "num_tix": "2"}}]}',
]

reference = [
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Regal Edwards 14", "location": "Mountain View CA", "showtime": "7:30", "date": "2024-03-30", "num_tix": "2"}}]}',
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Godzilla", "theater": "Regal Edwards 14", "location": "Mountain View CA", "showtime": "9:30", "date": "2024-03-30", "num_tix": "2"}}]}',
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Regal Edwards 14", "location": "Mountain View CA", "showtime": "7:30", "date": "2024-03-30", "num_tix": "2"}}]}',
    '{"content": "", "tool_calls": [{"name": "book_tickets", "arguments": {"movie": "Mission Impossible Dead Reckoning Part 1", "theater": "Regal Edwards 14", "location": "Mountain View CA", "showtime": "7:30", "date": "2024-03-30", "num_tix": "2"}}]}',
]

eval_dataset = pd.DataFrame(
    {
        "response": response,
        "reference": reference,
    }
)
```

#### Define EvalTask


```
experiment_name = "eval-saved-llm-tool-use"  # @param {type:"string"}

tool_use_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=tool_metrics,
    experiment=experiment_name,
)
```


```
run_id = generate_uuid()

experiment_run_name = f"eval-{run_id}"

eval_result = tool_use_eval_task.evaluate(experiment_run_name=experiment_run_name)
display_eval_report(
    (
        "Tool Use Quality Evaluation Metrics",
        eval_result.summary_metrics,
        eval_result.metrics_table,
    )
)
```


```
tool_use_eval_task.display_runs()
```

## 2. Tool Use and Function Calling with Gemini

[Function Calling Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)

### Define a function and tool

Define an API specification and register the function in a tool with the latest version of [Vertex AI SDK for Python](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk).



```
from vertexai.generative_models import FunctionDeclaration, Tool

book_tickets_func = FunctionDeclaration(
    name="book_tickets",
    description="Book movie tickets",
    parameters={
        "type": "object",
        "properties": {
            "movie": {"type": "string", "description": "The title of the movie."},
            "theater": {
                "type": "string",
                "description": "The name of the movie theater.",
            },
            "location": {
                "type": "string",
                "description": "The location of the movie theater.",
            },
            "showtime": {
                "type": "string",
                "description": "The showtime of the movie in ISO 8601 format.",
            },
            "date": {
                "type": "string",
                "description": "The date of the movie in ISO 8601 format.",
            },
            "num_tix": {
                "type": "string",
                "description": "The integer number of tickets to book.",
            },
        },
        "required": [
            "movie",
            "theater",
            "location",
            "showtime",
            "date",
            "num_tix",
        ],
    },
)


book_tickets_tool = Tool(
    function_declarations=[book_tickets_func],
)
```

### Generate a function call

Prompt the Gemini model and include the tool that you defined.


```
prompt = """I'd like to book 2 tickets for the movie "Mission Impossible Dead Reckoning Part 1"
at the Regal Edwards 14 theater in Mountain View, CA. The showtime is 7:30 PM on March 30th, 2024.
"""

gemini_model = GenerativeModel("gemini-1.5-pro")

gemini_response = gemini_model.generate_content(
    prompt,
    tools=[book_tickets_tool],
)

gemini_response.candidates[0].content
```

###  Unpack the Gemini response into a Python dictionary


```
def unpack_response(response):
    output = {}
    function_call = {}
    for key, value in response.candidates[0].content.parts[0].to_dict().items():
        function_call[key] = value
    output["content"] = ""
    output["tool_calls"] = [function_call["function_call"]]
    output["tool_calls"][0]["arguments"] = output["tool_calls"][0].pop("args")
    return json.dumps(output)


response = unpack_response(gemini_response)
response
```

### Evaluate the Gemini's Function Call Response


```
reference_str = json.dumps(
    {
        "content": "",
        "tool_calls": [
            {
                "name": "book_tickets",
                "arguments": {
                    "movie": "Mission Impossible Dead Reckoning Part 1",
                    "theater": "Regal Edwards 14",
                    "location": "Mountain View CA",
                    "showtime": "7:30",
                    "date": "2024-03-30",
                    "num_tix": "2",
                },
            }
        ],
    }
)

eval_dataset = pd.DataFrame({"response": [response], "reference": [reference_str]})
```


```
# Expected Tool Call Response
json.loads(eval_dataset.reference[0])
```


```
# Actual Gemini Tool Call Response
json.loads(eval_dataset.response[0])
```


```
experiment_name = "eval-gemini-model-function-call"  # @param {type:"string"}

gemini_functiona_call_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=tool_metrics,
    experiment=experiment_name,
)
```


```
run_id = generate_uuid()

eval_result = gemini_functiona_call_eval_task.evaluate(
    experiment_run_name=f"eval-{run_id}"
)

display_eval_report(
    (
        "Gemini Tool Use Quality Evaluation Metrics",
        eval_result.summary_metrics,
        eval_result.metrics_table,
    )
)
```
