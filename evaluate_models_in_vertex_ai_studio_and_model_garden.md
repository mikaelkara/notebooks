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

# Use Gen AI Evaluation SDK to Evaluate Models in Vertex AI Studio, Model Garden, and Model Registry

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_models_in_vertex_ai_studio_and_model_garden.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fevaluate_models_in_vertex_ai_studio_and_model_garden.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_models_in_vertex_ai_studio_and_model_garden.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/evaluate_models_in_vertex_ai_studio_and_model_garden.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai) |

This notebook demonstrates how to get started with using the *Vertex AI Python SDK for Gen AI Evaluation Service* for generative models in Vertex AI Studio, Model Garden, and Model Registry.

Gen AI Evaluation Service empowers you to comprehensively assess and enhance your generative AI models and applications. Whether you're selecting the ideal model, optimizing prompt templates, or evaluating fine-tuned checkpoints, this service provides the tools and insights you need.

In this Colab tutorial, we'll explore three major use cases:

1.  Run Evaluation on 1P Models
  *   Learn how to evaluate `Gemini` models in Vertex AI Studio using the *Gen AI Evaluation Service SDK*.

  *   Explore different evaluation metrics and techniques for assessing performance on various tasks.

  *   Discover how to leverage the SDK for in-depth analysis and comparison of `Gemini` model variants.


2.  Run Evaluation on 3P Models
  *   Learn how to evaluate third-party open models, such as a pretrained `Llama 3.1` model, or a fine-tuned `Llama 3` model deployed in Vertex Model Garden, using the *Gen AI Evaluation Service SDK*.

  *   Learn how to evaluate third-party closed model APIs, such as Anthropic's `Claude 3.5 Sonnet` model hosted on Vertex AI, using the *Gen AI Evaluation Service SDK*.

  *   Gain insights into conducting controlled experiments by maintaining the same `EvalTask` configuration with fixed dataset and evaluation metrics while evaluating various model architectures and capabilities.


3.  Prompt Engineering

  *   Explore the impact of prompt design on model performance.
  *   Utilize the SDK to systematically evaluate and refine your prompts.


For additional use cases and advanced features, refer to our public documentation and notebook tutorials for evaluation use cases:

* https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview#notebooks_for_evaluation_use_cases

* https://cloud.google.com/vertex-ai/generative-ai/docs/models/run-evaluation

Let's get started!

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.10

## Getting Started

### Install Vertex AI SDK for Gen AI Evaluation Service


```
%pip install -U -q google-cloud-aiplatform[evaluation]
```

### Install other required packages


```
%pip install -U -q datasets
%pip install -U -q anthropic[vertex]
%pip install -U -q openai
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

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
from anthropic import AnthropicVertex
from google.auth import default, transport
import openai
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PairwiseMetric,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)
from vertexai.generative_models import GenerativeModel
```

### Library settings


```
# @title

import logging
import warnings

import pandas as pd

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# pd.set_option('display.max_colwidth', None)
```

### Helper functions


```
# @title

import random
import string

from IPython.display import HTML, Markdown, display
import plotly.graph_objects as go


def display_explanations(eval_result, metrics=None, n=1):
    """Display the explanations."""
    style = "white-space: pre-wrap; width: 1500px; overflow-x: auto;"
    metrics_table = eval_result.metrics_table
    df = metrics_table.sample(n=n)

    if metrics:
        df = df.filter(
            ["response", "baseline_model_response"]
            + [
                metric
                for metric in df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
    for index, row in df.iterrows():
        for col in df.columns:
            display(HTML(f"<div style='{style}'><h4>{col}:</h4>{row[col]}</div>"))
        display(HTML("<hr>"))


def display_eval_result(eval_result, title=None, metrics=None):
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

    if title:
        # Display the title with Markdown for emphasis
        display(Markdown(f"## {title}"))
    # Display the summary metrics DataFrame
    display(Markdown("### Summary Metrics"))
    display(metrics_df)
    # Display the metrics table DataFrame
    display(Markdown("### Row-based Metrics"))
    display(metrics_table)


def display_radar_plot(eval_results, metrics=None):
    """Plot the radar plot."""
    fig = go.Figure()
    for item in eval_results:
        title, eval_result = item
        summary_metrics = eval_result.summary_metrics
        if metrics:
            summary_metrics = {
                k.replace("/mean", ""): summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric + "/mean" in k for selected_metric in metrics)
            }
        fig.add_trace(
            go.Scatterpolar(
                r=list(summary_metrics.values()),
                theta=list(summary_metrics.keys()),
                fill="toself",
                name=title,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True
    )
    fig.show()


def display_bar_plot(eval_results_list, metrics=None):
    """Plot the bar plot."""
    fig = go.Figure()
    data = []

    for eval_results in eval_results_list:
        title, eval_result = eval_results[0], eval_results[1]

        summary_metrics = eval_result.summary_metrics
        mean_summary_metrics = [f"{metric}/mean" for metric in metrics]
        updated_summary_metrics = []
        if metrics:
            for k, v in summary_metrics.items():
                if k in mean_summary_metrics:
                    updated_summary_metrics.append((k, v))
            summary_metrics = dict(updated_summary_metrics)
            # summary_metrics = {k: summary_metrics[k] for k, v in summary_metrics.items() if any(selected_metric in k for selected_metric in metrics)}

        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group", showlegend=True)
    fig.show()


def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))
```

## Load an evaluation dataset

Load a subset of the `OpenOrca` dataset using the `huggingface/datasets` library. We will use 10 samples from the first 100 rows of the "train" split of `OpenOrca` dataset to demonstrate evaluating prompts and model responses in this Colab.

### Dataset Summary

The OpenOrca dataset is a collection of augmented [FLAN Collection data](https://arxiv.org/abs/2301.13688). Currently ~1M GPT-4 completions, and ~3.2M GPT-3.5 completions. It is tabularized in alignment with the distributions presented in the ORCA paper and currently represents a partial completion of the full intended dataset, with ongoing generation to expand its scope. The data is primarily used for training and evaluation in the field of natural language processing.



```
from datasets import load_dataset

ds = (
    load_dataset(
        "Open-Orca/OpenOrca",
        data_files="1M-GPT4-Augmented.parquet",
        split="train[:100]",
    )
    .to_pandas()
    .drop(columns=["id"])
    .rename(columns={"response": "reference"})
)

dataset = ds.sample(n=10)
```

#### Preview the dataset


```
dataset.head()
```

## Run evaluation on 1P models

The *Gen AI Evaluation Service SDK* includes native support for evaluating Gemini models. This streamlines the evaluation process for Google's latest and most capable family of large language models. With minimal coding effort, you can leverage pre-defined metrics and workflows to assess the performance of Gemini models on various tasks. You can also customize your own model-based metrics based on your specific evaluation criteria.

This enhanced support enables you to:

- **Quickly evaluate Gemini models:** Effortlessly assess the performance of Gemini models using the SDK's streamlined workflows.
- **Compare models side-by-side:** Benchmark Gemini against other models to understand relative strengths and weaknesses.
- **Analyze prompt templates:** Evaluate the effectiveness of different prompt designs for optimizing Gemini's performance.

This native integration simplifies the evaluation process, allowing you to focus on understanding and improving the capabilities of Gemini.

#### Understand the `EvalTask` class

The `EvalTask` class is a core component of the *Gen AI Evaluation Service SDK* framework. It allows you to define and run evaluation jobs against your Gen AI models/applications, providing a structured way to measure their performance on specific tasks. Think of an `EvalTask` as a blueprint for your evaluation process.

`EvalTask` class requires an evaluation dataset and a list of metrics. Supported metrics are documented on the Generative AI on Vertex AI [Define your evaluation metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval) page. The dataset can be an `pandas.DataFrame`, Python dictionary or a file path URI and can contain default column names such as `prompt`, `reference`, `response`, and `baseline_model_response`. 

* **Bring-your-own-response (BYOR):** You already have the data that you want to evaluate stored in the dataset. You can customize the response column names for both your model and the baseline model using parameters like `response_column_name` and `baseline_model_response_column_name` or through the `metric_column_mapping`.

* **Perform model inference without a prompt template:** You have a dataset containing the input prompts to the model and want to perform model inference before evaluation. A column named `prompt` is required in the evaluation dataset and is used directly as input to the model.

* **Perform model inference with a prompt template:** You have a dataset containing the input variables to the prompt template and want to assemble the prompts for model inference. Evaluation dataset must contain column names corresponding to the variable names in the prompt template. For example, if prompt template is "Instruction: {instruction}, context: {context}", the dataset must contain `instruction` and `context` columns.



`EvalTask` supports extensive evaluation scenarios including BYOR, model inference with Gemini models, 3P models endpoints/SDK clients, or custom model generation functions, using computation-based metrics, model-based pointwise and pairwise metrics. The `evaluate()` method triggers the evaluation process, optionally taking a model, prompt template, experiment logging configuartions, and other evaluation run configurations. You can view the SDK reference documentation for [Gen AI Evaluation package](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/python/latest/vertexai.evaluation) for more details.

### Define a model


```
# Model to be evaluated
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config={"temperature": 0.6, "max_output_tokens": 256, "top_k": 1},
)
```

### Use computation-based metrics



```
# Define an EvalTask with ROUGE-L-SUM metric
rouge_eval_task = EvalTask(
    dataset=dataset,
    metrics=["rouge_l_sum"],
)
rouge_result = rouge_eval_task.evaluate(
    model=model,
    prompt_template="# System_prompt\n{system_prompt} # Question\n{question}",
)
```


```
display_eval_result(rouge_result)
```

### Use model-based pointwise metrics

#### Select a pointwise metric to use


```
supported_example_metric_names = (
    MetricPromptTemplateExamples.list_example_metric_names()
)

for metric in supported_example_metric_names:
    print(metric)
```


```
import ipywidgets as widgets

pointwise_single_turn_metrics = [
    metric
    for metric in supported_example_metric_names
    if not metric.startswith("pairwise") and not metric.startswith("multi_turn")
]

dropdown = widgets.Dropdown(
    options=pointwise_single_turn_metrics,
    description="Select a metric:",
    font_weight="bold",
    style={"description_width": "initial"},
)


def dropdown_eventhandler(change):
    global POINTWISE_METRIC
    if change["type"] == "change" and change["name"] == "value":
        POINTWISE_METRIC = change.new
        print("Selected:", change.new)


POINTWISE_METRIC = dropdown.value
dropdown.observe(dropdown_eventhandler, names="value")
display(dropdown)
```


```
# Define an EvalTask with a pointwise model-based metric
pointwise_eval_task = EvalTask(
    dataset=dataset,
    metrics=[POINTWISE_METRIC],
)

pointwise_result = pointwise_eval_task.evaluate(
    model=model,
    prompt_template="# System_prompt\n{system_prompt} # Question\n{question}",
)
```


```
display_eval_result(pointwise_result)
```


```
display_explanations(pointwise_result, metrics=[POINTWISE_METRIC], n=1)
```

#### Build your own pointwise metric

For more inforamation about metric customization, see this [notebook tutorial](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/customize_model_based_metrics.ipynb).


```
# Create a unique model-based metric for your own use cases
linguistic_acceptability = PointwiseMetric(
    metric="linguistic_acceptability",
    metric_prompt_template=PointwiseMetricPromptTemplate(
        criteria={
            "Proper Grammar": "The language's grammar rules are correctly followed, including but not limited to sentence structures, verb tenses, subject-verb agreement, proper punctuation, and capitalization.",
            "Appropriate word choice": "Words chosen are appropriate and purposeful given their relative context and positioning in the text. Vocabulary demonstrates prompt understanding.",
            "Reference Alignment": "The response is consistent and aligned with the reference.",
        },
        rating_rubric={
            "5": "Excellent: The writing is grammatically correct, uses appropriate vocabulary and aligns perfectly with the reference.",
            "4": "Good: The writing is generally grammatically correct, uses appropriate vocabulary and aligns well with the reference.",
            "3": "Satisfactory: The writing may have minor grammatical errors or use less-appropriate vocabulary, but it aligns reasonably well with the reference.",
            "2": "Unsatisfactory: The writing has significant grammatical errors, uses inappropriate vocabulary, deviates significantly from the reference.",
            "1": "Poor: The writing is riddled with grammatical errors, uses highly inappropriate vocabulary, is completely unrelated to the reference.",
        },
        input_variables=["prompt", "reference"],
    ),
)
```


```
pointwise_eval_task = EvalTask(
    dataset=dataset,
    metrics=[linguistic_acceptability],
)

pointwise_result = pointwise_eval_task.evaluate(
    model=model,
    prompt_template="# System_prompt\n{system_prompt} # Question\n{question}",
)
```


```
display_eval_result(pointwise_result)
```

### Use model-based pairwise metrics

Evaluate two Gen AI models side-by-side (SxS) with model-based pairwise metrics.

#### Select a pairwise metric to use


```
from IPython.display import display
import ipywidgets as widgets

pairwise_single_turn_metrics = [
    metric
    for metric in supported_example_metric_names
    if metric.startswith("pairwise") and "multi_turn" not in metric
]

dropdown = widgets.Dropdown(
    options=pairwise_single_turn_metrics,
    description="Select a metric:",
    font_weight="bold",
    style={"description_width": "initial"},
)


def dropdown_eventhandler(change):
    global pairwise_metric_name
    if change["type"] == "change" and change["name"] == "value":
        pairwise_metric = change.new
        print("Selected:", change.new)


pairwise_metric_name = dropdown.value
dropdown.observe(dropdown_eventhandler, names="value")
display(dropdown)
```

#### Define a baseline model to compare against


```
# Define a baseline model for pairwise comparison
baseline_model = GenerativeModel("gemini-1.5-flash-001")
```

#### Run SxS evaluation


```
# Create a pairwise metric
PAIRWISE_METRIC = PairwiseMetric(
    metric=pairwise_metric_name,
    metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template(
        pairwise_metric_name
    ),
    baseline_model=baseline_model,
)
```


```
pairwise_eval_task = EvalTask(
    dataset=dataset,
    metrics=[PAIRWISE_METRIC],
)
# Specify a candidate model for pairwise comparison
pairwise_result = pairwise_eval_task.evaluate(
    model=model,
    prompt_template="# System_prompt\n{system_prompt} # Question\n{question}",
)
```


```
display_eval_result(
    pairwise_result,
    title="Gemini-1.5-Flash vs. Gemini-1.5-Pro SxS Pairwise Evaluation Results",
)
```

## Run evaluation on 3P Models

The *Vertex Gen AI Evaluation Service SDK* provides robust support for evaluating third-party (3P) models, allowing you to assess and compare models from various sources. The *Gen AI Evaluation Service SDK* allows you to provide a generic Python function as input to specify how the model/application should be invoked for batch inference, which could be done through an endpoint or an SDK. This flexible approach accommodates a wide range of open and closed models.

**Open Models:**

Evaluate open models like a pre-trained `Llama 3.1` or a fine-tuned `Llama 3` models deployed with Vertex AI Model Garden using the *Gen AI Evaluation Service SDK*. This enables you to assess the performance of these models and understand how they align with your specific requirements.

**Closed Models:**

Evaluate the performance of closed model APIs, such as Anthropic's `Claude 3.5 Sonnet`, hosted on Vertex AI. This allows you to compare the capabilities of different closed models and make informed decisions about which best suits your needs.


```
# Define an EvalTask with a list of metrics
pointwise_eval_task = EvalTask(
    dataset=dataset,
    metrics=[
        "coherence",
        "fluency",
        "instruction_following",
        "text_quality",
        "rouge_l_sum",
    ],
)
```

### Run Evaluation on Llama 3.1 API Endpoint

You can experiment with various supported Llama models.

This tutorial use Llama 3 8B Instruct, 70B Instruct, and 405B Instruct using Model-as-a-Service (MaaS). Using Model-as-a-Service (MaaS), you can access Llama 3.1 models in just a few clicks without any setup or infrastructure hassles. Model-as-a-Service (MaaS) integrates [Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-3-8B) as a safety filter. It is switched on by default and can be switched off. Llama Guard enables us to safeguard model inputs and outputs. If a response is filtered, it will be populated with a `finish_reason` field (with value `content_filtered`) and a `refusal` field (stating the filtering reason).

You can also access Llama models for self-service in Vertex AI Model Garden, allowing you to choose your preferred infrastructure. [Check out Llama 3.1 model card](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama3_1?_ga=2.31261500.2048242469.1721714335-1107467625.1721655511) to learn how to deploy a Llama 3.1 models on Vertex AI.


```
MODEL_ID = "meta/llama3-8b-instruct-maas"  # @param {type:"string"} ["meta/llama3-8b-instruct-maas", "meta/llama3-70b-instruct-maas", "meta/llama3-405b-instruct-maas"]
```

#### Authentication

You can request an access token from the default credentials for the current environment. Note that the access token lives for [1 hour by default](https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.



```
credentials, _ = default()
auth_request = transport.requests.Request()
credentials.refresh(auth_request)
```

Then configure the OpenAI SDK to point to the Llama 3.1 API endpoint.

Note: only `us-central1` is supported region for Llama 3.1 models using Model-as-a-Service (MaaS).


```
MODEL_LOCATION = "us-central1"

client = openai.OpenAI(
    base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
    api_key=credentials.token,
)
```

#### Set model configurations for Llama 3.1

Use the following parameters to generate different answers:

*   `temperature` to control the randomness of the response
*   `max_tokens` to limit the response length
*   `top_p` to control the quality of the response


```
temperature = 1.0  # @param {type:"number"}
max_tokens = 50  # @param {type:"integer"}
top_p = 1.0  # @param {type:"number"}
apply_llama_guard = True  # @param {type:"boolean"}

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "Hello, Llama 3.1!"}],
    extra_body={
        "extra_body": {
            "google": {
                "model_safety_settings": {
                    "enabled": apply_llama_guard,
                    "llama_guard_settings": {},
                }
            }
        }
    },
)
print(response.choices[0].message.content)
```

#### Define the Llama 3.1 Model Function


```
def llama_model_fn(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra_body={
            "extra_body": {
                "google": {
                    "model_safety_settings": {
                        "enabled": apply_llama_guard,
                        "llama_guard_settings": {},
                    }
                }
            }
        },
    )
    return response.choices[0].message.content
```

#### Run evaluation on Llama 3.1 API Service


```
llama_result = pointwise_eval_task.evaluate(
    model=llama_model_fn,
    prompt_template="# System_prompt\n{system_prompt} # Question\n{question}",
)
```


```
display_eval_result(llama_result, title="Llama 3.1 API Service Evaluation Results")
```


```
display_explanations(llama_result, n=1)
```

### Run Evaluation on Claude 3



```
MODEL = "claude-3-5-sonnet@20240620"  # @param ["claude-3-5-sonnet@20240620", "claude-3-opus@20240229", "claude-3-haiku@20240307", "claude-3-sonnet@20240229" ]
if MODEL == "claude-3-5-sonnet@20240620":
    available_regions = ["europe-west1", "us-east5"]
elif MODEL == "claude-3-opus@20240229":
    available_regions = ["us-east5"]
elif MODEL == "claude-3-haiku@20240307":
    available_regions = ["us-east5", "europe-west1"]
elif MODEL == "claude-3-sonnet@20240229":
    available_regions = ["us-east5"]

dropdown = widgets.Dropdown(
    options=available_regions,
    description="Select a location:",
    font_weight="bold",
    style={"description_width": "initial"},
)


def dropdown_eventhandler(change):
    global LOCATION
    if change["type"] == "change" and change["name"] == "value":
        LOCATION = change.new
        print("Selected:", change.new)


LOCATION = dropdown.value
dropdown.observe(dropdown_eventhandler, names="value")
display(dropdown)
```

#### Define the Claude 3 Model Function


```
def anthropic_claude_model_fn(prompt):
    client = AnthropicVertex(region=LOCATION, project_id=PROJECT_ID)
    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model=MODEL,
    )
    response = message.content[0].text
    return response
```

#### Run evaluation on Claude 3 Model


```
claude_result = pointwise_eval_task.evaluate(
    model=anthropic_claude_model_fn,
    prompt_template="# System_prompt\n{system_prompt} # Question\n{question}",
)
```


```
display_eval_result(claude_result, title="Claude-3.5-Sonnet Evaluation Results")
```


```
display_explanations(claude_result, n=1)
```

## Prompt Engineering

The *Vertex AI Gen AI Evaluation Service SDK* simplifies prompt engineering by streamlining the process of creating and evaluating multiple prompt templates. It allows you to efficiently test different prompts against a chosen dataset and compare their performance using comprehensive evaluation metrics. This empowers you to identify the most effective prompts for your specific use case and optimize your generative AI applications.

### Design a prompt with Prompt Template


```
system_instruction = "You are a poetic assistant, skilled in explaining complex concepts with creative flair."
question = "How does LLM work?"
requirements = "Explain concepts in great depth using simple terms, and give examples to help people learn. At the end of each explanation, you ask a question to check for understanding"

prompt_template = f"{system_instruction} Answer this question: {question}, and follow the requirements: {requirements}."


model_response = (
    GenerativeModel("gemini-1.5-pro")
    .generate_content(prompt_template)
    .candidates[0]
    .content.parts[0]
    .text
)


display(HTML(f"<h2>Assembled Prompt:</h2><hr><h3>{prompt_template}</h3>"))
display(HTML("<h2>Model Response: </h2><hr>"))
Markdown(model_response)
```

###  Compare and optimize prompt template design

#### Define an evaluation dataset


To perform pointwise inference, the evaluation dataset is required to contain the following fields:

* Instruction: Part of the input user prompt. It refers to the inference instruction that is sent to your LLM.
* Context: User input for the Gen AI model or application in the current turn.
* Reference: The ground truth to compare your LLM response to.

Your dataset must include a minimum of one evaluation example. We recommend around 100 examples to ensure high-quality aggregated metrics and statistically significant results.


```
instruction = "Summarize the following article: \n"

context = [
    "Typhoon Phanfone has killed at least one person, a US airman on Okinawa who was washed away by high waves. Thousands of households have lost power and Japan's two largest airlines have suspended many flights. The storm also forced the suspension of the search for people missing after last week's volcanic eruption. The storm-tracking website Tropical Storm Risk forecasts that Phanfone will rapidly lose power over the next few hours as it goes further into the Pacific Ocean. Typhoon Phanfone was downgraded from an earlier status of a super typhoon, but the Japan Meteorological Agency had warned it was still a dangerous storm. Japan averages 11 typhoons a year, according to its weather agency. The typhoon made landfall on Monday morning near the central city of Hamamatsu, with winds of up to 180 km/h (112 mph). The airman was one of three US military personnel swept away by high waves whipped up by the typhoon off southern Okinawa island, where the US has a large military base. The remaining two are still missing. A police spokesman said they had been taking photographs of the sea. A university student who was surfing off the seas of Kanagawa Prefecture, south of Tokyo, was also missing, national broadcast NHK reports. It said at least 10 people had been injured and 9,500 houses were without power. The storm was expected to deposit about 100mm of rain on Tokyo over 24 hours, according to the Transport Ministry website. Many schools were closed on Monday and two car companies in Japan halted production at some plants ahead of the storm. More than 174 domestic flights were affected nationwide, NHK state broadcaster said on Sunday. On Sunday, heavy rain delayed the Japanese Formula One Grand Prix in Suzaka. French driver Jules Bianchi lost control in the wet conditions and crashed, sustaining a severe head injury.",
    "The blaze started at the detached building in Drivers End in Codicote, near Welwyn, during the morning. There was another fire at the building 20 years ago, after which fire-proof foil was placed under the thatch, which is protecting the main building. More than 15 fire engines and support vehicles were called to tackle the blaze. Roads in the area were closed and traffic diverted.",
    'The 18-year-old fell at the New Charter Academy on Broadoak Road in Ashton-under-Lyne at about 09:10 BST, Greater Manchester Police (GMP) said. GMP said he had gone to Manchester Royal Infirmary and his condition was "serious". Principal Jenny Langley said the school would remain "fully open" while police investigated. "Our thoughts are with the family and we\'re doing everything we can to support them along with staff and pupils," she said.',
    'But Belgian-born Dutchman Max Verstappen was unable to drive a car legally on his own in either country. That all changed on Wednesday when the youngster turned 18 and passed his driving test at the first attempt. Despite having competed in 14 grands prix since his debut in Australia in March, Verstappen admitted to feeling the pressure during his test. "It\'s a relief," said the Toro Rosso driver, who finished ninth in Japan on Sunday and had only started driving lessons a week ago. "I was a bit nervous to make mistakes, but the exam went well." A bonus of turning 18 is that Verstappen will now be able to drink the champagne if he ever makes it onto the podium.',
]

reference = [
    "A powerful typhoon has brought many parts of Japan to a standstill and briefly battered Tokyo before heading out to sea.",
    "A major fire has been burning in the thatched roof of a large property in Hertfordshire.",
    "A student has been taken to hospital after falling from a balcony at a Greater Manchester school.",
    "He is Formula 1's youngest ever driver and in charge of a car that can reach over 200mph.",
]

response = [
    "Typhoon Phanfone, while downgraded from super typhoon status, caused significant disruption and tragedy in Japan. One US airman died after being swept away by high waves, with two more missing. The storm caused power outages for thousands, flight cancellations, and the suspension of rescue efforts for missing volcano victims. Heavy rain and strong winds led to school and factory closures, transportation disruptions, and at least 10 injuries. The typhoon is expected to weaken as it moves over the Pacific Ocean.",
    "A large fire broke out in a detached thatched building in Codicote, near Welwyn. This is the second fire at the building in 20 years. Thankfully, fire-proof foil installed after the previous fire is protecting the main building. Over 15 fire engines and support vehicles responded, closing roads and diverting traffic in the area.",
    "An 18-year-old student at New Charter Academy in Ashton-under-Lyne suffered a serious fall and was hospitalized. The incident is under investigation by Greater Manchester Police, but the school remains open. The principal expressed support for the student's family and the school community.",
    "Max Verstappen, a Formula One driver, was finally able to get his driver's license at age 18. Despite already competing in 14 Grand Prix races, he was not of legal driving age in his native countries. He admitted to being nervous but passed the test on his first attempt.  As an added bonus of turning 18, Verstappen can now enjoy champagne on the podium if he places.",
]

eval_dataset = pd.DataFrame(
    {
        "instruction": instruction,
        "context": context,
        "reference": reference,
    }
)
```

#### Define prompt templates to compare



```
prompt_templates = [
    "Instruction: {instruction} such that you're explaining it to a 5 year old. Article: {context}. Summary:",
    "Article: {context}. Complete this task: {instruction}. Summary:",
    "Goal: {instruction} and give me a TL;DR in five words. Here's an article: {context}. Summary:",
    "Article: {context}. Reference Summary: {reference}. {instruction} to be more concise and verbose than the reference.",
]
```

#### Define a model


```
generation_config = {"temperature": 0.3, "max_output_tokens": 256, "top_k": 1}

gemini_model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=generation_config,
)
```

#### Define an EvalTask


```
metrics = [
    "rouge_l_sum",
    "bleu",
    "fluency",
    "coherence",
    "safety",
    "groundedness",
    "summarization_quality",
    "verbosity",
    "instruction_following",
    "text_quality",
]
```


```
EXPERIMENT_NAME = "eval-sdk-prompt-engineering"  # @param {type:"string"}

summarization_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment=EXPERIMENT_NAME,
)
```

#### Run Evaluation


```
eval_results = []
for i, prompt_template in enumerate(prompt_templates):
    eval_result = summarization_eval_task.evaluate(
        prompt_template=prompt_template,
        model=model,
        # Customize eval service rate limit based on your project's Gemini-1.5-pro model quota to improve speed.
        # See more details in https://cloud.google.com/vertex-ai/generative-ai/docs/models/run-evaluation#increase-quota
        evaluation_service_qps=1,
    )

    eval_results.append((f"Prompt Template #{i+1}", eval_result))
```

#### Display Evaluation report and explanations


```
for result in eval_results:
    display_eval_result(title=result[0], eval_result=result[1])
```


```
for eval_result in eval_results:
    display_explanations(eval_result[1], metrics=["summarization_quality"], n=2)
```

#### Visualize Results


```
display_radar_plot(
    eval_results,
    metrics=["instruction_following", "fluency", "coherence", "text_quality"],
)
```


```
display_bar_plot(
    eval_results,
    metrics=["instruction_following", "fluency", "coherence", "text_quality"],
)
```

####  View Experiment log for evaluation runs


```
summarization_eval_task.display_runs()
```

## Cleaning up


```
delete_experiment = True

# Please set your LOCATION to the same one used during Vertex AI SDK initialization.
LOCATION = "us-central1"  # @param {type:"string"}

if delete_experiment:

    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    experiment = aiplatform.Experiment(EXPERIMENT_NAME)
    experiment.delete()
```
