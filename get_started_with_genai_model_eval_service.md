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

# Get started with Generative AI evaluation service

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/legacy/get_started_with_genai_model_eval_service.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Flegacy%2Fget_started_with_genai_model_eval_service.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/legacy/get_started_with_genai_model_eval_service.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/legacy/get_started_with_genai_model_eval_service.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Ivan Nardini](https://github.com/inardini)|

## Overview

Assessing the performance of Large Language Models (LLMs) remains a complex task, especially when it comes to integrating them into production systems. Unlike conventional software and non-generative machine learning models, evaluating LLMs is subjective, challenging to automate, and prone to highly visible errors.

To tackle these challenges, Vertex AI offers a comprehensive evaluation framework through its Model Evaluation service. This framework encompasses the entire LLM lifecycle, from prompt engineering and model comparison to operationalizing automated model evaluation in production environments.

Learn more about [Vertex AI Generative AI evaluation service](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluate-models).

### Objective

In this tutorial, you learn how to use the Vertex AI Model Evaluation framework to evaluate Gemini, PaLM2 and Gemma in a summarization task.

This tutorial uses the following Google Cloud ML services and resources:

- Vertex AI Model Eval
- Vertex AI Pipelines
- Vertex AI Prediction

The steps performed include:

- Use Vertex AI Rapid Eval SDK to find the best prompt for a given model.
- Use Vertex AI Rapid Eval SDK to validate the best prompt across several models.
- Use Vertex AI Model Eval Pipeline service to measure performance and compare models with a more systematic evaluation.

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
%pip install --upgrade --quiet google-cloud-aiplatform[rapid_evaluation]
%pip install --upgrade --quiet datasets==2.18.0
%pip install --upgrade --quiet plotly==5.20.0
%pip install --upgrade --quiet nest-asyncio==1.6.0
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

3. [Enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

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

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.

This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
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

#### Service Account

**If you don't know your service account**, try to get your service account using `gcloud` command by executing the second cell below.


```
SERVICE_ACCOUNT = "[your-service-account]"  # @param {type:"string"}
```


```
import os
import sys

IS_COLAB = "google.colab" in sys.modules
if (
    SERVICE_ACCOUNT == ""
    or SERVICE_ACCOUNT is None
    or SERVICE_ACCOUNT == "[your-service-account]"
):
    # Get your service account from gcloud
    if not IS_COLAB:
        shell_output = !gcloud auth list 2>/dev/null
        SERVICE_ACCOUNT = shell_output[2].replace("*", "").strip()

    if IS_COLAB:
        shell_output = ! gcloud projects describe  $PROJECT_ID
        project_number = shell_output[-1].split(":")[1].strip().replace("'", "")
        SERVICE_ACCOUNT = f"{project_number}-compute@developer.gserviceaccount.com"

    print("Service Account:", SERVICE_ACCOUNT)
```

#### Set service account access

Run the following commands to grant your service account access


```
! gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectCreator $BUCKET_URI

! gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectViewer $BUCKET_URI
```

### Set tutorial folder

Set a folder to collect data and any tutorial artifacts.


```
from pathlib import Path as path

root_path = path.cwd()
tutorial_path = root_path / "tutorial"
data_path = tutorial_path / "data"

data_path.mkdir(parents=True, exist_ok=True)
```

### Import libraries


```
# General
import logging
import random
import string
import warnings

from IPython.display import HTML, Markdown, display

# Gen AI Evaluation
import datasets
from google.cloud import aiplatform
import nest_asyncio
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
from vertexai.preview.evaluation import EvalTask, make_metric
from vertexai.preview.language_models import (
    EvaluationTextSummarizationSpec,
    TextGenerationModel,
)
```

### Libraries settings

Set warnings, logging and Hugging Face datasets configuration to run tutorial.


```
warnings.filterwarnings("ignore")
nest_asyncio.apply()
datasets.disable_progress_bar()
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
```

### Initialize Vertex AI SDK for Python

Initialize the Vertex AI SDK for Python for your project.


```
vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)
```

### Define constants

Define evaluation dataset Cloud Bucket uris, AutoSxS pipeline template and pipeline root to use in this tutorial.


```
AUTO_METRICS_EVALUATION_FILE_URI = (
    "gs://github-repo/evaluate-gemini/sum_eval_palm_dataset_001.jsonl"
)

AUTOSXS_EVALUATION_FILE_URI = (
    "gs://github-repo/evaluate-gemini/sum_eval_gemini_dataset_001.jsonl"
)

AUTO_SXS_TEMPLATE_URI = (
    "https://us-kfp.pkg.dev/ml-pipeline/google-cloud-registry/autosxs-template/default"
)
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline"
```

### Helper functions

Initialize some helper functions to display evaluation results.


```
def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def display_eval_report(
    eval_result: tuple[str, dict, pd.DataFrame], metrics: list[str] = None
) -> None:
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


def display_explanations(
    df: pd.DataFrame, metrics: list[str] = None, n: int = 1
) -> None:
    """Display the explanations for the evaluation results."""

    # Set the style
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"

    # Sample the DataFrame
    df = df.sample(n=n)

    # Filter the DataFrame based on the selected metrics
    if metrics:
        df = df.filter(
            ["context", "reference", "completed_prompt", "response"]
            + [
                metric
                for metric in df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    # Display the explanations
    for index, row in df.iterrows():
        for col in df.columns:
            display(HTML(f"<h2>{col}:</h2> <div style='{style}'>{row[col]}</div>"))
        display(HTML("<hr>"))


def plot_radar_plot(eval_results, metrics=None):
    """Plot a radar plot for the evaluation results."""

    # Set the figure
    fig = go.Figure()

    # Create the radar plot for the evaluation metrics
    for eval_result in eval_results:
        title, summary_metrics, report_df = eval_result

        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        fig.add_trace(
            go.Scatterpolar(
                r=list(summary_metrics.values()),
                theta=list(summary_metrics.keys()),
                fill="toself",
                name=title,
            )
        )

    # Update figure layout
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True
    )

    fig.show()


def plot_bar_plot(
    eval_results: tuple[str, dict, pd.DataFrame], metrics: list[str] = None
) -> None:
    """Plot a bar plot for the evaluation results."""

    # Create data for the bar plot
    data = []
    for eval_result in eval_results:
        title, summary_metrics, _ = eval_result
        if metrics:
            summary_metrics = {
                k: summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric in k for selected_metric in metrics)
            }

        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    # Update the figure with the data
    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group")
    fig.show()


def print_aggregated_metrics(job: aiplatform.PipelineJob) -> None:
    """Print AutoMetrics"""

    # Collect rougeLSum
    rougeLSum = (
        round(
            job.task_details[0]
            .outputs["evaluation_metrics"]
            .artifacts[0]
            .metadata["rougeLSum"],
            3,
        )
        * 100
    )

    # Display the metric
    display(
        HTML(
            f"<h3>The {rougeLSum}% of the reference summary is represented by LLM when considering the longest common subsequence (LCS) of words.</h3>"
        )
    )


def print_autosxs_judgments(df: pd.DataFrame, n: int = 3):
    """Print AutoSxS judgments"""

    # Set the style
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"

    # Sample the dataframe
    df = df.sample(n=n)

    # Display the autorater explanations
    for index, row in df.iterrows():
        if row["confidence"] >= 0.5:
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


def print_autosxs_win_metrics(scores: dict) -> None:
    """Print AutoSxS aggregated metrics"""

    score_b = round(scores["autosxs_model_b_win_rate"] * 100)
    display(
        HTML(
            f"<h3>AutoSxS Autorater prefers {score_b}% of time Model B over Model A </h3>"
        )
    )
```

### Initiate LLMs

Initialize LLMs to evaluate with associated settings. For more information, see the [Vertex AI documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/overview).


```
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

generation_config = GenerationConfig(
    temperature=0.8,
    max_output_tokens=128,
)

llm1_model = GenerativeModel(
    "gemini-1.0-pro-002",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
```


```
llm2_model = TextGenerationModel.from_pretrained("text-bison@002")
```

## Vertex AI Model Evaluation for prompt engineering and model comparison using Rapid Eval SDK

To create more effective prompts that generate better output, you need to repeatedly test different prompts and interact with the LLMs multiple times to evaluate and validate your prompts.

The Rapid evaluation service allows you to evaluate prompts on demand using small data batches. You can use both predefined and custom metrics. And you can use the evaluation outputs in downstream representations for better understanding.

To use Rapid Eval SDK, you may want to cover the following steps:

1.   Initiate the evaluation dataset
2.   Define prompt templates and metrics
3.   Provide some model configurations
4.   Initiate an Evaluation Task
5.   Run an evaluation job

### Initiate the evaluation dataset

Prepare the dataset to evaluate prompts and compare models.


```
eval_dataset = datasets.load_dataset("xsum", split="all", data_dir=data_path)
eval_dataset = (
    eval_dataset.filter(lambda example: len(example["document"]) < 4096)
    .filter(lambda example: len(example["summary"]) < 4096)
    .rename_columns({"document": "context", "summary": "reference"})
    .remove_columns(["id"])
)

eval_sample_df = (
    eval_dataset.shuffle(seed=8)
    .select(random.sample(range(0, len(eval_dataset)), 3))
    .to_pandas()
)
```


```
eval_sample_df.head()
```

### Evaluate prompt engineering using predefined metrics

#### Define prompt templates and metrics

You provide some prompt templates you want to evaluate. Also you pass evaluation metrics. The metrics you choose will depend on whether or not you have access to ground truth data. If you have ground truth data, you can use computation-based metrics. If you don't have ground truth data, you can use pairwise model-based metrics. Check out the [documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview#determine-eval) to know more.


```
prompt_templates = [
    "Summarize the following article: {context}",
    "Summarize the following article in one main sentence: {context}",
    "Summarize the following article in three main sentences: {context}",
]

metrics = ["rouge_l_sum", "fluency", "coherence", "safety"]
```

#### Run the evaluation

To run evaluations for prompt templates, you run an evaluation job repeatedly against an evaluation dataset and its associated metrics. With EvalTask, you leverage integration with Vertex AI Experiments to track settings and results for each evaluation run.


```
experiment_name = "rapid-eval-with-llm1"

eval_task = EvalTask(
    dataset=eval_sample_df,
    metrics=metrics,
    experiment=experiment_name,
    content_column_name="content",
    reference_column_name="reference",
)

run_id = generate_uuid()

eval_results = []

for i, prompt_template in tqdm(
    enumerate(prompt_templates), total=len(prompt_templates)
):
    experiment_run_name = f"prompt-evaluation-llm1-{run_id}-{i}"

    eval_result = eval_task.evaluate(
        model=llm1_model,
        prompt_template=prompt_template,
        experiment_run_name=experiment_run_name,
    )

    eval_results.append(
        (f"Prompt #{i}", eval_result.summary_metrics, eval_result.metrics_table)
    )
```

#### Display Evaluation reports and explanations

Display detailed evaluation reports, explanations, and useful charts to summarize key metrics in an informative manner.


```
for eval_result in eval_results:
    display_eval_report(eval_result)
```


```
for eval_result in eval_results:
    display_explanations(eval_result[2], metrics=["fluency"])
```


```
plot_radar_plot(eval_results, metrics=["fluency/mean", "coherence/mean", "safety/mean"])
plot_bar_plot(eval_results, metrics=["rouge_l_sum/mean"])
```

**Comment**: You evaluate your prompts by passing metrics manually. Rapid Eval SDK supports metric-bundles which group metrics based on their tasks/criteria/inputs to facilitate convenient usage. For more information, check out the [Metrics bundles](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval#metric-bundles) documentation.

### Evaluate prompt engineering using locally-defined custom metrics

To evaluate your prompts with a custom metric, you need to define and register a function that encapsulates metric logic as an evaluation metric. In this case you define a custom faithfulness with score and explanation. Once these steps are completed, you can use the metric directly in the evaluation task.

##### Register CustomMetrics locally

Use helper function `make_metric` function to register a customly defined metric function.


```
def custom_faithfulness(instance: str) -> dict:
    """Evaluates the faithfulness of a text using an LLM."""

    response = instance["response"]

    def generate_prompt(task, score=None):
        """
        Generates a prompt for the LLM based on the task and optional score.
        """
        prompt_start = f"""You are examining written text content. Here is the text:

        [BEGIN DATA]
        ************
        [Text]: {response}
        ************
        [END DATA]

        """
        if task == "score":
            prompt_end = """Examine the text and determine whether the text is faithful or not.
                          Faithfulness refers to how accurately a generated summary reflects the essential information and key concepts present in the original source document.
                          A faithful summary stays true to the facts and meaning of the source text, without introducing distortions, hallucinations, or information that wasn't originally there.
                          Your response must be a single integer number on a scale of 0-5, 0 the least faithful and 5 being the most faithful."""

        elif task == "explain":
            prompt_end = f"""Consider the text has been scored as {score} in faithful using the following definition:
                             Faithfulness refers to how accurately a generated summary reflects the essential information and key concepts present in the original source document.
                             A faithful summary stays true to the facts and meaning of the source text, without introducing distortions, hallucinations, or information that wasn't originally there.
                             Your response must be an explanation of why the text is faithful or not. If score is -1.0, return "No explanation can be provided for this prompt\""""
        else:
            raise ValueError("Invalid task for prompt generation.")
        return prompt_start + prompt_end

    # Generate score prompt and extract the score
    score_prompt = generate_prompt("score")
    score_text = (
        llm1_model.generate_content(score_prompt).candidates[0].content.parts[0].text
    )

    try:
        score = int(score_text) / 1.0
    except ValueError:
        score = -1.0

    # Generate explanation prompt and extract explanation
    explanation_prompt = generate_prompt("explain", score)
    explanation = (
        llm1_model.generate_content(explanation_prompt)
        .candidates[0]
        .content.parts[0]
        .text
    )

    return {
        "custom_faithfulness": score,
        "explanation": f"The model gave a score of {score} with the following explanation: {explanation}",
    }
```


```
custom_faithfulness_metric = make_metric(
    name="custom_faithfulness",
    metric_function=custom_faithfulness,
)
```

##### Run the evaluation using the custom metric

Run evaluations for prompt templates against an evaluation dataset with the defined custom metrics.


```
experiment_name = "rapid-eval-with-llm1-custom-eval"

eval_task = EvalTask(
    dataset=eval_sample_df,
    metrics=metrics + [custom_faithfulness_metric],
    experiment=experiment_name,
)

run_id = generate_uuid()

eval_results = []

for i, prompt_template in tqdm(
    enumerate(prompt_templates), total=len(prompt_templates)
):
    experiment_run_name = f"prompt-evaluation-llm1-{run_id}-{i}"

    eval_result = eval_task.evaluate(
        model=llm1_model,
        prompt_template=prompt_template,
        experiment_run_name=experiment_run_name,
    )

    eval_results.append(
        (f"Prompt #{i}", eval_result.summary_metrics, eval_result.metrics_table)
    )
```

#### Display Evaluation reports and explanations

Display the resulting evaluation reports and explanations for the custom metrics.


```
for eval_result in eval_results:
    display_eval_report(eval_result, metrics=["row", "custom_faithfulness"])
```


```
for eval_result in eval_results:
    display_explanations(eval_result[2], metrics=["custom_faithfulness"])
```

### Validate prompt by comparing LLM 1 with LLM 2

Once you know which is the best prompt template according to your metrics, you can validate it across several models.

Vertex AI Rapid Eval SDK allows you to compare any models, including Google proprietary and open models, against an evaluation dataset with a prompt template and the defined metrics.


```
prompt_template = "Summarize the following article in three main sentences: {context}"  # @param {type:"string"}
```

#### Set model function

To compare a model which is not natively supported by the Vertex AI Rapid SDK, you can define a generate function. The function takes the prompt as input and generate a text as output.


```
def llm2_model_fn(prompt):
    return llm2_model.predict(prompt, **generation_config).text
```

#### Run the evaluation

To run evaluations along models, you run an evaluation job against an evaluation dataset and its associated metrics using EvalTask.


```
experiment_name = "rapid-eval-llm1-llm2-comparison"

models = {
    "llm1": llm1_model,
    "llm2": llm2_model_fn,
}

metrics = [
    "bleu",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "rouge_l_sum",
    "fluency",
    "coherence",
    "safety",
]

eval_task = EvalTask(
    dataset=eval_sample_df, metrics=metrics, experiment=experiment_name
)

run_id = generate_uuid()

eval_results = []

for i, (model_name, model) in tqdm(
    enumerate(zip(models.keys(), models.values())), total=len(models.keys())
):
    experiment_run_name = f"prompt-evaluation-{model_name}-{run_id}-{i}"

    eval_result = eval_task.evaluate(
        model=model,
        prompt_template=prompt_template,
        experiment_run_name=experiment_run_name,
    )

    eval_results.append(
        (f"Model {model_name}", eval_result.summary_metrics, eval_result.metrics_table)
    )
```

#### Display Evaluation reports and explanations

Display the resulting evaluation reports and explanations for each model.


```
for eval_result in eval_results:
    display_eval_report(eval_result)
```


```
plot_radar_plot(eval_results, metrics=["fluency/mean", "coherence/mean", "safety/mean"])
plot_bar_plot(
    eval_results,
    metrics=[
        "bleu/mean",
        "rouge_1/mean",
        "rouge_2/mean",
        "rouge_l/mean",
        "rouge_l_sum/mean",
    ],
)
```

## Vertex AI Model Evaluation at scale

If you're planning to deploy or are already using your Gen AI application with a prompt template and model, you might want to consider evaluating the model or comparing different models over a larger set of data over time.

In this scenario, you need a more systematic and scalable way to evaluate Gen AI application components.

The Vertex AI Eval service provides end-to-end prebuilt evaluation pipelines for evaluating generative AI models at scale by leveraging Vertex AI Pipelines. Two distinct evaluation pipelines are available:

*   Computation-based for pointwise metric-based evaluation.
*   AutoSxS for pairwise model-based evaluations.

To learn more, [check out](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview#pipeline_services_autosxs_and_computation-based) the official documentation.

### Using Vertex AI Model Evaluation Computation-based metrics.

You use Computation-based metrics (RougeLSum) for evaluating an LLM in summarization task.

To run a Computation-based evaluation pipeline job, you need to provide an evaluation dataset which contains both context and the ground truth summaries. Then you define an evaluation task configuration, in this case you use text summarization task. Finally, you submit an evaluation job associated with your LLM.

#### Read the evaluation data

Read the evaluation dataset as Pandas DataFrame for having a quick view.


```
evaluation_df = pd.read_json(AUTO_METRICS_EVALUATION_FILE_URI, lines=True)
evaluation_df = evaluation_df.rename(
    columns={"prompt": "input_text", "ground_truth": "output_text"}
)
evaluation_df.head()
```

#### Run a model evaluation job

Define a specification for text summarization model evaluation task and run the evaluation job.


```
task_spec = EvaluationTextSummarizationSpec(
    ground_truth_data=evaluation_df, task_name="summarization"
)
```


```
job = llm2_model.evaluate(task_spec=task_spec)
```

#### Evaluate the results

Display resulting metrics.


```
print_aggregated_metrics(job)
```

### Using Vertex AI Model Evaluation AutoSxS metrics

You use AutoSxS to compare different model responses for evaluating how better a model is able to generate summaries against another.

To run an AutoSxS evaluation job, you need to provide an evaluation dataset which contains context and responses of models you want to compare. Then you define AutoSxS parameters including task to evaluate, inference context and instructions and model response columns. Finally, you submit an evaluation job associated with your LLM.

#### Read the evaluation data

Read the evaluation dataset as Pandas DataFrame for having a quick view.


```
evaluation_df = pd.read_json(AUTOSXS_EVALUATION_FILE_URI, lines=True)
evaluation_df.head()
```

#### Run a model evaluation job

Define AutoSxS parameters and run the AutoSxS evaluation pipeline job.


```
display_name = f"autosxs-eval-{generate_uuid()}"
parameters = {
    "evaluation_dataset": AUTOSXS_EVALUATION_FILE_URI,
    "id_columns": ["id", "document"],
    "task": "summarization",
    "autorater_prompt_parameters": {
        "inference_context": {"column": "document"},
        "inference_instruction": {
            "template": "Summarize the following article in three main sentences: "
        },
    },
    "response_column_a": "response_a",
    "response_column_b": "response_b",
}
```


```
job = aiplatform.PipelineJob(
    job_id=display_name,
    display_name=display_name,
    pipeline_root=os.path.join(BUCKET_URI, display_name),
    template_path=AUTO_SXS_TEMPLATE_URI,
    parameter_values=parameters,
    enable_caching=False,
)
job.run(sync=True)
```

#### Evaluate the results

Vertex AI AutoSxS evaluation pipeline produces the following artifacts:


*   The judgments table is produced by the AutoSxS arbiter helping users understand model performance at the example level.
*   Aggregate metrics are produced by the AutoSxS metrics component helping users understand the most performing model compare to the task under evaluation.

To know more about the AutoSxS artifacts, [check out](https://cloud.google.com/vertex-ai/generative-ai/docs/models/side-by-side-eval#view-eval-results) the documentation.

##### AutoSxS Judgments


```
for details in job.task_details:
    if details.task_name == "online-evaluation-pairwise":
        break

judgments_uri = details.outputs["judgments"].artifacts[0].uri
judgments_df = pd.read_json(judgments_uri, lines=True)
```


```
print_autosxs_judgments(judgments_df)
```

##### AutoSxS Aggregate metrics


```
for details in job.task_details:
    if details.task_name == "model-evaluation-text-generation-pairwise":
        break

win_rate_metrics = details.outputs["autosxs_metrics"].artifacts[0].metadata
print_autosxs_win_metrics(win_rate_metrics)
```

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial.


```
import os
import shutil

delete_experiments = False
delete_pipeline = False
delete_bucket = False
delete_tutorial = False

# Delete Experiments
if delete_experiments or os.getenv("IS_TESTING"):
    experiments_list = aiplatform.Experiment.list()
    for experiment in experiments_list:
        experiment.delete()

# Delete Pipeline
if delete_pipeline or os.getenv("IS_TESTING"):
    pipelines_list = aiplatform.PipelineJob.list()
    for pipeline in pipelines_list:
        pipeline.delete()

# Delete Cloud Storage
if delete_bucket or os.getenv("IS_TESTING"):
    ! gsutil -m rm -r $BUCKET_URI

# Delete tutorial folder
if delete_tutorial or os.getenv("IS_TESTING"):
    shutil.rmtree(tutorial_path)
```
