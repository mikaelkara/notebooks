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

# Gen AI Evaluation Service SDK Preview-to-GA Migration Guide | Gen AI Evaluation SDK Tutorial


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/migration_guide_preview_to_ga_sdk.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fmigration_guide_preview_to_ga_sdk.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/migration_guide_preview_to_ga_sdk.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/migration_guide_preview_to_ga_sdk.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai), [Xi Liu](https://github.com/xiliucity) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.9

## Overview


In this tutorial, you will get detailed guidance on how to migrate from the Preview version to the latest GA version of *Vertex AI Python SDK for Gen AI Evaluation Service* to evaluate **Retrieval-Augmented Generation** (RAG) and compare two models **side-by-side (SxS)**.

In the GA release, instead of providing predefined black-box model-based metrics, the evaluation service start providing capability to support defining metrics based on your own criteria. You can still run out-of-box metrics through `MetricPromptTemplateExamples` class we provide in the SDK. The examples covers the following metrics in both Pointwise and Pairwise style.
* `coherence`
* `fluency`
* `safety`
* `groundedness`
* `instruction_following`
* `verbosity`
* `text_quality`
* `summarization_quality`
* `question_answering_quality`
* `multi_turn_chat_quality`
* `multi_turn_safety`

This notebook would focus on handling the breaking changes. If you need actionable help to deal with bugs triggered by breaking changes, please jump to the following sections:
* How to handle discontinued metrics
* How to handle the new input schema


To learn more about the GA release details, please refer to the latest documentation and notebook tutorials in [Vertex Gen AI Evaluation Service](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview).

The examples used in this notebook is from Stanford Question Answering Dataset [SQuAD 2.0](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15785042.pdf).


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
EXPERIMENT = "eval-migration-ga"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries

Please update import path to the GA version SDK by changing `from vertexai.preview.evaluation` to **`from vertexai.evaluation`**.


```
# General
import inspect
import logging
import warnings

from IPython.display import HTML, Markdown, display
import pandas as pd
import plotly.graph_objects as go

# Main
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PairwiseMetric,
    PointwiseMetric,
)
from vertexai.generative_models import GenerativeModel
```

### Library settings


```
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
```

### Helper functions


```
def print_doc(function):
    print(f"{function.__name__}:\n{inspect.getdoc(function)}\n")


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


def plot_radar_plot(eval_results, max_score=5, metrics=None):
    fig = go.Figure()

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

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_score])), showlegend=True
    )

    fig.show()


def display_radar_plot(eval_results, metrics=None):
    """Plot the radar plot."""
    fig = go.Figure()
    for item in eval_results:
        eval_result, title = item
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
    fig = go.Figure()
    data = []

    for eval_results in eval_results_list:
        eval_result, title = eval_results[0], eval_results[1]

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


def sample_pairwise_result(eval_result, n=1, metric=None):
    """Display a random row of pairwise metric result with model responses."""
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"
    df = eval_result.metrics_table
    df = df.sample(n=n)
    for index, row in df.iterrows():
        display(HTML(f"<h2>Prompt:</h2> <div style='{style}'>{row['prompt']}</div>"))
        display(
            HTML(
                f"<h2>Baseline Model Response:</h2> <div style='{style}'>{row['baseline_model_response']}</div>"
            )
        )
        display(
            HTML(
                f"<h2>Candidate Model Response:</h2> <div style='{style}'>{row['response']}</div>"
            )
        )
        display(
            HTML(
                f"<h2>Explanation:</h2> <div style='{style}'>{row[f'{metric}/explanation']}</div>"
            )
        )
        display(
            HTML(
                f"<h2>Winner:</h2> <div style='{style}'>{row[f'{metric}/pairwise_choice']}</div>"
            )
        )
        display(HTML("<hr>"))


def display_pairwise_win_rate(eval_result, metric=None):
    """Display pairwise aggregated metrics"""
    summary_metrics = eval_result.summary_metrics
    candidate_model_win_rate = round(
        summary_metrics[f"{metric}/candidate_model_win_rate"] * 100
    )
    display(
        HTML(
            f"<h3>Win rate: Autorater prefers Candidate Model over Baseline Model {candidate_model_win_rate}% of time.</h3>"
        )
    )
```

## How to handle discontinued metrics

We removed the following metrics support from SDK:
* `question_answering_helpfulness`
* `question_answering_relevance`
* `question_answering_correctness`
* `summarization_helpfulness`
* `summarization_verbosity`
* `fulfillment` (rename to `instruction_following`)

The rationale of this is because
* We now provide two generic metric interface (`PointwiseMetirc` and `PairwiseMetric`) for customers to define the metrics with their own criteria, which is more transpairent and more affective. We also provide
* Many of the metrics here should not be task specific. For example, `helpfulness`, `relevance`, and `verbosity` can be applied to all text-related tasks.
* Some metrics here are not intuitive to users and are very subjective. For example, what is `question_answering_helpfulness`? How to define `helpfulness`? For different customers, the criteria of `helpfulness` can be totally different.

**We recommend you to use the new `MetricPromptTemplateExamples` we provide, and adjust them for your own use cases.** However, if you still want to use the above discontinued metrics:
* You can pin to an old version of the SDK, since we still maintain the API to support those metrics. The previous version `1.62.0` would be the recommended preview version to pin to. Example code:

  ```%pip install -q google-cloud-aiplatform[rapid-evaluation]==1.62.0```

* You can use `instruction_following` to replace `fulfillment`, use `verbosity` to replace `summarization_verbosity`.

* We provide examples below to help you define `fulfillment`, `helpfulness`, `relevance` in case you would still like to use them in your application.


### Define my own version of the discontinued metrics

#### Prepare Dataset

To evaluate the RAG generated answers, the evaluation dataset is required to contain the following fields:

* Prompt: The user supplied prompt consisting of the User Question and the RAG Retrieved Context
* Response: The RAG Generated Answer


```
questions = [
    "Which part of the brain does short-term memory seem to rely on?",
    "What provided the Roman senate with exuberance?",
    "What area did the Hasan-jalalians command?",
]

retrieved_contexts = [
    "Short-term memory is supported by transient patterns of neuronal communication, dependent on regions of the frontal lobe (especially dorsolateral prefrontal cortex) and the parietal lobe. Long-term memory, on the other hand, is maintained by more stable and permanent changes in neural connections widely spread throughout the brain. The hippocampus is essential (for learning new information) to the consolidation of information from short-term to long-term memory, although it does not seem to store information itself. Without the hippocampus, new memories are unable to be stored into long-term memory, as learned from patient Henry Molaison after removal of both his hippocampi, and there will be a very short attention span. Furthermore, it may be involved in changing neural connections for a period of three months or more after the initial learning.",
    "In 62 BC, Pompey returned victorious from Asia. The Senate, elated by its successes against Catiline, refused to ratify the arrangements that Pompey had made. Pompey, in effect, became powerless. Thus, when Julius Caesar returned from a governorship in Spain in 61 BC, he found it easy to make an arrangement with Pompey. Caesar and Pompey, along with Crassus, established a private agreement, now known as the First Triumvirate. Under the agreement, Pompey's arrangements would be ratified. Caesar would be elected consul in 59 BC, and would then serve as governor of Gaul for five years. Crassus was promised a future consulship.",
    "The Seljuk Empire soon started to collapse. In the early 12th century, Armenian princes of the Zakarid noble family drove out the Seljuk Turks and established a semi-independent Armenian principality in Northern and Eastern Armenia, known as Zakarid Armenia, which lasted under the patronage of the Georgian Kingdom. The noble family of Orbelians shared control with the Zakarids in various parts of the country, especially in Syunik and Vayots Dzor, while the Armenian family of Hasan-Jalalians controlled provinces of Artsakh and Utik as the Kingdom of Artsakh.",
]

generated_answers = [
    "frontal lobe and the parietal lobe",
    "The Roman Senate was filled with exuberance due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Syunik and Vayots Dzor.",
]

baseline_answers = [
    "the frontal cortex and the parietal cortex, which are crucial for sensory and cognitive functions",
    "The Roman Senate celebrated triumphantly after significant victories over Catiline, bolstering their political influence",
    "The Hasan-Jalalians held control over the regions of Syunik and Vayots Dzor, maintaining power through strategic alliances and military strength",
]

eval_dataset = pd.DataFrame(
    {
        "instruction": questions,
        "context": retrieved_contexts,
        "response": generated_answers,
    }
)
```

#### Define the metrics


We will define `fulfillment`, `helpfulness`, and `relevance` here in order to replace the discontinued ones.


```
relevance_prompt_template = """
# Instruction
You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria.
You will be assessing question answering relevance, which measures the ability to respond with relevant information when asked a question.
You will assign the writing response a score from 5, 4, 3, 2, 1, following the INDIVIDUAL RATING RUBRIC and EVALUATION STEPS.

# Evaluation
## Criteria
Relevance: The response should be relevant to the instruction and directly address the instruction.

## Rating Rubric
5 (completely relevant): Response is entirely relevant to the instruction and provides clearly defined information that addresses the instruction's core needs directly.
4 (mostly relevant): Response is mostly relevant to the instruction and addresses the instruction mostly directly.
3 (somewhat relevant): Response is somewhat relevant to the instruction and may address the instruction indirectly, but could be more relevant and more direct.
2 (somewhat irrelevant): Response is minimally relevant to the instruction and does not address the instruction directly.
1 (irrelevant): Response is completely irrelevant to the instruction.

## Evaluation Steps
STEP 1: Assess relevance: is response relevant to the instruction and directly address the instruction?
STEP 2: Score based on the criteria and rubrics.

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

# User Inputs and AI-generated Response
## User Inputs
### INSTRUCTION
{instruction}

### CONTEXT
{context}

## AI-generated Response
{response}
"""

relevance = PointwiseMetric(
    metric="relevance",
    metric_prompt_template=relevance_prompt_template,
)
```


```
helpfulness_prompt_template = """
# Instruction
You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria.
You will be assessing question answering helpfulness, which measures the ability to provide important details when answering a question.
You will assign the writing response a score from 5, 4, 3, 2, 1, following the INDIVIDUAL RATING RUBRIC and EVALUATION STEPS.

# Evaluation
## Criteria
Helpfulness: The response is comprehensive with well-defined key details. The user would feel very satisfied with the content in a good response.

## Rating Rubric
5 (completely helpful): Response is useful and very comprehensive with well-defined key details to address the needs in the question and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.
4 (mostly helpful): Response is very relevant to the question, providing clearly defined information that addresses the question's core needs.  It may include additional insights that go slightly beyond the immediate question.  The user would feel quite satisfied with the content in the response.
3 (somewhat helpful): Response is relevant to the question and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.
2 (somewhat unhelpful): Response is minimally relevant to the question and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.
1 (unhelpful): Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.

## Evaluation Steps
STEP 1: Assess comprehensiveness: does the response provide specific, comprehensive, and clearly defined information for the user needs expressed in the question?
STEP 2: Assess relevance: When appropriate for the question, does the response exceed the question by providing relevant details and related information to contextualize content and help the user better understand the response.
STEP 3: Assess accuracy: Is the response free of inaccurate, deceptive, or misleading information?
STEP 4: Assess safety: Is the response free of harmful or offensive content?

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

# User Inputs and AI-generated Response
## User Inputs
### INSTRUCTION
{instruction}

### CONTEXT
{context}

## AI-generated Response
{response}
"""

helpfulness = PointwiseMetric(
    metric="helpfulness",
    metric_prompt_template=helpfulness_prompt_template,
)
```


```
fulfillment_prompt_template = """
# Instruction
You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria.
You will be assessing fulfillment, which measures the ability to follow instructions.
You will assign the writing response a score from 5, 4, 3, 2, 1, following the INDIVIDUAL RATING RUBRIC and EVALUATION STEPS.

# Evaluation
## Criteria
Instruction following: The response demonstrates a clear understanding of the instructions, satisfying all of the instruction's requirements.

## Rating Rubric
5 (complete fulfillment): Response addresses all aspects and adheres to all requirements of the instruction. The user would feel like their instruction was completely understood.
4 (good fulfillment): Response addresses most aspects and requirements of the instruction. It might miss very minor details or have slight deviations from requirements. The user would feel like their instruction was well understood.
3 (some fulfillment): Response does not address some minor aspects and/or ignores some requirements of the instruction. The user would feel like their instruction was partially understood.
2 (poor fulfillment): Response addresses some aspects of the instruction but misses key requirements or major components. The user would feel like their instruction was misunderstood in significant ways.
1 (no fulfillment): Response does not address the most important aspects of the instruction. The user would feel like their request was not at all understood.

## Evaluation Steps
STEP 1: Assess instruction understanding: Does the response address the intent of the instruction such that a user would not feel the instruction was ignored or misinterpreted by the response?
STEP 2: Assess requirements adherence: Does the response adhere to any requirements indicated in the instruction such as an explicitly specified word length, tone, format, or information that the response should include?

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

# User Inputs and AI-generated Response
## User Inputs
### INSTRUCTION
{instruction}

## AI-generated Response
{response}
"""

fulfillment = PointwiseMetric(
    metric="fulfillment",
    metric_prompt_template=fulfillment_prompt_template,
)
```

#### Run Evaluation with defined metrics

Now, you can run evaluation as before, using those three metrics.


```
eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[
        relevance,
        helpfulness,
        fulfillment,
    ],
    experiment=EXPERIMENT,
)

eval_result = eval_task.evaluate()
```


```
display_eval_result(eval_result)
```

## How to handle the new input schema


In the GA release, all of the `MetricPromptTemplateExamples` requires taking in `prompt` and `response`/`baseline_model_response`.  instead of more fine-grained inputs as before, such as `instruction` and `context`. The rationales are:
* For most users, input user prompt is what they have, instead of `instruction` or `context`. It's also difficult to preprocess the user input prompt and breaking it down to `instruction` and `context`.
* In an increasing number of use cases, we've observed that the input prompts contain such complex and intertwined information that they can't be broken down any further.

The solution is simple:
* **(Recommend)** Assemble "instruction" and "context" with a simple prompt_template `{instruction}: {context}` to a full input prompt and then use it for evaluation. Or just assemble them with a simple line of python of code.
* (Not recommend) Modify the MetricPromptTemplateExample and make it take "instruction" and "context" as inputs instead of "prompt".


### Example of preprocessing the dataset


```
new_eval_dataset = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + context
            for question, context in zip(questions, retrieved_contexts)
        ],
        "response": generated_answers,
    }
)
```


```
# Run evaluation with new metric prompt template examples
eval_task = EvalTask(
    dataset=new_eval_dataset,
    metrics=[
        "question_answering_quality",
        "groundedness",
        "safety",
        "instruction_following",
    ],
    experiment=EXPERIMENT,
)

eval_result = eval_task.evaluate()

display_eval_result(eval_result)
```

## How to migrate to `PairwiseMetric` for AutoSxS Evaluation

The pipeline-based `AutoSxS` evaluation will be deprecated and replaced by GA version of Gen AI Eval Service SDK. The rationales are:

* Better judge model (Autorater) quality: Gen AI Eval Service SDK uses the latest `Gemini-1.5-Pro` instead of legacy `PaLM` model that AutoSxS uses.

* Faster and easier to use: SDK provides a more faster and more intuitive interface than pipelines, allowing users to perform side-by-side (SxS) evaluation and see result more rapidly.

* More flexibility: You can define your own pairwise comparison criteria and rating rubrics, and compute multiple pairwise metrics together in an `EvalTask`.

**Solution:**

* Use `PairwiseMetric` class in Gen AI Eval Service SDK for performing SxS evaluation for 2 models.

* If you have a stored evaluation in Google Cloud Storage(GCS) or BigQuery(BQ), you can directly provide the URI in the `dataset` parameter when defining your `EvalTask`.

* If your dataset contains fine-grained columns like `instruction`, `context`,  assemble them with a simple prompt_template `{instruction}: {context}` to a full input prompt and then use it for evaluation. Or just assemble them with a simple line of python of code.


### Evaluate two models side-by-side with `PairwiseMetric`


```
eval_dataset = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + context
            for question, context in zip(questions, retrieved_contexts)
        ],
    }
)
```


```
# Define Baseline and Candidate models for pairwise comparison
baseline_model = GenerativeModel(
    "gemini-1.5-flash-001",
)
candidate_model = GenerativeModel("gemini-1.0-pro")
```


```
# Create a "Pairwise Text Quality" metric from examples
text_quality_prompt_template = MetricPromptTemplateExamples.get_prompt_template(
    "pairwise_text_quality"
)

pairwise_text_quality_metric = PairwiseMetric(
    metric="pairwise_text_quality",
    metric_prompt_template=text_quality_prompt_template,
    baseline_model=baseline_model,  # Specify baseline model for pairwise comparison
)
```


```
pairwise_text_quality_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[pairwise_text_quality_metric],
    experiment=EXPERIMENT,
)

# Specify candidate model for pairwise comparison
pairwise_text_quality_result = pairwise_text_quality_eval_task.evaluate(
    model=candidate_model,
)
```


```
display_eval_result(pairwise_text_quality_result)
```


```
sample_pairwise_result(
    pairwise_text_quality_result, metric="pairwise_text_quality", n=1
)
```


```
display_pairwise_win_rate(pairwise_text_quality_result, metric="pairwise_text_quality")
```

### Bring-your-own-response for SxS Evaluation

#### Calculate a pairwise metric on the saved responses in eval dataset


```
eval_dataset = pd.DataFrame(
    {
        "question": questions,
        "context": retrieved_contexts,
        "response": generated_answers,
        "baseline_model_response": baseline_answers,
    }
)
```


```
# Define an EvalTask with 2 example pairwise metrics
byor_pairwise_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[
        MetricPromptTemplateExamples.Pairwise.VERBOSITY,
        MetricPromptTemplateExamples.Pairwise.SAFETY,
    ],
    experiment=EXPERIMENT,
)

# Assemble the fine-grained columns with prompt template
byor_pairwise_result = byor_pairwise_eval_task.evaluate(
    prompt_template="Answer the question: {question} Context: {context}",
    evaluation_service_qps=10,
)
```


```
display_eval_result(byor_pairwise_result)
```


```
sample_pairwise_result(byor_pairwise_result, metric="pairwise_verbosity")
```


```
display_pairwise_win_rate(byor_pairwise_result, metric="pairwise_verbosity")
```
