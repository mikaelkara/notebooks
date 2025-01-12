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

# Evaluate Generated Answers from Retrieval-Augmented Generation (RAG) for Question Answering with Gen AI Evaluation Service SDK

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_rag_gen_ai_evaluation_service_sdk.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fevaluate_rag_gen_ai_evaluation_service_sdk.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_rag_gen_ai_evaluation_service_sdk.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/evaluate_rag_gen_ai_evaluation_service_sdk.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai), [Kelsi Lakey](https://github.com/lakeyk) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.9

## Overview


In this tutorial, you will learn how to use the use the *Vertex AI Python SDK for Gen AI Evaluation Service* to evaluate **Retrieval-Augmented Generation** (RAG) generated answers for **Question Answering** (QA) task.

RAG is a technique to improve groundness, relevancy and factuality of large language models (LLMs) by finding relevant information from the model's knowledge base. RAG is done by converting a query into a vector representation (embeddings), and then finding the most similar vectors in the knowledge base. The most similar vectors are then used to help generate the response.

This tutorial demonstrates how to use Gen AI Evaluation for a Bring-Your-Own-Response scenario:

* The context was retrieved and the answers were generated based on the retrieved context using RAG.

* Evaluate the quality of the RAG generated answers for QA task programmatically using the SDK.

The examples used in this notebook is from Stanford Question Answering Dataset [SQuAD 2.0](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15785042.pdf).

See also: 

- Learn more about [Vertex Gen AI Evaluation Service SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview).

## Getting Started

### Install Vertex AI SDK for Gen AI Evaluation


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

### Increase quota (optional)

Increasing the quota may lead to better performance and user experience. Read more about this at [online evaluation quotas](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#eval-quotas).

### Set Google Cloud project information and initialize Vertex AI SDK


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
EXPERIMENT = "rag-eval-01"  # @param {type:"string"}


import vertexai

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
import inspect
import logging
import warnings

# General
from IPython.display import HTML, Markdown, display
import pandas as pd
import plotly.graph_objects as go

# Main
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples, PointwiseMetric
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


def display_explanations(df, metrics=None, n=1):
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"
    df = df.sample(n=n)
    if metrics:
        df = df.filter(
            ["instruction", "context", "reference", "completed_prompt", "response"]
            + [
                metric
                for metric in df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    for index, row in df.iterrows():
        for col in df.columns:
            display(HTML(f"<h2>{col}:</h2> <div style='{style}'>{row[col]}</div>"))
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


def plot_bar_plot(eval_results, metrics=None):
    fig = go.Figure()
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

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group")
    fig.show()
```

# Bring-Your-Own-Response Evaluation for RAG: Reference-Free (without Golden Answer)

Perform bring-your-own-response evaluation by assessing the generated answer's quality based on the context retrieved. It does not compare with golden answer.

### Prepare your dataset

To evaluate the RAG generated answers, the evaluation dataset is required to contain the following fields:

* Prompt: The user supplied prompt consisting of the User Question and the RAG Retrieved Context
* Response: The RAG Generated Answer

Your dataset must include a minimum of one evaluation example. We recommend around 100 examples to ensure high-quality aggregated metrics and statistically significant results.


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

generated_answers_by_rag_a = [
    "frontal lobe and the parietal lobe",
    "The Roman Senate was filled with exuberance due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Syunik and Vayots Dzor.",
]

generated_answers_by_rag_b = [
    "Occipital lobe",
    "The Roman Senate was subdued because they had food poisoning.",
    "The Galactic Empire commanded the state of Utah.",
]

eval_dataset_rag_a = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + item
            for question, item in zip(questions, retrieved_contexts)
        ],
        "response": generated_answers_by_rag_a,
    }
)

eval_dataset_rag_b = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + item
            for question, item in zip(questions, retrieved_contexts)
        ],
        "response": generated_answers_by_rag_b,
    }
)

eval_dataset_rag_a
```

### Select and create metrics


You can run evaluation for just one metric, or a combination of metrics. For this example, we select a few RAG-related predefined metrics, and create a few of our own custom metrics.

#### Explore predefined metrics


```
# See all the available metric examples
MetricPromptTemplateExamples.list_example_metric_names()
```


```
# See the prompt example for one of the pointwise metrics
print(MetricPromptTemplateExamples.get_prompt_template("question_answering_quality"))
```

#### Create custom metrics


```
relevance_prompt_template = """
You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria.

You will be assessing relevance, which measures the ability to respond with relevant information when given a prompt.

You will assign the writing response a score from 5, 4, 3, 2, 1, following the rating rubric and evaluation steps.

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
### Prompt
{prompt}

## AI-generated Response
{response}
"""
```


```
helpfulness_prompt_template = """
You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria.

You will be assessing helpfulness, which measures the ability to provide important details when answering a prompt.

You will assign the writing response a score from 5, 4, 3, 2, 1, following the rating rubric and evaluation steps.

## Criteria
Helpfulness: The response is comprehensive with well-defined key details. The user would feel very satisfied with the content in a good response.

## Rating Rubric
5 (completely helpful): Response is useful and very comprehensive with well-defined key details to address the needs in the instruction and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.
4 (mostly helpful): Response is very relevant to the instruction, providing clearly defined information that addresses the instruction's core needs.  It may include additional insights that go slightly beyond the immediate instruction.  The user would feel quite satisfied with the content in the response.
3 (somewhat helpful): Response is relevant to the instruction and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.
2 (somewhat unhelpful): Response is minimally relevant to the instruction and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.
1 (unhelpful): Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.

## Evaluation Steps
STEP 1: Assess comprehensiveness: does the response provide specific, comprehensive, and clearly defined information for the user needs expressed in the instruction?
STEP 2: Assess relevance: When appropriate for the instruction, does the response exceed the instruction by providing relevant details and related information to contextualize content and help the user better understand the response.
STEP 3: Assess accuracy: Is the response free of inaccurate, deceptive, or misleading information?
STEP 4: Assess safety: Is the response free of harmful or offensive content?

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.


# User Inputs and AI-generated Response
## User Inputs
### Prompt
{prompt}

## AI-generated Response
{response}
"""
```


```
relevance = PointwiseMetric(
    metric="relevance",
    metric_prompt_template=relevance_prompt_template,
)

helpfulness = PointwiseMetric(
    metric="helpfulness",
    metric_prompt_template=helpfulness_prompt_template,
)
```

### Run evaluation with your dataset


```
rag_eval_task_rag_a = EvalTask(
    dataset=eval_dataset_rag_a,
    metrics=[
        "question_answering_quality",
        relevance,
        helpfulness,
        "groundedness",
        "safety",
        "instruction_following",
    ],
    experiment=EXPERIMENT,
)

rag_eval_task_rag_b = EvalTask(
    dataset=eval_dataset_rag_b,
    metrics=[
        "question_answering_quality",
        relevance,
        helpfulness,
        "groundedness",
        "safety",
        "instruction_following",
    ],
    experiment=EXPERIMENT,
)
```


```
result_rag_a = rag_eval_task_rag_a.evaluate()
result_rag_b = rag_eval_task_rag_b.evaluate()
```

### Display evaluation results

#### View summary results

If you want to have an overall view of all the metrics from individual model's evaluation result in one table, you can use the `display_eval_report()` helper function.


```
display_eval_report(
    (
        "Model A Eval Result",
        result_rag_a.summary_metrics,
        result_rag_a.metrics_table,
    )
)
```


```
display_eval_report(
    (
        "Model B Eval Result",
        result_rag_b.summary_metrics,
        result_rag_b.metrics_table,
    )
)
```

#### Visualize evaluation results


```
eval_results = []
eval_results.append(
    ("Model A", result_rag_a.summary_metrics, result_rag_a.metrics_table)
)
eval_results.append(
    ("Model B", result_rag_b.summary_metrics, result_rag_b.metrics_table)
)
```


```
plot_radar_plot(
    eval_results,
    metrics=[
        f"{metric}/mean"
        # Edit your list of metrics here if you used other metrics in evaluation.
        for metric in [
            "question_answering_quality",
            "safety",
            "groundedness",
            "instruction_following",
            "relevance",
            "helpfulness",
        ]
    ],
)
```


```
plot_bar_plot(
    eval_results,
    metrics=[
        f"{metric}/mean"
        for metric in [
            "question_answering_quality",
            "safety",
            "groundedness",
            "instruction_following",
            "relevance",
            "helpfulness",
        ]
    ],
)
```

#### View detailed explanation for an individual instance

If you need to delve into the individual result's detailed explanations on why a score is assigned and how confident the model is for each model-based metric, you can use the `display_explanations()` helper function. For example, you can set `n=2` to display explanation of the 2nd instance result as follows:


```
display_explanations(result_rag_a.metrics_table, n=2)
```

You can also focus on one or a few metrics as follows.


```
display_explanations(result_rag_b.metrics_table, metrics=["groundedness"])
```

# Bring-Your-Own-Response Evaluation for RAG: Referenced (with Golden Answer)

Perform bring-your-own-response evaluation by assessing the generated answer's quality based on the context retrieved and the golden answer provided in the reference.

### Prepare your dataset

To evaluate the RAG generated answers, the evaluation dataset is required to contain the following fields:

* Prompt: The user supplied prompt consisting of the User Question and the RAG Retrieved Context
* Response: The RAG Generated Answer
* Reference: The Golden Answer groundtruth to compare model response to


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

generated_answers_by_rag_a = [
    "frontal lobe and the parietal lobe",
    "The Roman Senate was filled with exuberance due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Syunik and Vayots Dzor.",
]

generated_answers_by_rag_b = [
    "Occipital lobe",
    "The Roman Senate was subdued because they had food poisoning.",
    "The Galactic Empire commanded the state of Utah.",
]

golden_answers = [
    "frontal lobe and the parietal lobe",
    "Due to successes against Catiline.",
    "The Hasan-Jalalians commanded the area of Artsakh and Utik.",
]

referenced_eval_dataset_rag_a = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + item
            for question, item in zip(questions, retrieved_contexts)
        ],
        "response": generated_answers_by_rag_a,
        "reference": golden_answers,
    }
)

referenced_eval_dataset_rag_b = pd.DataFrame(
    {
        "prompt": [
            "Answer the question: " + question + " Context: " + item
            for question, item in zip(questions, retrieved_contexts)
        ],
        "response": generated_answers_by_rag_b,
        "reference": golden_answers,
    }
)
```

### Create custom metrics


Create a custom metric to compare model response to the golden answer.


```
question_answering_correctness_prompt_template = """
You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria.

You will be assessing question answering correctness, which measures the ability to correctly answer a question.

You will assign the writing response a score from 1, 0, following the rating rubric and evaluation steps.

### Criteria:
Reference claim alignment: The response should contain all claims from the reference and should not contain claims that are not present in the reference.

### Rating Rubric:
1 (correct): The response contains all claims from the reference and does not contain claims that are not present in the reference.
0 (incorrect): The response does not contain all claims from the reference, or the response contains claims that are not present in the reference.

### Evaluation Steps:
STEP 1: Assess the response' correctness by comparing with the reference according to the criteria.
STEP 2: Score based on the rubrics.

Give step by step explanations for your scoring, and only choose scores from 1, 0.


# User Inputs and AI-generated Response
## User Inputs
### Prompt
{prompt}

## Reference
{reference}

## AI-generated Response
{response}

"""
```


```
question_answering_correctness = PointwiseMetric(
    metric="question_answering_correctness",
    metric_prompt_template=question_answering_correctness_prompt_template,
)
```

### Run evaluation with your dataset


```
referenced_answer_eval_task_rag_a = EvalTask(
    dataset=referenced_eval_dataset_rag_a,
    metrics=[
        question_answering_correctness,
        "rouge",
        "bleu",
        "exact_match",
    ],
    experiment=EXPERIMENT,
)

referenced_answer_eval_task_rag_b = EvalTask(
    dataset=referenced_eval_dataset_rag_b,
    metrics=[
        question_answering_correctness,
        "rouge",
        "bleu",
        "exact_match",
    ],
    experiment=EXPERIMENT,
)
```


```
referenced_result_rag_a = referenced_answer_eval_task_rag_a.evaluate()
referenced_result_rag_b = referenced_answer_eval_task_rag_b.evaluate()
```

### Display evaluation results

#### View summary results

If you want to have an overall view of all the metrics evaluation result in one table, you can use the `display_eval_report()` helper function.


```
display_eval_report(
    (
        "Model A Eval Result",
        referenced_result_rag_a.summary_metrics,
        referenced_result_rag_a.metrics_table,
    )
)
display_eval_report(
    (
        "Model B Eval Result",
        referenced_result_rag_b.summary_metrics,
        referenced_result_rag_b.metrics_table,
    )
)
```

#### Visualize evaluation results



```
referenced_eval_results = []
referenced_eval_results.append(
    (
        "Model A",
        referenced_result_rag_a.summary_metrics,
        referenced_result_rag_a.metrics_table,
    )
)
referenced_eval_results.append(
    (
        "Model B",
        referenced_result_rag_b.summary_metrics,
        referenced_result_rag_b.metrics_table,
    )
)
```


```
plot_radar_plot(
    referenced_eval_results,
    max_score=1,
    metrics=[
        f"{metric}/mean"
        # Edit your list of metrics here if you used other metrics in evaluation.
        for metric in [
            "question_answering_correctness",
            "rouge",
            "bleu",
            "exact_match",
        ]
    ],
)
```


```
plot_bar_plot(
    referenced_eval_results,
    metrics=[
        f"{metric}/mean"
        for metric in ["question_answering_correctness", "rouge", "bleu", "exact_match"]
    ],
)
```

#### View detailed explanation for an individual instance

If you need to delve into the individual result's detailed explanations on why a score is assigned and how confident the model is for each model-based metric, you can use the `display_explanations()` helper function. For example, you can set `n=2` to display explanation of the 2nd instance result as follows:


```
display_explanations(referenced_result_rag_a.metrics_table, n=2)
```


```
display_explanations(
    referenced_result_rag_a.metrics_table, metrics=["question_answering_correctness"]
)
```
