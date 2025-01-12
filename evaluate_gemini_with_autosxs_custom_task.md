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

# Evaluate Gemini with AutoSxS using custom task

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/legacy/evaluate_gemini_with_autosxs_custom_task.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Flegacy%2Fevaluate_gemini_with_autosxs_custom_task.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/legacy/evaluate_gemini_with_autosxs_custom_task.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/legacy/evaluate_gemini_with_autosxs_custom_task.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>    
</table>

| | |
|-|-|
| Author(s) | [Ivan Nardini](https://github.com/inardini) |

## Overview

This notebook demonstrates how to use Vertex AI automatic side-by-side (AutoSxS) with custom task (translation) to evaluate the performance between Gemini on Vertex AI and PaLM API in a media newspaper company for translation scenario.

In this tutorial, you will perform the following steps to evaluate models on a custom task:

1. Fetch an evaluation dataset, containing context and pre-generated Gemini translations. We will compare model inference responses during evaluation.
2. Define evaluation criteria and a pointwise response score rubric for the custom task evaluation.
4. Create and run a Vertex AI AutoSxS Pipeline.
5. Explore the output evaluation metrics, containing both prompt-level metrics in the form of a judgements table and aggregate metrics.

Learn more about [Vertex AI AutoSxS Model Evaluation](https://cloud.google.com/vertex-ai/docs/generative-ai/models/side-by-side-eval#autosxs).

## Get started

### Install Vertex AI SDK for Python and other required packages

Install required packages.


```
%pip install --upgrade --user --quiet \
    pandas \
    etils \
    gcsfs \
    fsspec \
    google-cloud-aiplatform \
    google-cloud-pipeline-components==2.13.1
```

### Restart runtime (Colab only)

To use the newly installed packages, you must restart the runtime on Google Colab.


```
import sys

if "google.colab" in sys.modules:
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

Authenticate your environment on Google Colab.


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
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}

# Set the project id
! gcloud config set project {PROJECT_ID}
```


```
REGION = "us-central1"  # @param {type: "string"}
```


```
BUCKET_NAME = f"your-bucket-name-{PROJECT_ID}-unique"  # @param {type:"string"}

BUCKET_URI = f"gs://{BUCKET_NAME}"

! gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}
```

### Set tutorial folder

Set a folder to collect data and any tutorial artifacts.


```
from pathlib import Path as path

root_path = path.cwd()
tutorial_path = root_path / "tutorial"
pipeline_path = tutorial_path / "pipeline"

pipeline_path.mkdir(parents=True, exist_ok=True)
```

### Import libraries

Import the required libraries.


```
import random
import string

from IPython.display import HTML, display

# General
from etils import epath
from google.cloud import aiplatform
from google_cloud_pipeline_components.preview import model_evaluation
from kfp import compiler
import pandas as pd

# Model Eval
import vertexai
```

### Set constants

Set tutorial variables.


```
WORKSPACE_BUCKET_PATH = epath.Path(BUCKET_URI) / "evaluate-gemini-translation"
SOURCE_EVALUATION_DATASET_URI = (
    "gs://github-repo/evaluate-gemini-autosxs-custom-task/eval_dataset.jsonl"
)
INSTRUCTION = "You are an expert translator of a famous newspaper media company. Translate the following email in italian"
EVALUATION_DATASET_URI = str(
    WORKSPACE_BUCKET_PATH / "data" / "evaluation_translate_dataset.jsonl"
)
FEW_SHOT_EXAMPLES_DATASET_URI = str(
    WORKSPACE_BUCKET_PATH / "data" / "evaluation_translate_few_shots_dataset.jsonl"
)
PIPELINE_ROOT_URI = str(WORKSPACE_BUCKET_PATH / "pipeline")
```

### Initialize Vertex AI SDK for Python

Initiate a session with Vertex AI SDK for Python.


```
vertexai.init(project=PROJECT_ID, location=REGION)
```

### Helpers

Initialize helper functions required in the tutorial.


```
def print_content(df: pd.DataFrame, columns: list, n: int = 2):
    """Prints specified text columns"""

    # Set the style
    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"

    # Prepare dataset
    selected_df = df[columns].copy()
    selected_df = selected_df.sample(n=n)

    # Iterate through each row
    for i, row in selected_df.iterrows():
        for col_name in columns:
            display(
                HTML(f"<h2>{col_name}:</h2> <div style='{style}'>{row[col_name]}</div>")
            )
        display(HTML("<hr>"))


def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def print_autosxs_judgments(df: pd.DataFrame, n: int = 3):
    base_style = (
        "white-space: pre-wrap; width: 800px; overflow-x: auto; font-size: 16px;"
    )
    header_style = "white-space: pre-wrap; width: 800px; overflow-x: auto; font-size: 24px; font-weight: bold;"

    printed_samples = 0
    for _, row in df.iterrows():
        if row["confidence"] > 0.5:
            for field in df.columns:
                display(
                    HTML(f"<div style='{header_style}'>{field.capitalize()}:</div>")
                )
                display(HTML("<br>"))
                if field != "confidence":
                    display(HTML(f"<div style='{base_style}'>{row[field]}</div>"))
                else:
                    display(HTML(f"<div style='{base_style}'>{row[field]:.2f}</div>"))
                display(HTML("<br>"))
            display(HTML("<hr>"))
            printed_samples += 1
        if printed_samples >= n:
            break


def print_aggregated_metrics(scores: dict):
    """Print AutoSxS aggregated metrics with enhanced formatting."""

    score_a = round(scores["autosxs_model_a_win_rate"] * 100)
    display(
        HTML(
            f"""
    <div style='font-size: 18px;'>
        AutoSxS Autorater prefers <span style='font-weight: bold;'>{score_a}%</span> of time Model A over Model B
    </div>
    """
        )
    )
```

## Evaluating Gemini on Vertex AI in translation task

### Load the evaluation dataset

To conduct an evaluation, you require an evaluation dataset comprising a set of inference prompts along with their corresponding model responses.  For more details on this see [here](https://cloud.google.com/vertex-ai/generative-ai/docs/models/side-by-side-eval#prep-eval-dataset)

In the current scenario, you are loading a pre-built evaluation dataset containing original emails (in the "content" column) and their Italian translations (in the "model_a_response" column) generated using Gemini on Vertex AI.

However, the evaluation dataset assumes the utilization of AutoSxS to produce Italian translations from model B, specifically the `text-bison@002` model. Therefore, adjustments or adaptations may be necessary to align the evaluation dataset with the actual translation method employed in your evaluation process.


```
evaluation_translate_df = pd.read_json(
    SOURCE_EVALUATION_DATASET_URI, orient="records", lines=True
)
```


```
evaluation_translate_df.head()
```


```
evaluation_translate_df.to_json(EVALUATION_DATASET_URI, orient="records", lines=True)
```

### Define custom evaluation criteria

Before initiating the task, establish evaluation criteria to guide the model's assessment of responses. The `**evaluation_criteria**` is a list of such criteria, each represented as a dictionary with the following keys:

*   **`name` (str):** The criterion's name.
*   **`definition` (str):** The criterion's definition.
*   **`criteria_steps` (Optional[List[str]]):** A step-by-step guide for the AutoRater on assessing criterion fulfillment.

For instance, in a translation task, you might employ various criteria like accuracy, fluency, use of loanwords, etc.


```
EVALUATION_CRITERIA = [
    {
        "name": "Overall Correctness",
        "definition": "The translation accurately conveys the meaning, nuances, and style of the content text in the italian language, while maintaining fluency and appropriateness for the intended audience and purpose.",
        "evaluation_steps": [
            "Compare the responses containing translated emails.",
            "Assess which translated email accurately reflects the meaning of the content text in the italian language.",
            "Consider which translated email maintains the original style and tone.",
            "Ensure the translated email is fluent and natural-sounding in the italian language.",
        ],
    },
    {
        "name": "Mistranslation",
        "definition": "The translation contains errors where the meaning of the content text is not accurately conveyed in the italian language.",
        "evaluation_steps": [
            "Compare the responses containing translated emails.",
            "Identify any discrepancies in meaning between translated emails and the content text.",
            "Determine if the discrepancies are significant enough to be considered mistranslations.",
            "Consider if cultural references or nuances are lost or misinterpreted in the translation.",
        ],
    },
    {
        "name": "Fluency",
        "definition": "The translation reads smoothly and naturally in the italian language, adhering to its grammatical structures and idiomatic expressions.",
        "evaluation_steps": [
            "Go through the translated emails, paying attention to its flow.",
            "Assess if the sentence structures sound natural and idiomatic in the italian language.",
            "Identify any awkward phrasing, grammatical errors, or unnatural word choices.",
            "Consider if the translation would be easily understood by a native speaker of the italian language.",
        ],
    },
    {
        "name": "Grammatical correctness",
        "definition": "The translation adheres to the grammatical rules of the italian language, including syntax, morphology, and word choice.",
        "evaluation_steps": [
            "Analyze translated emails for grammatical errors such as incorrect verb conjugations, agreement issues, or misuse of articles.",
            "Assess if the word order follows the conventions of the italian language.",
            "Ensure punctuation is used correctly.",
        ],
    },
    {
        "name": "Loanwording",
        "definition": "The translation leaves individual words or phrases in the italian language untranslated.",
    },
]
```

### Define Score Rubric

After defining the evaluation criteria, you need to establish a scoring rubric to assess model responses point by point, ensuring alignment with those criteria. The **`score_rubric`** is a dictionary that assigns a numerical score to a specific level of criteria fulfillment. Scores must be consecutive integers within the range of 1 to 5.

For this translation task, you will evaluate translations using a 5-point scoring rubric where 5 denotes the highest quality translation (most accurate, fluent, and faithful), and 1 indicates the lowest quality (least accurate, fluent, and faithful).


```
SCORE_RUBRIC = {
    "5": "Response is an excellent translation that accurately conveys the meaning and nuances of the content text while maintaining fluency and naturalness in the italian language. It adheres to grammar and style conventions, ensuring the content is appropriate and relevant.",
    "4": "Response is a good translation that accurately conveys the meaning of the content text with minor issues in fluency, style or cultural appropriateness.",
    "3": "Response is an average translation that conveys the general meaning of the content text but may contain some inaccuracies, awkward phrasing, or stylistic inconsistencies.",
    "2": "Response is a below average translation that contains significant inaccuracies, is not fluent or natural-sounding, and may not be appropriate.",
    "1": "Response is a poor translation that fails to convey the meaning of the content text, is full of errors, and is not suitable.",
}
```

### Define few-shot examples

To refine AutoRater's behavior for complex tasks, we'll create a custom few-shot examples dataset. Here's the process:

1. **Craft concise examples:** Develop simple yet representative examples that reflect the task's nuances.
2. **Align with the rubric:** Assign calibration scores to each example based on your predefined scoring rubric. These scores will guide the AutoRater's evaluation.
3. **Store in Cloud Storage:** Save your few-shot examples dataset as a JSON file in Cloud Storage for easy access and integration.

In this translation scenario, you have only one example with the two Italian translations (response_a and response_b) of an English email, along with scores and an explanation indicating that the reason why response_b is slightly superior of response_a due to its more natural language flow and idiomatic expressions.

While a single example is shown below, it is strongly recommended providing numerous examples spanning the full range of scores and criteria to ensure comprehensive tuning.

**Important Note:** Remember that more examples consume more tokens sent to the AutoRater, impacting your available context length. For instance, if your model has an 8000-token context limit and your few-shot examples total 1000 tokens, your context length will be reduced to 7000.


```
few_shot_examples_df = pd.DataFrame.from_records(
    [
        {
            "business_unit": "digital/online",
            "content": """
                  Subject: Urgent: Website Downtime Impact on Subscription Revenue - Action Required

                  Hi Digital Team,

                  As you know, our website experienced a significant outage last night, lasting approximately 3 hours between 10 PM and 1 AM. We're still gathering information on the cause, but early indications point to a server overload related to a surge in traffic during the live broadcast of the "The Voice" finale.

                  This outage has had a direct impact on our subscription revenue, as users were unable to access premium content during this period. Our analytics team has preliminary data showing a 25% drop in new subscription sign-ups during the outage window, compared to the average for the same time period over the last month. This represents an estimated loss of ~$15,000 in revenue for the night.

                  **Immediate Actions Required:**

                  * **Identify the Root Cause:** Work with the IT team to understand the exact cause of the server overload and implement solutions to prevent future occurrences.
                  * **Assess the Impact:** Continue to monitor the situation closely and gather complete data on the revenue impact from both new subscriptions and existing subscriber engagement.
                  * **Customer Communication:** We need to acknowledge the issue with our users and provide an explanation. We should consider offering a temporary discount on subscriptions as a goodwill gesture to those affected.
                  * **Performance Optimization:** Evaluate our website's scalability and infrastructure to ensure we can handle future traffic spikes, especially during major events like "The Voice".

                  Please prioritize these actions and provide a status report by end-of-day today. I will be available to discuss this further in a meeting tomorrow at 10 AM.

                  Thanks,

                  Ivan
                  Expert Assistant
                  Cymbal Media
        """,
            "response_a": """
                  Oggetto: Urgente: Interruzione del sito web e impatto sulle entrate da abbonamento - Azioni necessarie

                  Ciao Team Digitale,

                  Come sapete, il nostro sito web ha subito un'interruzione significativa la scorsa notte, durata circa 3 ore tra le 22:00 e l'1:00. Stiamo ancora raccogliendo informazioni sulla causa, ma le prime indicazioni puntano a un sovraccarico del server legato a un picco di traffico durante la diretta della finale di "The Voice".

                  Questa interruzione ha avuto un impatto diretto sulle nostre entrate da abbonamento, poiché gli utenti non hanno potuto accedere ai contenuti premium durante questo periodo. Il nostro team di analisi ha dati preliminari che mostrano un calo del 25% nelle nuove iscrizioni durante la finestra di interruzione, rispetto alla media dello stesso periodo negli ultimi mesi. Questo rappresenta una perdita stimata di circa $ 15.000 di entrate per la notte.

                  **Azioni immediate richieste:**

                  * **Identificare la causa principale:** Collaborare con il team IT per comprendere la causa esatta del sovraccarico del server e implementare soluzioni per prevenire future occorrenze.
                  * **Valutare l'impatto:** Continuare a monitorare la situazione attentamente e raccogliere dati completi sull'impatto sulle entrate sia dalle nuove iscrizioni che dal coinvolgimento degli abbonati esistenti.
                  * **Comunicazione con i clienti:** Dobbiamo riconoscere il problema con i nostri utenti e fornire una spiegazione. Dovremmo prendere in considerazione la possibilità di offrire uno sconto temporaneo sugli abbonamenti come gesto di buona volontà a coloro che sono stati colpiti.
                  * **Ottimizzazione delle prestazioni:** Valutare la scalabilità e l'infrastruttura del nostro sito web per garantire che possiamo gestire i futuri picchi di traffico, soprattutto durante eventi importanti come "The Voice".

                  Vi prego di dare la priorità a queste azioni e di fornire un rapporto sullo stato entro la fine della giornata odierna. Sarò disponibile a discutere ulteriormente di questo argomento in una riunione domani alle 10:00.

                  Grazie,

                  Ivan
                  Assistente Esperto
                  Cymbal Media

        """,
            "response_b": """
                  Oggetto: Urgente: Impatto del downtime del sito web sulle entrate degli abbonamenti - Azioni richieste

                  Ciao Team Digitale,

                  Come sapete, il nostro sito web ha subito un'interruzione significativa la scorsa notte, con una durata di circa 3 ore tra le 22:00 e l'01:00. Stiamo ancora raccogliendo informazioni sulla causa, ma le prime indicazioni puntano a un sovraccarico del server correlato a un'impennata di traffico durante la trasmissione in diretta della finale di "The Voice".

                  Questa interruzione ha avuto un impatto diretto sulle entrate dei nostri abbonamenti, poiché gli utenti non sono stati in grado di accedere ai contenuti premium durante questo periodo. Il nostro team di analisi ha dati preliminari che mostrano un calo del 25% nelle nuove iscrizioni agli abbonamenti durante la finestra di interruzione, rispetto alla media dello stesso periodo nell'ultimo mese. Questo rappresenta una perdita stimata di ~$15.000 di entrate per la notte.

                  **Azioni immediate richieste:**

                  * **Identificare la causa principale:** Collaborare con il team IT per comprendere la causa esatta del sovraccarico del server e implementare soluzioni per prevenire futuri eventi simili.
                  * **Valutare l'impatto:** Continuare a monitorare la situazione da vicino e raccogliere dati completi sull'impatto sui ricavi sia dalle nuove iscrizioni che dal coinvolgimento degli abbonati esistenti.
                  * **Comunicazione con i clienti:** Dobbiamo riconoscere il problema con i nostri utenti e fornire una spiegazione. Dovremmo considerare di offrire uno sconto temporaneo sugli abbonamenti come gesto di buona volontà a coloro che sono stati colpiti.
                  * **Ottimizzazione delle prestazioni:** Valutare la scalabilità e l'infrastruttura del nostro sito web per garantire che possiamo gestire futuri picchi di traffico, specialmente durante eventi importanti come "The Voice".

                  Vi prego di dare la priorità a queste azioni e di fornire un rapporto sullo stato di avanzamento entro la fine della giornata di oggi. Sarò disponibile a discuterne ulteriormente in una riunione domani alle 10:00.

                  Grazie,

                  Ivan
                  Assistente Esperto
                  Cymbal Media
        """,
            "response_a_score": 2,
            "response_b_score": 4,
            "winner": "B",
            "explanation": "Both A and B provide accurate and fluent translations of the original English email. However, B demonstrates a slightly better grasp of natural language flow and idiomatic expressions, making it the slightly better translation overall.",
        }
    ]
)
```


```
few_shot_examples_df.to_json(
    FEW_SHOT_EXAMPLES_DATASET_URI, orient="records", lines=True
)
```

### Create and run AutoSxS job

To run AutoSxS, first you need to define parameters for:

* **Evaluation data:** Path to your dataset (`evaluation_dataset`), task type (`task`), and unique identifiers (`id_columns`).
* **Autorater guidance:**  Parameters for the autorater's prompt (`autorater_prompt_parameters`) and custom task details (`experimental_args`).
* **Model details:** Either model resource name (`model_a`) and parameters (`model_a_prompt_parameters`, `model_a_parameters`) for batch prediction, or prediction column name (`response_column_a`) if using pre-generated predictions.

Refer to the documentation [here](https://google-cloud-pipeline-components.readthedocs.io/en/google-cloud-pipeline-components-2.13.0/api/preview/model_evaluation.html#preview.model_evaluation.autosxs_pipeline) for detailed configuration options and additional parameters for features like exporting judgments and checking alignment with human preferences.


```
display_name = f"autosxs-custom-evaluate-{generate_uuid()}"

PIPELINE_PARAMETERS = {
    "evaluation_dataset": EVALUATION_DATASET_URI,
    "id_columns": ["content"],
    "task": "custom@001",
    "response_column_a": "model_a_response",
    "response_column_b": "model_b_response",
    "autorater_prompt_parameters": {
        "instruction": {"template": (INSTRUCTION)},
        "content": "content",
    },
    "experimental_args": {
        "custom_task_definition": {
            "evaluation_criteria": EVALUATION_CRITERIA,
            "autorater_prompt_parameter_keys": [
                "instruction",
                "content",
            ],
            "few_shot_examples_config": {
                "dataset": FEW_SHOT_EXAMPLES_DATASET_URI,
                "autorater_prompt_parameters": {
                    "instruction": {"template": (INSTRUCTION)},
                    "content": "content",
                },
                "response_a_column": "response_a",
                "response_b_column": "response_b",
                "choice_column": "winner",
                "explanation_column": "explanation",
                "response_a_score_column": "response_a_score",
                "response_b_score_column": "response_b_score",
            },
            "score_rubric": SCORE_RUBRIC,
        }
    },
}
```

Then, you compile the AutoSxS pipeline locally.


```
template_uri = str(pipeline_path / "pipeline.yaml")
compiler.Compiler().compile(
    pipeline_func=model_evaluation.autosxs_pipeline,
    package_path=template_uri,
)
```

Finally, you submit the evaluation job to initiate a Vertex AI Pipeline job, which can be monitored through the Vertex AI UI. This process will take approximately 10 minutes.


```
pipeline_job = aiplatform.PipelineJob(
    job_id=display_name,
    display_name=display_name,
    template_path=template_uri,
    pipeline_root=PIPELINE_ROOT_URI,
    parameter_values=PIPELINE_PARAMETERS,
    enable_caching=False,
)
pipeline_job.run()
```

### Get the judgments and AutoSxS win-rate metrics

You can access the evaluation results in the Vertex AI Pipelines by examining the artifacts generated by the AutoSxS pipeline. These include:

- `judgments table` provides example-level metrics, including inference prompts, model responses, autorater decisions, rating explanations, and confidence scores.
- `aggregate metrics` which offer an overview of model performance, such as the AutoRater model A win rate, which indicates the percentage of times the autorater preferred model A's response.

By analyzing these outputs, you can gain a comprehensive understanding of both individual example performance and overall model comparison.

#### Judgments


```
for details in pipeline_job.task_details:
    if details.task_name == "online-evaluation-pairwise":
        break

# Judgments
judgments_uri = details.outputs["judgments"].artifacts[0].uri
judgments_df = pd.read_json(judgments_uri, lines=True)
```


```
print_autosxs_judgments(judgments_df)
```

#### Aggregate metrics


```
for details in pipeline_job.task_details:
    if details.task_name == "model-evaluation-text-generation-pairwise":
        break
win_rate_metrics = details.outputs["autosxs_metrics"].artifacts[0].metadata
```


```
print_aggregated_metrics(win_rate_metrics)
```

## Cleaning up


```
# Delete Model Evaluation pipeline run
delete_pipeline = False
if delete_pipeline:
    job.delete()

# Delete Cloud Storage objects that were created
delete_bucket = False
if delete_bucket:
    ! gsutil -m rm -r $BUCKET_URI
```
