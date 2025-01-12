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

# Evaluating multimodal task

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_multimodal_task_image.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fevaluate_multimodal_task_image.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/evaluate_multimodal_task_image.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_multimodal_task_image.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) |[Ivan Nardini](https://github.com/inardini) |

## Overview

This notebook shows how to use Vertex AI Python SDK for Gen AI Evaluation Service for evaluating multimodal task with your locally-defined `CustomMetric`, and use your own autorater model to perform model-based metric evaluation.

<div class="alert alert-block alert-warning">
<b>⚠️ This notebook shows an experiment approach to evaluate multimodal task using Vertex AI Python SDK for Gen AI Evaluation Service. The result of the evaluation depends on the autorater's capabilities of handling multimodal inputs with evaluation criteria. ⚠️</b>
</div>

## Getting Started

### Install Vertex AI SDK for Gen AI Evaluation Service


```
%pip install -U -q google-cloud-aiplatform[evaluation]
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

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Evaluate your multimodal use case with custom metric

Imagine you have a customer insurance app which provides several services including live incident support using an LLM. Given a conversation about a car accident and an image of the damaged car, you want to evaluate the coherence of the generated responses.


### Import libraries


```
# General
from IPython.display import HTML, Markdown, display
from vertexai.evaluation import CustomMetric, EvalTask
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
```

### Library settings


```
# @title
import json
import logging
import warnings

import pandas as pd

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# pd.set_option('display.max_colwidth', None)
```

### Helpers


```
def display_eval_result(
    eval_result: dict | object,
    title: str | None = None,
    metrics: list[str] | None = None,
) -> None:
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


def display_explanations(
    eval_result: dict | object, metrics: list[str] | None = None, n: int = 1
) -> None:
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
```

### Load your evaluation dataset

Depending on your task and metrics you are planning to calculate, you need to collect your evaluation dataset.

In this scenario, you have a `context` column which contains both the conversation and the location of the image for a specific car accident scenario.



```
instruction = """
You are an insurance agent specializing in car accident assessments.
You will be provided with a conversation about a car accident and an image of the damaged car.
Your task is to analyze the image and identify the primary type of damage visible.
Use the conversation for context, but prioritize the visual evidence from the image.

Categorize the primary damage as one of the following:

* bumper
* engine_compartment
* hood
* lateral
* windshield

If the image is unclear or the damage is not visible, respond with "Unable to determine damage type from the provided image."
If the primary damage is something other than the listed categories, respond with "Damage type not listed in available categories."

Conversation:
{conversation}

Image:
{image_of_car_accident}

Provide your assessment of the primary damage type based on the image.
"""

context = [
    {
        "conversation": '''AI insurance app: "Hello, I'm the AI assistant for your car insurance. It looks like you've been in an accident. Could you please tell me what happened?" App user: "Yeah, I was just rear-ended while waiting at a red light." AI insurance app: "I'm sorry to hear that.  Could you take some pictures of the damage to your vehicle, including the license plate of the other car if possible?" App user:  (uploads the provided image) "Here's the damage to my bumper." AI insurance app: "Thank you.  Can you describe any injuries to yourself or any passengers?" App user: "No, thankfully everyone is okay. Just a bit shaken up." AI insurance app: "That's good to hear. I've created a claim based on the information and photo you provided. A representative will be in touch with you shortly to gather further details and discuss the next steps." App user: "Okay, thank you." AI insurance app: "You're welcome. Please don't hesitate to contact us if you have any questions."''',
        "image_of_car_accident": "gs://cloud-samples-data/generative-ai/evaluation/use_cases/car_assessment/bumper.jpg",
    },
    {
        "conversation": '''AI insurance app: "Hi there! I'm Amelia, your AI assistant for [Insurance company name]. I see you've been in an accident. I'm so sorry to hear that. Are you okay?" Driver: "I'm a little shaken up, but I'm okay. My car isn't so lucky, though." AI insurance app: "Oh no, I'm so sorry to hear that. Can you tell me what happened?" Driver: "I was stopped at a red light when I was rear-ended by another car. The damage to my car is pretty bad." AI insurance app: "I understand. Can you take some pictures of the damage to your car, including the license plate of the other car if possible?" Driver: "Sure, here you go." (uploads the provided image) AI insurance app: "Thank you. Is there anyone else involved in the accident?" Driver: "No, just me and the driver of the other car." AI insurance app: "Okay. Do you need medical attention?" Driver: "No, I'm fine." AI insurance app: "I'm glad to hear that. I've created a claim based on the information and photo you provided. A representative will be in touch with you shortly to gather further details and discuss the next steps." Driver: "Okay, thank you." AI insurance app: "You're welcome. Please don't hesitate to contact us if you have any questions."''',
        "image_of_car_accident": "gs://cloud-samples-data/generative-ai/evaluation/use_cases/car_assessment/engine_compartment.jpg",
    },
    {
        "conversation": '''**AI insurance app:** "Hello, it appears you've been in an accident. Are you alright?" **App user:** "Yes, I'm okay. Just a bit shaken up." **AI insurance app:** "I'm glad to hear you're physically unharmed. Could you please describe what happened?" **App user:** "Someone ran a red light and hit the front of my car." **AI insurance app:** "I understand. To help assess the damage, could you please take some photos of your vehicle, especially the impacted areas? If possible, include a photo of the other vehicle's license plate." **App user:** (uploads the provided image) "Here's the damage to my car." **AI insurance app:** "Thank you for providing that.  Were there any other vehicles involved, or was it just the two cars?" **App user:** "No, it was just us." **AI insurance app:** "Okay. And to confirm, you don't require any medical assistance at this time?" **App user:** "No, I don't think so. Thankfully." **AI insurance app:** "Alright. I've created an accident claim with the information and photos you've provided. One of our representatives will contact you soon to gather more details and guide you through the next steps." **App user:** "Thank you, I appreciate the help." **AI insurance app:** "You're very welcome. Please don't hesitate to reach out through the app if you have any further questions."''',
        "image_of_car_accident": "gs://cloud-samples-data/generative-ai/evaluation/use_cases/car_assessment/hood.jpg",
    },
    {
        "conversation": '''AI insurance app: "Hi there! I'm Amelia, your AI assistant for [Insurance company name]. I see you've been in an accident. I'm so sorry to hear that. Are you okay?" Driver: "I'm a little shaken up, but I'm okay. My car isn't so lucky, though." AI insurance app: "Oh no, I'm so sorry to hear that. Can you tell me what happened?" Driver: "I was stopped at a red light when I was rear-ended by another car. The damage to my car is pretty bad." AI insurance app: "I understand. Can you take some pictures of the damage to your car, including the license plate of the other car if possible?" Driver: "Sure, here you go." (uploads the provided image) AI insurance app: "Thank you. Is there anyone else involved in the accident?" Driver: "No, just me and the driver of the other car." AI insurance app: "Okay. Do you need medical attention?" Driver: "No, I'm fine." AI insurance app: "I'm glad to hear that. I've created a claim based on the information and photo you provided. A representative will be in touch with you shortly to gather further details and discuss the next steps." Driver: "Okay, thank you." AI insurance app: "You're welcome. Please don't hesitate to contact us if you have any questions.""''',
        "image_of_car_accident": "gs://cloud-samples-data/generative-ai/evaluation/use_cases/car_assessment/lateral.jpg",
    },
    {
        "conversation": '''AI insurance app: "Hello. I've received an alert that you may have been involved in an accident. Can you confirm and tell me if you're okay?" App user: "Yes, I was just in an accident. I'm okay, just a little shaken." AI insurance app: "I'm relieved to hear you're not hurt. Can you tell me what happened?" App user: "A rock flew up from a truck in front of me and cracked my windshield." AI insurance app: "I understand.  To assess the damage, could you please take a photo of the damage?" App user: (uploads the provided image) "Here's a photo of the crack." AI insurance app:  "Thank you for providing that.  Were there any other vehicles involved?" App user: "No, just my car." AI insurance app: "Okay. And you didn't sustain any injuries?" App user: "No, thankfully not." AI insurance app: "That's good to hear. I've created a claim for you based on the information and photo you provided. A representative will be in touch shortly to gather more details and guide you through the next steps." App user: "Okay, thank you." AI insurance app: "You're welcome. Please don't hesitate to contact us if you have any questions."''',
        "image_of_car_accident": "gs://cloud-samples-data/generative-ai/evaluation/use_cases/car_assessment/windshield.jpg",
    },
]

generated_response = [
    "bumper",
    "engine_compartment",
    "lateral",
    "lateral",
    "windshield",
]

reference = ["bumper", "engine_compartment", "hood", "lateral", "windshield"]

eval_dataset = pd.DataFrame(
    {
        "instruction": instruction,
        "context": context,
        "response": generated_response,
        "reference": reference,
    }
)
```


```
eval_dataset.head()
```

### Bring your own autorater

Gen AI Evaluation Service allows you to bring any autorater you prefer. In this case, you use `gemini-1.5-pro` with a typical structured output to evaluate your task.


```
def get_autorater_response(metric_prompt: list) -> dict:
    metric_response_schema = {
        "type": "OBJECT",
        "properties": {
            "score": {"type": "NUMBER"},
            "explanation": {"type": "STRING"},
        },
        "required": ["score", "explanation"],
    }

    autorater = GenerativeModel(
        "gemini-1.5-pro",
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=metric_response_schema,
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
    )

    response = autorater.generate_content(metric_prompt)

    response_json = {}

    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if (
            candidate.content
            and candidate.content.parts
            and len(candidate.content.parts) > 0
        ):
            part = candidate.content.parts[0]
            if part.text:
                response_json = json.loads(part.text)

    return response_json
```

### Build your custom metric

To evaluate a multimodal task, you define a custom metric which takes any multimodal content (image, pdf, video and more) and returns a score and an explanation according to some criteria,  rating rubric, and evaluation steps using the defined autorater.

In this scenario, you provide define a coherence metric in this car accident assessment scenario.


```
def custom_coherence_fn(instance):

    conversation = instance["context"]["conversation"]
    image_of_car_accident = instance["context"]["image_of_car_accident"]
    response = instance["response"]

    eval_instruction_template = """

  # Instruction
  You are an insurance agent specializing in evaluating car accident assessments.
  You will be provided with a conversation about a car accident and an image of the damaged car.
  You should first read the conversation and look at the image carefully, and then evaluate the coherence of the generated responses based on the Criteria provided in the Evaluation section below.
  You will assign the response a rating following the Rating Rubric and Evaluation Steps. Give step-by-step explanations for your rating, and only choose ratings from the Rating Rubric.

  # Evaluation
  ## Metric Definition
  You will be assessing coherence, which measures the ability to provide a coherent response based on the conversation and car accident image.

  ## Criteria
  Coherence: It is the quality of being logical and consistent.
  In the context of conversation, it refers to the way that ideas and information are presented in a way that is easy to understand and follow.
  A coherent conversation will have a clear flow and will not jump around from topic to topic.
  The user will also use language that is appropriate for the audience and will avoid making claims that are not supported by evidence.

  ## Rating Rubric
  5: (Perfectly Aligned) The image precisely matches the damage described in the conversation, and the response accurately reflects the damaged car part.
  4: (Highly Aligned) The image generally supports the conversation's description of the damage, and the response is a suitable representation of the affected area.
  3: (Moderately Aligned) The image shows damage that is plausibly related to the accident described, but there might be minor inconsistencies, and the response is broadly relevant but not entirely specific.
  2: (Poorly Aligned)  The image and/or the response have significant inconsistencies with the described accident in the conversation, raising doubts about the claim's validity.
  1: (Misaligned) The image, response, and conversation have major contradictions or  are completely unrelated, making the claim appear illogical or fraudulent.

  ## Evaluation Steps
  STEP 1:  Assess Claim Consistency:  Carefully read the conversation to understand the user's description of the accident and the claimed damage.
  STEP 2:  Analyze Image Relevance: Examine the image to determine if the depicted damage aligns with the user's account. Pay attention to the location and type of damage.
  STEP 3: Evaluate Label Accuracy:  Check if the generated label correctly identifies the damaged car part as described in the conversation and shown in the image.
  STEP 4:  Identify Inconsistencies: Look for any discrepancies between the conversation, image, and label. For example, does the image show damage not mentioned in the conversation, or is the label incorrect for the damaged part?
  STEP 5:  Determine Overall Coherence: Based on the previous steps, assign a coherence score using the 1-5 rubric.  Consider the severity of any inconsistencies and their potential impact on the claim's validity.
  """

    # read image from uri
    image_file = Part.from_uri(image_of_car_accident, "image/jpeg")

    # generate the eval
    evaluation_prompt = [
        eval_instruction_template,
        "CONVERSATION: ",
        conversation,
        "IMAGE: ",
        image_file,
        "GENERATED RESPONSE: ",
        response,
    ]

    evaluation_response = get_autorater_response(evaluation_prompt)
    return {
        "custom_coherence": evaluation_response.get("score", ""),
        "explanation": evaluation_response.get("explanation", ""),
    }
```


```
custom_coherence_metric = CustomMetric(
    name="custom_coherence",
    metric_function=custom_coherence_fn,
)
```

### Run Evaluation

With the autorater, and the custom metric, you run your evaluation.


```
metrics = ["exact_match", custom_coherence_metric]

experiment_name = "eval-multimodal-metric"

eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment=experiment_name,
)
```


```
eval_result = eval_task.evaluate()
```

### Visualize and validate your evaluation results.

Using some helpers, you can both evaluation results (scores, and explanations).


```
display_eval_result(eval_result, title="Evaluation Results")
```


```
display_explanations(eval_result, metrics=["custom_coherence"])
```

## Cleaning up


```
delete_experiment = True

if delete_experiment:

    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    experiment = aiplatform.Experiment(experiment_name)
    experiment.delete()
```
