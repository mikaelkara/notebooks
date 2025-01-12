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

 # Bring-Your-Own-Autorater using `CustomMetric` | Gen AI Evaluation SDK Tutorial


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/bring_your_own_autorater_with_custom_metric.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fbring_your_own_autorater_with_custom_metric.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/bring_your_own_autorater_with_custom_metric.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/bring_your_own_autorater_with_custom_metric.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai) |

## Overview

Use the *Vertex AI Python SDK for Gen AI Evaluation Service* to evaluate generative AI models with your locally-defined `CustomMetric`, and use your own autorater model to perform model-based metric evaluation.


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

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
# General
import inspect
import json
import logging
import warnings

from IPython.display import HTML, Markdown, display
import pandas as pd
import plotly.graph_objects as go

# Main
from vertexai.evaluation import CustomMetric, EvalTask
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)
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

## Using locally-defined `CustomMetric` class


### Documentation



```
print_doc(CustomMetric)
```

    CustomMetric:
    The custom evaluation metric.
    
    A fully-customized CustomMetric that can be used to evaluate a single model
    by defining a metric function for a computation-based metric. The
    CustomMetric is computed on the client-side using the user-defined metric
    function in SDK only, not by the Vertex Gen AI Evaluation Service.
    
      Attributes:
        name: The name of the metric.
        metric_function: The user-defined evaluation function to compute a metric
          score. Must use the dataset row dictionary as the metric function
          input and return per-instance metric result as a dictionary output.
          The metric score must mapped to the name of the CustomMetric as key.
    
    

### Define a `CustomMetric` and "Bring-Your-Own-Autorater"

Define a custom model-based metric and "Bring-Your-Own-Autorater". The custom model-based metric can contain multiple fields such as an autorater-graded score, explanations for the score, and more. The score field must match the name of the metric, other fields can be freeform strings.


```
# Bring-Your-Own-Autorater and define your autorater inference function


def get_autorater_response(metric_prompt: str) -> dict:
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


```
# Define your metric computation function


def linguistic_acceptability_fn(instance: dict) -> dict:
    metric_prompt_template = """
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models. We will provide you with the user prompt and an AI-generated responses.
You should first read the user input carefully for analyzing the task, and then evaluate the quality of the responses based on the Criteria provided in the Evaluation section below.
You will assign the response a rating following the Rating Rubric and Evaluation Steps. Give step by step explanations for your rating, and only choose ratings from the Rating Rubric.


# Evaluation
## Criteria
Appropriate word choice: Words chosen are appropriate and purposeful given their relative context and positioning in the text. Vocabulary demonstrates prompt understanding.
Proper Grammar: The language's grammar rules are correctly followed, including but not limited to sentence structures, verb tenses, subject-verb agreement, proper punctuation, and capitalization.
Reference Alignment: The response is consistent and aligned with the reference.

## Rating Rubric
1: Poor: The writing is riddled with grammatical errors, uses highly inappropriate vocabulary, is completely unrelated to the reference.
2: Unsatisfactory: The writing has significant grammatical errors, uses inappropriate vocabulary, deviates significantly from the reference.
3: Satisfactory: The writing may have minor grammatical errors or use less-appropriate vocabulary, but it aligns reasonably well with the reference.
4: Good: The writing is generally grammatically correct, uses appropriate vocabulary and aligns well with the reference.
5: Excellent: The writing is grammatically correct, uses appropriate vocabulary and aligns perfectly with the reference.

## Evaluation Steps
Step 1: Assess the response in aspects of all criteria provided. Provide assessment according to each criterion.
Step 2: Score based on the rating rubric. Give a brief rationale to explain your evaluation considering each individual criterion.


# User Inputs and AI-generated Response
## User Inputs
### prompt
{prompt}

## AI-generated Response
{response}
"""
    prompt = instance["prompt"]
    response = instance["response"]

    metric_prompt = metric_prompt_template.format(prompt=prompt, response=response)

    response_dict = get_autorater_response(metric_prompt)

    return {
        "linguistic_acceptability": response_dict["score"],
        "explanation": response_dict["explanation"],
    }
```


```
# Define a CustomMetric
linguistic_acceptability_metric = CustomMetric(
    name="linguistic_acceptability",
    metric_function=linguistic_acceptability_fn,
)
```

### Run Evaluation with `CustomMetric`

* The registered custom metrics are computed on the client side, without using Vertex Gen AI Evaluation Service APIs.

* Mixing CustomMetric instances and other types of metrics is supported.

* You can handle rate limiting of your own autorater with Eval SDK's [RateLimiter utils function](https://github.com/googleapis/python-aiplatform/blob/20f2cad85719b4793c2b77bf36c5a0d775b1a825/vertexai/evaluation/utils.py#L104) based on your autorater's capacity and QPS limit.


#### Define an evaluation dataset


```
instruction = "Summarize the following article"

context = [
    "To make a classic spaghetti carbonara, start by bringing a large pot of salted water to a boil. While the water is heating up, cook pancetta or guanciale in a skillet with olive oil over medium heat until it's crispy and golden brown. Once the pancetta is done, remove it from the skillet and set it aside. In the same skillet, whisk together eggs, grated Parmesan cheese, and black pepper to make the sauce. When the pasta is cooked al dente, drain it and immediately toss it in the skillet with the egg mixture, adding a splash of the pasta cooking water to create a creamy sauce.",
    "Preparing a perfect risotto requires patience and attention to detail. Begin by heating butter in a large, heavy-bottomed pot over medium heat. Add finely chopped onions and minced garlic to the pot, and cook until they're soft and translucent, about 5 minutes. Next, add Arborio rice to the pot and cook, stirring constantly, until the grains are coated with the butter and begin to toast slightly. Pour in a splash of white wine and cook until it's absorbed. From there, gradually add hot chicken or vegetable broth to the rice, stirring frequently, until the risotto is creamy and the rice is tender with a slight bite.",
    "For a flavorful grilled steak, start by choosing a well-marbled cut of beef like ribeye or New York strip. Season the steak generously with kosher salt and freshly ground black pepper on both sides, pressing the seasoning into the meat. Preheat a grill to high heat and brush the grates with oil to prevent sticking. Place the seasoned steak on the grill and cook for about 4-5 minutes on each side for medium-rare, or adjust the cooking time to your desired level of doneness. Let the steak rest for a few minutes before slicing against the grain and serving.",
    "Creating a creamy homemade tomato soup is a comforting and simple process. Begin by heating olive oil in a large pot over medium heat. Add diced onions and minced garlic to the pot and cook until they're soft and fragrant. Next, add chopped fresh tomatoes, chicken or vegetable broth, and a sprig of fresh basil to the pot. Simmer the soup for about 20-30 minutes, or until the tomatoes are tender and falling apart. Remove the basil sprig and use an immersion blender to puree the soup until smooth. Season with salt and pepper to taste before serving.",
    "To bake a decadent chocolate cake from scratch, start by preheating your oven to 350°F (175°C) and greasing and flouring two 9-inch round cake pans. In a large mixing bowl, cream together softened butter and granulated sugar until light and fluffy. Beat in eggs one at a time, making sure each egg is fully incorporated before adding the next. In a separate bowl, sift together all-purpose flour, cocoa powder, baking powder, baking soda, and salt. Divide the batter evenly between the prepared cake pans and bake for 25-30 minutes, or until a toothpick inserted into the center comes out clean.",
]

reference = [
    "The process of making spaghetti carbonara involves boiling pasta, crisping pancetta or guanciale, whisking together eggs and Parmesan cheese, and tossing everything together to create a creamy sauce.",
    "Preparing risotto entails sautéing onions and garlic, toasting Arborio rice, adding wine and broth gradually, and stirring until creamy and tender.",
    "Grilling a flavorful steak involves seasoning generously, preheating the grill, cooking to desired doneness, and letting it rest before slicing.",
    "Creating homemade tomato soup includes sautéing onions and garlic, simmering with tomatoes and broth, pureeing until smooth, and seasoning to taste.",
    "Baking a decadent chocolate cake requires creaming butter and sugar, beating in eggs and alternating dry ingredients with buttermilk before baking until done.",
]


eval_dataset = pd.DataFrame(
    {
        "context": context,
        "instruction": [instruction] * len(context),
        "reference": reference,
    }
)
```

#### Define prompt templates to compare



```
prompt_templates = [
    "Instruction: {instruction}. Article: {context}. Summary:",
    "Article: {context}. Complete this task: {instruction}, in three sentence. Summary:",
    "Goal: {instruction} and give me a TLDR. Here's a news article: {context}. Summary:",
]
```

#### Define metrics



```
metrics = ["rouge_l_sum", "fluency", "coherence", linguistic_acceptability_metric]
```

#### Define an EvalTask


```
generation_config = GenerationConfig(
    temperature=0.3,
    top_k=1,
)

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

gemini_model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
```


```
experiment_name = "eval-custom-metric"  # @param {type:"string"}

eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment=experiment_name,
)
```

#### Running Evaluation


```
eval_results = []

for i, prompt_template in enumerate(prompt_templates):
    eval_result = eval_task.evaluate(
        model=gemini_model,
        prompt_template=prompt_template,
    )

    eval_results.append((f"Prompt #{i}", eval_result))
```

#### Display Evaluation result and explanations


```
for title, eval_result in eval_results:
    display_eval_result(eval_result, title=title)
```


```
for title, eval_result in eval_results:
    display_explanations(eval_result, metrics=["linguistic_acceptability"])
```

#### Compare Eval Results


```
display_radar_plot(
    eval_results,
    metrics=["fluency", "coherence", "linguistic_acceptability"],
)
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>                <div id="a9d3298d-6b02-4b51-a961-10531de91359" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("a9d3298d-6b02-4b51-a961-10531de91359")) {                    Plotly.newPlot(                        "a9d3298d-6b02-4b51-a961-10531de91359",                        [{"fill":"toself","name":"Prompt #0","r":[4.8,4.2,4.0],"theta":["fluency","coherence","linguistic_acceptability"],"type":"scatterpolar"},{"fill":"toself","name":"Prompt #1","r":[4.0,4.6,4.8],"theta":["fluency","coherence","linguistic_acceptability"],"type":"scatterpolar"},{"fill":"toself","name":"Prompt #2","r":[4.0,4.0,4.4],"theta":["fluency","coherence","linguistic_acceptability"],"type":"scatterpolar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"polar":{"radialaxis":{"visible":true,"range":[0,5]}},"showlegend":true},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('a9d3298d-6b02-4b51-a961-10531de91359');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```
display_bar_plot(eval_results, metrics=["rouge_l_sum"])
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>                <div id="5230ecbe-5543-41b4-b036-0baf703c106c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("5230ecbe-5543-41b4-b036-0baf703c106c")) {                    Plotly.newPlot(                        "5230ecbe-5543-41b4-b036-0baf703c106c",                        [{"name":"Prompt #0","x":["rouge_l_sum\u002fmean"],"y":[0.318399916],"type":"bar"},{"name":"Prompt #1","x":["rouge_l_sum\u002fmean"],"y":[0.294384218],"type":"bar"},{"name":"Prompt #2","x":["rouge_l_sum\u002fmean"],"y":[0.252391738],"type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"barmode":"group","showlegend":true},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('5230ecbe-5543-41b4-b036-0baf703c106c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```
eval_task.display_runs()
```
