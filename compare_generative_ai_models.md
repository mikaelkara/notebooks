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

 # Compare Generative AI Models | Gen AI Evaluation SDK Tutorial

 <table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/compare_generative_ai_models.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fcompare_generative_ai_models.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/compare_generative_ai_models.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/compare_generative_ai_models.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Jason Dai](https://github.com/jsondai) [Bo Zheng](https://github.com/coolalexzb) |

## Overview

In this tutorial, you will learn how to use the *Vertex AI Python SDK for Gen AI Evaluation Service* to score and evaluate different generative AI models on a specific evaluation task, `EvalTask`. Then visualize and compare the evaluation results for the `EvalTask` to select a generative model.


## Getting Started

### Install Vertex AI Python SDK for Gen AI Evaluation Service


```
%pip install --upgrade --user --quiet google-cloud-aiplatform[evaluation] plotly
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
EXPERIMENT_NAME = "eval-sdk-model-selection"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
# General
import inspect
import logging
import random
import string
import warnings

from IPython.display import HTML, Markdown, display
import pandas as pd
import plotly.graph_objects as go

# Main
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
from vertexai.generative_models import GenerativeModel, HarmBlockThreshold, HarmCategory
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
            ["context", "reference", "completed_prompt", "response"]
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


def plot_radar_plot(eval_results, metrics=None):
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True
    )

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

## Compare and Select Generative Models

You can define an `EvalTask` with pointwise metrics and an evaluation dataset, and then conduct a controlled experiment by running the evaluation multiple times with the same setup, but each run utilizing a different model. This allows you to isolate the impact of the model on the results, and ensuring consistent conditions for comparison.

### Define a Dataset


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

### Create an Evaluation Task

#### Documentation

Documentation and example usages for the `EvalTask`


```
print_doc(EvalTask)
```

#### Design text prompt template

For more task-specific prompt guidance, see https://cloud.google.com/vertex-ai/generative-ai/docs/text/text-prompts.


```
prompt_template = "{instruction}. Article: {context}. Summary:"
```

#### Define metrics



```
metrics = [
    MetricPromptTemplateExamples.Pointwise.TEXT_QUALITY,
    MetricPromptTemplateExamples.Pointwise.FLUENCY,
    MetricPromptTemplateExamples.Pointwise.SAFETY,
    MetricPromptTemplateExamples.Pointwise.VERBOSITY,
]
```

#### Define EvalTask & Experiment



```
summarization_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment=EXPERIMENT_NAME,
)
```

### Compare Gemini-1.5-Pro with Gemini-1.0-Pro

Evaluate **Gemini-1.5-Pro** model and **Gemini-1.0-Pro** model on the text summarization task defined above using Gen AI Eval Service SDK

#### Model settings



```
generation_config = {
    "max_output_tokens": 128,
    "temperature": 0.4,
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# Gemini-1.5-Pro
gemini_model_15 = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Gemini-1.0-Pro
gemini_model_1 = GenerativeModel(
    "gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
```


```
models = {
    "gemini-pro-1-5": gemini_model_15,
    "gemini-pro-1-0": gemini_model_1,
}
```

#### Running evaluation


```
eval_results = []
run_id = generate_uuid()

for model_name, model in models.items():
    experiment_run_name = f"eval-{model_name}-{run_id}"

    eval_result = summarization_eval_task.evaluate(
        model=model,
        prompt_template=prompt_template,
        experiment_run_name=experiment_run_name,
    )

    eval_results.append(
        (f"Model {model_name}", eval_result.summary_metrics, eval_result.metrics_table)
    )
```

#### Evaluation Report


```
for eval_result in eval_results:
    display_eval_report(eval_result)
```

### Explanations for model-based metrics


```
for eval_result in eval_results:
    display_explanations(eval_result[2], metrics=["fluency"])
```

### Compare Eval Results



```
plot_radar_plot(
    eval_results,
    metrics=[
        f"{metric}/mean"
        for metric in ["fluency", "text_quality", "safety", "verbosity"]
    ],
)
```


```
summarization_eval_task.display_runs()
```

## Compare Two Models Side-by-side (SxS)

To directly compare two models, you can define a `PairwiseMetric` within an `EvalTask` run. This approach allows for a head-to-head assessment of the models' performance.


```
from vertexai.evaluation import PairwiseMetric
```


```
# Baseline model for pairwise comparison
baseline_model = GenerativeModel(
    "gemini-1.0-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Candidate model for pairwise comparison
candidate_model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=generation_config,
    safety_settings=safety_settings,
)
```


```
# Create a "Pairwise Text Quality" metric
text_quality_prompt_template = MetricPromptTemplateExamples.get_prompt_template(
    "pairwise_text_quality"
)

pairwise_text_quality_metric = PairwiseMetric(
    metric="pairwise_text_quality",
    metric_prompt_template=text_quality_prompt_template,
    baseline_model=baseline_model,
)
```


```
pairwise_text_quality_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[pairwise_text_quality_metric],
    experiment=EXPERIMENT_NAME,
)

# Specify candidate model for pairwise comparison
pairwise_text_quality_result = pairwise_text_quality_eval_task.evaluate(
    model=candidate_model,
    prompt_template=prompt_template,
)
```


```
display_eval_report(
    (
        "Side-by-side EvalTask",
        pairwise_text_quality_result.summary_metrics,
        pairwise_text_quality_result.metrics_table,
    )
)
```


```
sample_pairwise_result(
    pairwise_text_quality_result, metric="pairwise_text_quality", n=1
)
```


```
display_pairwise_win_rate(pairwise_text_quality_result, metric="pairwise_text_quality")
```
