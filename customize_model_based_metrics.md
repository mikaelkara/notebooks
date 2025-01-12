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

# Customize Model-based Metrics to Evaluate a Gen AI model | Gen AI Evaluation SDK Tutorial

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/customize_model_based_metrics.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fcustomize_model_based_metrics.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/customize_model_based_metrics.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/customize_model_based_metrics.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai), [Naveksha Sood](https://github.com/navekshasood) |

## Overview

In this notebook, you'll learn how to use the *Vertex AI Python SDK for Gen AI Evaluation Service* to customize the model-based metrics and evaluate a generative AI model based on your criteria.

This notebook demonstrates:
* Templated customization using pre-defined fields for pointwise and pairwise model-based metrics

* Fully customized pointwise and pairwise model-based metrics


See also: 

- Learn more about [Vertex Gen AI Evaluation Service SDK](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview).

- Learn more about how to [define your evaluation metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval).

## Get Started

### Install Vertex AI Python SDK for Gen AI Evaluation Service


```
%pip install --upgrade --quiet google-cloud-aiplatform[evaluation]
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

### Set Google Cloud project information and initialize Vertex AI SDK for Python

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com). Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
EXPERIMENT_NAME = "customize-metrics"  # @param {type:"string"}

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries

Import the Vertex AI Python SDK and other required Python libraries.


```
# General
import inspect

from IPython.display import HTML, Markdown, display
import pandas as pd
import plotly.graph_objects as go

# Main
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PairwiseMetric,
    PairwiseMetricPromptTemplate,
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)
from vertexai.generative_models import GenerativeModel
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

## Define an evaluation dataset


In this tutorial, we use a few examples from the open-source [XSum dataset](https://huggingface.co/datasets/EdinburghNLP/xsum?row=3) for summarization.

 **Note: For best results, we recommend using at least 100 examples.**



```
instruction = "Summarize the following article: "

context = [
    'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail. "It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\'re neglected or forgotten," she said. "That may not be true but it is perhaps my perspective over the last few days. "Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?" Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Party\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process. "I was quite taken aback by the amount of damage that has been done," he said. "Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses." He said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk.',
    'A fire alarm went off at the Holiday Inn in Hope Street at about 04:20 BST on Saturday and guests were asked to leave the hotel. As they gathered outside they saw the two buses, parked side-by-side in the car park, engulfed by flames. One of the tour groups is from Germany, the other from China and Taiwan. It was their first night in Northern Ireland. The driver of one of the buses said many of the passengers had left personal belongings on board and these had been destroyed. Both groups have organised replacement coaches and will begin their tour of the north coast later than they had planned. Police have appealed for information about the attack. Insp David Gibson said: "It appears as though the fire started under one of the buses before spreading to the second. "While the exact cause is still under investigation, it is thought that the fire was started deliberately.',
    'Ferrari appeared in a position to challenge until the final laps, when the Mercedes stretched their legs to go half a second clear of the red cars. Sebastian Vettel will start third ahead of team-mate Kimi Raikkonen. The world champion subsequently escaped punishment for reversing in the pit lane, which could have seen him stripped of pole. But stewards only handed Hamilton a reprimand, after governing body the FIA said "no clear instruction was given on where he should park". Belgian Stoffel Vandoorne out-qualified McLaren team-mate Jenson Button on his Formula 1 debut. Vandoorne was 12th and Button 14th, complaining of a handling imbalance on his final lap but admitting the newcomer "did a good job and I didn\'t". Mercedes were wary of Ferrari\'s pace before qualifying after Vettel and Raikkonen finished one-two in final practice, and their concerns appeared to be well founded as the red cars mixed it with the silver through most of qualifying. After the first runs, Rosberg was ahead, with Vettel and Raikkonen splitting him from Hamilton, who made a mistake at the final corner on his first lap. But Hamilton saved his best for last, fastest in every sector of his final attempt, to beat Rosberg by just 0.077secs after the German had out-paced him throughout practice and in the first qualifying session. Vettel rued a mistake at the final corner on his last lap, but the truth is that with the gap at 0.517secs to Hamilton there was nothing he could have done. The gap suggests Mercedes are favourites for the race, even if Ferrari can be expected to push them. Vettel said: "Last year we were very strong in the race and I think we are in good shape for tomorrow. We will try to give them a hard time." Vandoorne\'s preparations for his grand prix debut were far from ideal - he only found out he was racing on Thursday when FIA doctors declared Fernando Alonso unfit because of a broken rib sustained in his huge crash at the first race of the season in Australia two weeks ago. The Belgian rookie had to fly overnight from Japan, where he had been testing in the Super Formula car he races there, and arrived in Bahrain only hours before first practice on Friday. He also had a difficult final practice, missing all but the final quarter of the session because of a water leak. Button was quicker in the first qualifying session, but Vandoorne pipped him by 0.064secs when it mattered. The 24-year-old said: "I knew after yesterday I had quite similar pace to Jenson and I knew if I improved a little bit I could maybe challenge him and even out-qualify him and that is what has happened. "Jenson is a very good benchmark for me because he is a world champion and he is well known to the team so I am very satisfied with the qualifying." Button, who was 0.5secs quicker than Vandoorne in the first session, complained of oversteer on his final run in the second: "Q1 was what I was expecting. Q2 he did a good job and I didn\'t. Very, very good job. We knew how quick he was." The controversial new elimination qualifying system was retained for this race despite teams voting at the first race in Australia to go back to the 2015 system. FIA president Jean Todt said earlier on Saturday that he "felt it necessary to give new qualifying one more chance", adding: "We live in a world where there is too much over reaction." The system worked on the basis of mixing up the grid a little - Force India\'s Sergio Perez ended up out of position in 18th place after the team miscalculated the timing of his final run, leaving him not enough time to complete it before the elimination clock timed him out. But it will come in for more criticism as a result of lack of track action at the end of each session. There were three minutes at the end of the first session with no cars on the circuit, and the end of the second session was a similar damp squib. Only one car - Nico Hulkenberg\'s Force India - was out on the track with six minutes to go. The two Williams cars did go out in the final three minutes but were already through to Q3 and so nothing was at stake. The teams are meeting with Todt and F1 commercial boss Bernie Ecclestone on Sunday at noon local time to decide on what to do with qualifying for the rest of the season. Todt said he was "optimistic" they would be able to reach unanimous agreement on a change. "We should listen to the people watching on TV," Rosberg said. "If they are still unhappy, which I am sure they will be, we should change it." Red Bull\'s Daniel Ricciardo was fifth on the grid, ahead of the Williams cars of Valtteri Bottas and Felipe Massa and Force India\'s Nico Hulkenberg. Ricciardo\'s team-mate Daniil Kvyat was eliminated during the second session - way below the team\'s expectation - and the Renault of Brit Jolyon Palmer only managed 19th fastest. German Mercedes protege Pascal Wehrlein managed an excellent 16th in the Manor car. Bahrain GP qualifying results Bahrain GP coverage details',
    'Gundogan, 26, told BBC Sport he "can see the finishing line" after tearing cruciate knee ligaments in December, but will not rush his return. The German missed the 2014 World Cup following back surgery that kept him out for a year, and sat out Euro 2016 because of a dislocated kneecap. He said: "It is heavy mentally to accept that." Gundogan will not be fit for the start of the Premier League season at Brighton on 12 August but said his recovery time is now being measured in "weeks" rather than months. He told BBC Sport: "It is really hard always to fall and fight your way back. You feel good and feel ready, then you get the next kick. "The worst part is behind me now. I want to feel ready when I am fully back. I want to feel safe and confident. I don\'t mind if it is two weeks or six." Gundogan made 15 appearances and scored five goals in his debut season for City following his £20m move from Borussia Dortmund. He is eager to get on the field again and was impressed at the club\'s 4-1 win over Real Madrid in a pre-season game in Los Angeles on Wednesday. Manager Pep Guardiola has made five new signings already this summer and continues to have an interest in Arsenal forward Alexis Sanchez and Monaco\'s Kylian Mbappe. Gundogan said: "Optimism for the season is big. It is huge, definitely. "We felt that last year as well but it was a completely new experience for all of us. We know the Premier League a bit more now and can\'t wait for the season to start." City complete their three-match tour of the United States against Tottenham in Nashville on Saturday. Chelsea manager Antonio Conte said earlier this week he did not feel Tottenham were judged by the same standards as his own side, City and Manchester United. Spurs have had the advantage in their recent meetings with City, winning three and drawing one of their last four Premier League games. And Gundogan thinks they are a major threat. He said: "Tottenham are a great team. They have the style of football. They have young English players. Our experience last season shows it is really tough to beat them. "They are really uncomfortable to play against. "I am pretty sure, even if they will not say it loud, the people who know the Premier League know Tottenham are definitely a competitor for the title.',
]

reference = [
    "Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.",
    "Two tourist buses have been destroyed by fire in a suspected arson attack in Belfast city centre.",
    "Lewis Hamilton stormed to pole position at the Bahrain Grand Prix ahead of Mercedes team-mate Nico Rosberg.",
    "Manchester City midfielder Ilkay Gundogan says it has been mentally tough to overcome a third major injury.",
]

eval_dataset = pd.DataFrame(
    {
        "prompt": [instruction + item for item in context],
        "reference": reference,
    }
)

eval_dataset.head()
```

## Define a Candidate Model


```
# Model to be evaluated
generation_config = {"temperature": 0.5, "max_output_tokens": 256, "top_k": 1}

model = GenerativeModel("gemini-1.5-pro", generation_config=generation_config)
```

## Customize metrics with your criteria

Gen AI Evaluation Service SDK empowers you to customize metrics to align precisely with your specific evaluation criteria. Whether you need to modify individual components, or the entire metric structure, or define your own
metric computation logic, the SDK offers the flexibility to tailor the evaluation process to your unique requirements.


### Templated Customization

#### Create a Pointwise MetricPromptTemplate with pre-defined default fields

The required fields for templated customization are the criteria for evaluation, the rating rubric and the input variables.


* **Criteria** can be a single criterion or a combination of multiple sub-criteria.

* **Rating rubric** contains the values for the score and what each score corresponds to.

* **Input variables** are the input columns users need to provide in the evaluation dataset to complete the prompt for the autorater and get a response.


Other parts of the template such as "instruction" and "evaluation steps" have default text that can be updated as well if needed.


```
linguistic_acceptability_criteria = {
    "Proper Grammar": "The language's grammar rules are correctly followed, including but not limited to sentence structures, verb tenses, subject-verb agreement, proper punctuation, and capitalization.",
    "Appropriate word choice": "Words chosen are appropriate and purposeful given their relative context and positioning in the text. Vocabulary demonstrates prompt understanding.",
    "Reference Alignment": "The response is consistent and aligned with the reference.",
}

linguistic_acceptability_pointwise_rubric = {
    "5": "Excellent: The writing is grammatically correct, uses appropriate vocabulary and aligns perfectly with the reference.",
    "4": "Good: The writing is generally grammatically correct, uses appropriate vocabulary and aligns well with the reference.",
    "3": "Satisfactory: The writing may have minor grammatical errors or use less-appropriate vocabulary, but it aligns reasonably well with the reference.",
    "2": "Unsatisfactory: The writing has significant grammatical errors, uses inappropriate vocabulary, deviates significantly from the reference.",
    "1": "Poor: The writing is riddled with grammatical errors, uses highly inappropriate vocabulary, is completely unrelated to the reference.",
}
```


```
# The metric prompt template contains default prompts pre-defined for unspecified components.
linguistic_acceptability_metric_prompt_template = PointwiseMetricPromptTemplate(
    criteria=linguistic_acceptability_criteria,
    rating_rubric=linguistic_acceptability_pointwise_rubric,
    input_variables=["prompt", "reference"],
)

# Display the assembled prompt template that will be sent to Gen AI Eval Service
# along with the input data for model-based evaluation.
print(linguistic_acceptability_metric_prompt_template.prompt_data)
```


```
# Define this custom "linguistic_acceptability" model-based metric.
linguistic_acceptability = PointwiseMetric(
    metric="linguistic_acceptability",
    metric_prompt_template=linguistic_acceptability_metric_prompt_template,
)

linguistic_acceptability_eval_task = EvalTask(
    dataset=eval_dataset, metrics=[linguistic_acceptability], experiment=EXPERIMENT_NAME
)

linguistic_acceptability_result = linguistic_acceptability_eval_task.evaluate(
    model=model,
)
```


```
display_eval_result(linguistic_acceptability_result)
```

#### Create a Pairwise MetricPromptTemplate with pre-defined default fields


```
# Define a baseline model for pairwise evaluation
baseline_model = GenerativeModel(
    "gemini-1.0-pro",
)
```


```
linguistic_acceptability_pairwise_rubric = {
    "A": "Response A answers the given question as per the criteria better than response B.",
    "SAME": "Response A and B answers the given question equally well as per the criteria.",
    "B": "Response B answers the given question as per the criteria better than response A.",
}

pairwise_linguistic_acceptability_prompt_template = PairwiseMetricPromptTemplate(
    criteria=linguistic_acceptability_criteria,
    rating_rubric=linguistic_acceptability_pairwise_rubric,
    input_variables=["prompt", "reference"],
)

pairwise_linguistic_acceptability = PairwiseMetric(
    metric="pairwise_linguistic_acceptability",
    metric_prompt_template=pairwise_linguistic_acceptability_prompt_template,
    baseline_model=baseline_model,
)

print(pairwise_linguistic_acceptability_prompt_template.prompt_data)
```


```
pairwise_linguistic_acceptability_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[pairwise_linguistic_acceptability],
    experiment=EXPERIMENT_NAME,
)

pairwise_linguistic_acceptability_result = (
    pairwise_linguistic_acceptability_eval_task.evaluate(
        model=model,
    )
)
```


```
display_eval_result(pairwise_linguistic_acceptability_result)
```

### Free-form customization

#### Create a fully-customized PointwiseMetric


```
# Take a look at an example Pointwise metric prompt template
print(MetricPromptTemplateExamples.get_prompt_template("verbosity"))
```


```
# Make changes to create your own metric prompt template
free_form_pointwise_metric_prompt = """
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by AI models.
We will provide you with the user input and an AI-generated response.
You should first read the user input carefully for analyzing the task, and then evaluate the quality of the responses based on the Criteria provided in the Evaluation section below.
You will assign the response a rating following the Rating Rubric and Evaluation Steps. Give step-by-step explanations for your rating, and only choose ratings from the Rating Rubric.


# Evaluation
## Metric Definition
You will be assessing the verbosity of the model's response, which measures its conciseness and ability to provide sufficient detail without being overly wordy or excessively brief.

## Criteria
Verbosity: The response is appropriately concise, providing sufficient detail without using complex language to thoroughly address the prompt without being overly wordy or excessively brief.
Reference Alignment: The response is consistent and aligned with the reference.

## Rating Rubric
2: (Excessive Verbosity) The response is overly verbose, filled with unnecessary words and repetition, lacks clarity and conciseness, and deviates significantly from the reference.
1: (Somewhat Verbose) The response contains some unnecessary wordiness or repetition, but still provides all necessary information, is generally easy to understand, and aligns reasonably well with the reference.
0: (Just Right) The response is perfectly concise, providing all necessary information in a clear and succinct manner without any unnecessary wordiness or repetition, and aligns closely with the reference.
-1: (Somewhat Brief) The response is slightly brief and could benefit from additional details or explanations to fully address the prompt, but still provides the core information and is generally understandable, though it may deviate slightly from the reference.
-2: (Too Short) The response is excessively brief and lacks crucial information or explanations needed to adequately address the prompt, leaving the reader with unanswered questions or a sense of incompleteness, and deviates significantly from the reference.


## Evaluation Steps
STEP 1: Assess conciseness: Is the response free of unnecessary wordiness, repetition, or filler words? Could any sentences or phrases be shortened or simplified without losing meaning?
STEP 2: Assess reference alignment: Does the response align with the reference provided?
STEP 3: Assess overall balance: Does the response strike the right balance between providing sufficient detail and being concise? Is it aligned with the reference.

# User Inputs and AI-generated Response
## User Inputs
### Prompt
{prompt}

### Reference
{reference}

## AI-generated Response
{response}
"""
```


```
free_form_pointwise_text_quality_metric = PointwiseMetric(
    metric="free_form_pointwise_text_quality",
    metric_prompt_template=free_form_pointwise_metric_prompt,
)


free_form_pointwise_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[free_form_pointwise_text_quality_metric],
    experiment=EXPERIMENT_NAME,
)

free_form_pointwise_result = free_form_pointwise_eval_task.evaluate(model=model)
```


```
display_eval_result(free_form_pointwise_result)
```

#### Create a fully-customized PairwiseMetric


```
free_form_pairwise_metric_prompt = """
# Instruction
You are an expert evaluator. Your task is to evaluate the quality of the responses generated by two AI models. We will provide you with the user input and a pair of AI-generated responses (Response A and Response B).
You should first read the user input carefully for analyzing the task, and then evaluate the quality of the responses based on the Criteria provided in the Evaluation section below.
You will first judge responses individually, following the Rating Rubric and Evaluation Steps.
Then you will give step-by-step explanations for your judgement, compare results to declare the winner based on the Rating Rubric and Evaluation Steps.

# Evaluation
## Metric Definition
You will be assessing the verbosity of each model's response, which measures its conciseness and ability to provide sufficient detail without being overly wordy or excessively brief.

## Criteria
Verbosity: The response is appropriately concise, providing sufficient detail without using complex language to thoroughly address the prompt without being overly wordy or excessively brief.
Reference Alignment: The response is consistent and aligned with the reference.

## Rating Rubric
"A": Response A is more appropriately concise than Response B. Response A is better aligned with the reference.
"SAME": Response A and B are equally concise. They both are equally aligned with the reference.
"B": Response B is more appropriately concise than Response A. Response B is better aligned with the reference.

## Evaluation Steps
STEP 1: Analyze Response A based on the criteria regarding completeness, conciseness, and alignment with reference.
STEP 2: Analyze Response B based on the criteria regarding completeness, conciseness, and alignment with reference.
STEP 3: Compare the overall performance of Response A and Response B based on your analyses and assessment.
STEP 4: Output your preference of "A", "SAME" or "B" to the pairwise_choice field according to the Rating Rubric.
STEP 5: Output your assessment reasoning in the explanation field, justifying your choice by highlighting the specific strengths and weaknesses of each response in terms of verbosity.

# User Inputs and AI-generated Responses
## User Inputs
### Prompt
{prompt}

### Reference
{reference}

# AI-generated Responses
### Response A
{baseline_model_response}

### Response B
{response}
"""
```


```
free_form_pairwise_metric = PairwiseMetric(
    metric="free_form_pairwise_verbosity_reference_alignment",
    metric_prompt_template=free_form_pairwise_metric_prompt,
    baseline_model=baseline_model,
)
free_form_pairwise_eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[free_form_pairwise_metric],
    experiment=EXPERIMENT_NAME,
)

free_form_pairwise_result = free_form_pairwise_eval_task.evaluate(
    model=model,
)
```


```
display_eval_result(free_form_pairwise_result)
```
