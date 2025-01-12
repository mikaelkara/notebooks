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

# Bring your own computation-based `CustomMetric` | Gen AI Evaluation SDK Tutorial


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/bring_your_own_computation_based_metric.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fbring_your_own_computation_based_metric.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/bring_your_own_computation_based_metric.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/bring_your_own_computation_based_metric.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Jason Dai](https://github.com/jsondai) |

## Overview

In this notebook, you'll learn how to use the *Vertex AI Python SDK for Gen AI Evaluation Service* to evaluate a generative AI model using locally-defined computation-based `CustomMetric`.


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
from vertexai.evaluation import CustomMetric, EvalTask, MetricPromptTemplateExamples
from vertexai.generative_models import GenerationConfig, GenerativeModel
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
    
    

### Define an evaluation dataset
In this tutorial we use a few examples from the open-source [XSum dataset](https://huggingface.co/datasets/EdinburghNLP/xsum?row=3) for summarization.

**Note: For best results we recommend using at least 100 examples.**



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

### Define the model for inference


```
# Model to be evaluated
generation_config = GenerationConfig(temperature=0.5, max_output_tokens=256, top_k=1)

model = GenerativeModel("gemini-1.5-pro", generation_config=generation_config)
```

### Create a computation-based `CustomMetric`

Create a fully-customized `CustomMetric` by defining a metric function.

* Custom metrics are computed on the client side, without using Vertex Gen AI Evaluation Service APIs.

* Mixing CustomMetric instances and other types of metrics is supported.


### Custom computation-based Metric "word_count"

Count the number of words in the generated model response.


```
def word_count(instance: dict[str, str]) -> dict[str, float]:
    """Count the number of words in the response."""

    response = instance["response"]
    score = len(response.split(" "))

    return {
        "word_count": score,
    }
```


```
custom_word_count_metric = CustomMetric(name="word_count", metric_function=word_count)
```


```
eval_dataset = pd.DataFrame(
    {
        "prompt": [
            "Describe the weather today.",
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "What are the benefits of regular exercise?",
            "Summarize the plot of 'The Great Gatsby'.",
        ],
    }
)

eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[
        custom_word_count_metric,
        MetricPromptTemplateExamples.Pointwise.VERBOSITY,
    ],
    experiment=EXPERIMENT_NAME,
)

result = eval_task.evaluate(
    model=model,
)
```


```
display_eval_result(result)
```


### Summary Metrics




  <div id="df-48061fca-74f6-407d-b6f8-050948c3d394" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_count</th>
      <th>word_count/mean</th>
      <th>word_count/std</th>
      <th>verbosity/mean</th>
      <th>verbosity/std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>114.4</td>
      <td>88.089727</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-48061fca-74f6-407d-b6f8-050948c3d394')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-48061fca-74f6-407d-b6f8-050948c3d394 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-48061fca-74f6-407d-b6f8-050948c3d394');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




### Row-based Metrics




  <div id="df-793d8dab-ea7b-492e-ace5-3e8a3f328047" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>response</th>
      <th>word_count/score</th>
      <th>verbosity/explanation</th>
      <th>verbosity/score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Describe the weather today.</td>
      <td>I do not have access to real-time information,...</td>
      <td>29</td>
      <td>The response is a bit verbose as it could simp...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the capital of France?</td>
      <td>The capital of France is **Paris**. 🇫🇷 \n</td>
      <td>8</td>
      <td>While the response accurately identifies Paris...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Explain how photosynthesis works.</td>
      <td>## Photosynthesis: Turning Sunlight into Sugar...</td>
      <td>187</td>
      <td>The response provides a good overview of photo...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What are the benefits of regular exercise?</td>
      <td>Regular exercise offers a wide range of benefi...</td>
      <td>179</td>
      <td>STEP 1: Assess completeness: The response prov...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Summarize the plot of 'The Great Gatsby'.</td>
      <td>'The Great Gatsby' follows narrator Nick Carra...</td>
      <td>169</td>
      <td>The response provides a comprehensive summary ...</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-793d8dab-ea7b-492e-ace5-3e8a3f328047')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-793d8dab-ea7b-492e-ace5-3e8a3f328047 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-793d8dab-ea7b-492e-ace5-3e8a3f328047');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a058e6dc-cf23-4aa6-80eb-a1ab05e611de">
  <button class="colab-df-quickchart" onclick="quickchart('df-a058e6dc-cf23-4aa6-80eb-a1ab05e611de')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a058e6dc-cf23-4aa6-80eb-a1ab05e611de button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>



#### Custom computation-based Metric "word_count_match" without inference

For scenarios where there is a preferred output length, word_count_match calculates a corresponding score on the scale of 0 to 1. Specifically, this scorers calculates how similar the number of words in the candidate model response is to the number of words in the reference output, where a score of 1.0 indicates that there are the same number of words in the candidate response as in the reference output. Scores less than 1.0 are calculated as ((len_reference-delta)/len_reference) where delta is the absolute difference in word lengths between the candidate response and reference. All negative computed values are truncated to 0.


```
import re


def _remove_punctuation(text: str, remove_apostrophe: bool):
    """Remove punctuation from the given text."""
    if remove_apostrophe:
        punctuation_regex = r"[^\w\s]"
    else:
        text = re.sub(r"\'(?![tsd]\b|ve\b|ll\b|re\b)", '"', text)
        punctuation_regex = r"[^\w\s\']"
    text = re.sub(punctuation_regex, "", text)
    return text


def _calculate_word_count_match(
    reference: str,
    response: str,
    remove_punctuation: bool,
    remove_apostrophe: bool,
):
    """Calculate word count match score for a single instance mapping negative values to 0."""
    if remove_punctuation:
        reference = _remove_punctuation(reference, remove_apostrophe)
        response = _remove_punctuation(response, remove_apostrophe)
    reference_length = len(reference.split())
    response_length = len(response.split())
    length_delta = abs(reference_length - response_length)
    if reference_length == 0:
        return 1 if response_length == 0 else 0
    if length_delta > reference_length:
        return 0
    return (reference_length - length_delta) / reference_length


def word_count_match(instance: dict[str, str]) -> dict[str, float]:
    """Calculate word count match scores for an instance."""
    response = instance["response"]
    reference = instance["reference"]

    score = _calculate_word_count_match(
        reference,
        response,
        remove_punctuation=True,
        remove_apostrophe=True,
    )
    return {
        "word_count_match": score,
    }
```


```
word_count_match_metric = CustomMetric(
    name="word_count_match",
    metric_function=word_count_match,
)
```


```
eval_dataset = pd.DataFrame(
    {
        "reference": [
            "It's sunny with a few clouds and a light breeze.",
            "The capital of France is Paris.",
            "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water.",
            "Regular exercise helps to maintain a healthy weight, improves mental health, and reduces the risk of chronic diseases.",
            "The Great Gatsby is about the mysterious millionaire Jay Gatsby and his obsession with the beautiful Daisy Buchanan, set against the backdrop of the Roaring Twenties.",
        ],
        "response": [
            "It's a bright sunny day with some clouds and a gentle breeze.",
            "Paris is the capital of France.",
            "Photosynthesis occurs when plants use sunlight to convert carbon dioxide and water into nutrients.",
            "Exercising regularly can help you keep a healthy weight, boost mental health, and lower the chances of chronic illnesses.",
            "The Great Gatsby tells the story of Jay Gatsby, a wealthy man who is in love with Daisy Buchanan, set during the Roaring Twenties.",
        ],
    }
)

eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[word_count_match_metric],
    experiment=EXPERIMENT_NAME,
)

custom_metric_result = eval_task.evaluate()
```


```
display_eval_result(custom_metric_result)
```


### Summary Metrics




  <div id="df-8a36a7a9-07f5-423a-a6d6-ab6b50d8e42c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row_count</th>
      <th>word_count_match/mean</th>
      <th>word_count_match/std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>0.88906</td>
      <td>0.095979</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8a36a7a9-07f5-423a-a6d6-ab6b50d8e42c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8a36a7a9-07f5-423a-a6d6-ab6b50d8e42c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8a36a7a9-07f5-423a-a6d6-ab6b50d8e42c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    </div>
  </div>




### Row-based Metrics




  <div id="df-eb69481e-4722-42e3-a2ed-44da6af9370e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reference</th>
      <th>response</th>
      <th>word_count_match/score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>It's sunny with a few clouds and a light breeze.</td>
      <td>It's a bright sunny day with some clouds and a...</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The capital of France is Paris.</td>
      <td>Paris is the capital of France.</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Photosynthesis is the process by which green p...</td>
      <td>Photosynthesis occurs when plants use sunlight...</td>
      <td>0.777778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regular exercise helps to maintain a healthy w...</td>
      <td>Exercising regularly can help you keep a healt...</td>
      <td>0.944444</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Great Gatsby is about the mysterious milli...</td>
      <td>The Great Gatsby tells the story of Jay Gatsby...</td>
      <td>0.923077</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-eb69481e-4722-42e3-a2ed-44da6af9370e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-eb69481e-4722-42e3-a2ed-44da6af9370e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-eb69481e-4722-42e3-a2ed-44da6af9370e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e2e6f4fb-0208-4573-bf04-aebc117465f9">
  <button class="colab-df-quickchart" onclick="quickchart('df-e2e6f4fb-0208-4573-bf04-aebc117465f9')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e2e6f4fb-0208-4573-bf04-aebc117465f9 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>


