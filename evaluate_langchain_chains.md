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

 # Evaluate LangChain | Gen AI Evaluation SDK Tutorial

 <table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_langchain_chains.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fevaluation%2Fevaluate_langchain_chains.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/evaluation/evaluate_langchain_chains.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/evaluation/evaluate_langchain_chains.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Elia Secchi](https://github.com/eliasecchig) |

## Overview

With this tutorial, you learn how to evaluate the performance of a conversational LangChain chain using the *Vertex AI Python SDK for Gen AI Evaluation Service*. The notebook utilizes a dummy chatbot designed to provide recipe suggestions.

The tutorial goes trough:
1. Data preparation
2. Setting up the LangChain chain
3. Set-up a custom metric
4. Run evaluation with a combination of custom metrics and built-in metrics.
5. Log results into an experiment run and analyze different runs.

### Costs

This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.


## Getting Started

### Install Vertex AI SDK for Rapid Evaluation


```
%pip install --quiet --upgrade nest_asyncio
%pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain
%pip install --upgrade --user --quiet "google-cloud-aiplatform[evaluation]"
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
# import sys

# if "google.colab" in sys.modules:
#     from google.colab import auth

#     auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import logging
from typing import Any
import warnings

from google.cloud import aiplatform
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI

# Main
import pandas as pd
from tqdm import tqdm
import vertexai
from vertexai.evaluation import CustomMetric, EvalTask

# General
import yaml
```

### Library settings


```
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
```

### Helper functions


```
def generate_multiturn_history(df: pd.DataFrame):
    """Processes a DataFrame of messages to add conversation history for each message.

    This function takes a DataFrame containing message data and iterates through each row.
    For each message in a row, it constructs the conversation history up to that point by
    accumulating previous user and AI messages. This conversation history is then added
    to the message data, and the processed messages are returned as a new DataFrame.

    Args:
        df: A DataFrame containing message data. It is expected to have a column named
            "messages" where each entry is a list of dictionaries representing messages in
            a conversation. Each message dictionary should have "user" and "reference" keys.

    Returns:
        A DataFrame with the processed messages. Each message dictionary will now have an
        additional "conversation_history" key containing a list of tuples representing the
        conversation history leading up to that message. The tuples are of the form
        ("user", message_text) or ("ai", message_text).
    """
    processed_messages = []
    for i, row in df.iterrows():
        conversation_history = []
        for message in row["messages"]:
            message["conversation_history"] = conversation_history
            processed_messages.append(message)
            conversation_history = conversation_history + [
                ("user", message["user"]),
                ("ai", message["reference"]),
            ]
    return pd.DataFrame(processed_messages)


def pairwise(iterable):
    """Creates an iterable with tuples paired together
    e.g s -> (s0, s1), (s2, s3), (s4, s5), ...
    """
    a = iter(iterable)
    return zip(a, a)


def batch_generate_message(row: dict, callable: Any) -> dict:
    """
    Predicts a response from a chat agent.

    Args:
        callable (ChatAgent): A chat agent.
        row (dict): A message.
    Returns:
        dict: The predicted response.
    """
    index, message = row

    messages = []
    for user_message, ground_truth in pairwise(message.get("conversation_history", [])):
        messages.append(("user", user_message))
        messages.append(("ai", ground_truth))
    messages.append(("user", message["user"]))
    input_callable = {"messages": messages, **message.get("callable_kwargs", {})}
    response = callable.invoke(input_callable)
    message["response"] = response.content
    message["response_obj"] = response
    return message


def batch_generate_messages(
    messages: pd.DataFrame, callable: Any, max_workers: int = 4
) -> pd.DataFrame:
    """
    Generates AI-powered responses to a series of user messages using a provided callable.

    This function efficiently processes a Pandas DataFrame containing user-AI message pairs,
     utilizing the specified callable (either a LangChain Chain or a custom class with an
     `invoke` method) to predict AI responses in parallel.

    Args:
        callable (callable): A callable object (e.g., LangChain Chain, custom class) used
            for response generation. Must have an `invoke(messages: dict) ->
            langchain_core.messages.ai.AIMessage` method.
            The `messages` dict follows this structure:
            {"messages" [("user", "first"),("ai", "a response"), ("user", "a follow up")]}

        messages (pd.DataFrame): A DataFrame with one column named 'messages' containing
            the list of user-AI message pairs as described above.

        max_workers (int, optional): The number of worker processes to use for parallel
            prediction. Defaults to the number of available CPU cores.

    Returns:
        pd.DataFrame: A DataFrame containing the original messages and a new column with the predicted AI responses.

    Example:
        ```python
        messages_df = pd.DataFrame({
            "messages": [
                [{"user": "What's the weather today?", "reference": "It's sunny."}],
                [{"user": "Tell me a joke.", "reference": "Why did the scarecrow win an award?...
                    Because he was outstanding in his field!"}]
            ]
        })

        responses_df = batch_generate_messages(my_callable, messages_df)
        ```
    """
    logging.info("Executing batch scoring")
    predicted_messages = []
    with ThreadPoolExecutor(max_workers) as pool:
        partial_func = partial(batch_generate_message, callable=callable)
        for message in tqdm(
            pool.map(partial_func, messages.iterrows()), total=len(messages)
        ):
            predicted_messages.append(message)
    return pd.DataFrame(predicted_messages)
```

## Import ground truth data for evaluation

In this sample, we will use 2 conversations as a ground truth. Every message in the conversations, along with the relative history, will be used to produce a response with a foundational model.



```
%%writefile chats.yaml
- messages:
  - user: Hi
    reference: Hi, how can I help you?
  - user: I'm looking for a recipe for a healthy dinner. Do you have any recommendations?
    reference: Sure, I can help you with that. What are your dietary restrictions? Are you vegetarian, vegan, gluten-free, or anything else?
  - user: I'm not vegetarian or vegan, but I am gluten-free.
    reference: 'Okay, I ll keep that in mind. Here are a few recipes that I think you might like:
  * **Grilled Salmon with Roasted Vegetables:** This is a delicious and healthy recipe that is perfect for a weeknight meal. The salmon is grilled to perfection and the roasted vegetables add a touch of sweetness.
  * **Chicken Stir-Fry:** This is a quick and easy stir-fry that is perfect for busy weeknights. The chicken is cooked with vegetables and a light sauce.
  * **Lentil Soup:** This is a hearty and healthy soup that is perfect for a cold winter day. The lentils are packed with protein and fiber, and the soup is also gluten-free.'
  - user: Those all sound great! I think I'm going to try the grilled salmon with roasted vegetables.
    reference: That's a great choice! I hope you enjoy it.
  - user: Thanks for your help!
    reference: You're welcome! Is there anything else I can help you with today?
  - user: No, that's all. Thanks again!
    reference: You're welcome! Have a great day!
- messages:
  - user: Hi
    reference: Hi, how can I help you?
  - user: I'm looking for a recipe for a romantic dinner. Do you have any recommendations?
    reference: Sure, I can help you with that. What are your dietary restrictions? Are you vegetarian, vegan, gluten-free, or anything else?
  - user: I'm vegetarian.
    reference: 'Sure, I can help you find a healthy vegetarian dinner recipe. Here are a few ideas:
    * **Burnt aubergine veggie chilli:** This is a hearty and flavorful dish that is packed with nutrients. The roasted aubergine gives it a smoky flavor, and the lentils and beans add protein and fiber.
  * **Simple mushroom curry:** This is a quick and easy curry that is perfect for a weeknight meal. The mushrooms are cooked in a creamy sauce with spices, and the whole dish is ready in under 30 minutes.
  * **Vegetarian enchiladas:** This is a classic Mexican dish that is easy to make vegetarian. The enchiladas are filled with a variety of vegetables, and they are topped with a delicious sauce.
  * **Braised sesame tofu:** This is a flavorful and satisfying dish that is perfect for a cold night. The tofu is braised in a sauce with sesame, ginger, and garlic, and it is served over rice or noodles.
  * **Roast garlic & tahini spinach:** This is a light and healthy dish that is perfect for a spring or summer meal. The spinach is roasted with garlic and tahini, and it is served with a side of pita bread.

  These are just a few ideas to get you started. There are many other great vegetarian dinner recipes out there, so you are sure to find something that you will enjoy.'
  - user: Those all sound great! I like the Burnt aubergine veggie chilli
    reference: That's a great choice! I hope you enjoy it.
```

Let's now load all the messages into a Pandas DataFrame:


```
y = yaml.safe_load(open("chats.yaml"))
df = pd.DataFrame(y)
df
```

**Decomposing the chat message input/output pairs**

We are now ready for decomposing multi-turn history. This is essential to enable batch prediction.

We decompose the `messages` list into single input/output pairs. The input is always composed by `user message`, `reference message` and `conversation_history`.

**Example:**

Given the following chat:

```yaml
- messages:
- user: Hi
reference: Hi, how can I help you?
- user: I'm looking for a recipe for a healthy dinner. Do you have any recommendations?
reference: Sure, I can help you with that. What are your dietary restrictions? Are you vegetarian, vegan, gluten-free, or anything else?
```

We can generate these two input/output samples:

```yaml
- user: Hi
  reference: Hi, how can I help you?
  conversation_history: []

- user: I'm looking for a recipe for a healthy dinner....
  reference: Sure, I can help you with that. What are your ...
  conversation_history:
  - user: Hi
    reference: Hi, how can I help you?
```


```
df_processed = generate_multiturn_history(df)
df_processed
```

## Let's define our LangChain chain!

We now need to define our LangChain Chain. For this tutorial, we will create a simple conversational chain capable of producing cooking recipes for users.


```
llm = ChatVertexAI(model_name="gemini-1.5-flash-001", temperature=0)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a conversational bot that produce nice recipes for users based on a question.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = template | llm
```

We can test our chain:


```
chain.invoke([("human", "Hi there!")])
```

## Batch scoring

We are now ready to perform batch scoring. To perform batch scoring we will leverage the utility function `batch_generate_messages`

Have a look at the definition to see the expected input format.


```
help(batch_generate_messages)
```


```
scored_data = batch_generate_messages(messages=df_processed, callable=chain)
scored_data
```

## Evaluation

We'll utilize [Vertex AI Rapid Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/rapid-evaluation) to assess our generative AI model's performance. This service within Vertex AI streamlines the evaluation process, integrates with [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments) for tracking, and offers a range of [pre-built metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval#task-and-metrics) and the capability to define custom ones.


#### Define a CustomMetric using Gemini model

Define a customized Gemini model-based metric function, with explanations for the score. The registered custom metrics are computed on the client side, without using online evaluation service APIs.


```
evaluator_llm = ChatVertexAI(
    model_name="gemini-1.5-flash-001",
    temperature=0,
    response_mime_type="application/json",
)


def custom_faithfulness(instance):
    prompt = f"""You are examining written text content. Here is the text:
************
Written content: {instance["response"]}
************
Original source data: {instance["reference"]}

Examine the text and determine whether the text is faithful or not.
Faithfulness refers to how accurately a generated summary reflects the essential information and key concepts present in the original source document.
A faithful summary stays true to the facts and meaning of the source text, without introducing distortions, hallucinations, or information that wasn't originally there.

Your response must be an explanation of your thinking along with single integer number on a scale of 0-5, 0
the least faithful and 5 being the most faithful.

Produce results in JSON

Expected format:

```json
{{
    "explanation": "< your explanation>",
    "custom_faithfulness": <your score>
}}
```
"""

    result = evaluator_llm.invoke([("human", prompt)])
    result = json.loads(result.content)
    return result


# Register Custom Metric
custom_faithfulness_metric = CustomMetric(
    name="custom_faithfulness",
    metric_function=custom_faithfulness,
)
```

### Run Evaluation with CustomMetric


```
experiment_name = "rapid-eval-langchain-eval"  # @param {type:"string"}
```

We are now ready to run the evaluation. We will use different metrics, combining the custom metric we defined above with some pre-built metrics.

Results of the evaluation will be automatically tagged into the experiment_name we defined.

You can click `View Experiment`, to see the experiment in Google Cloud Console.


```
metrics = ["fluency", "coherence", "safety", custom_faithfulness_metric]

eval_task = EvalTask(
    dataset=scored_data,
    metrics=metrics,
    experiment=experiment_name,
    metric_column_mapping={"prompt": "user"},
)
eval_result = eval_task.evaluate()
```

Once an eval result is produced, we are able to display summary metrics:


```
eval_result.summary_metrics
```

We are also able to display a pandas dataframe containing a detailed summary of how our eval dataset performed and relative granular metrics.


```
eval_result.metrics_table
```

## Iterating over the prompt

Let's perform some simple changes to our chain to see how our evaluation results change.


```
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a conversational bot that produce nice recipes for users based on a question.
Before suggesting a recipe, you should ask for the dietary requirements..""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

new_chain = template | llm
```


```
scored_data = batch_generate_messages(messages=df_processed, callable=new_chain)
scored_data.rename(columns={"text": "response"}, inplace=True)
scored_data
```


```
metrics = ["fluency", "coherence", "safety", custom_faithfulness_metric]
eval_task = EvalTask(
    dataset=scored_data,
    metrics=metrics,
    experiment=experiment_name,
    metric_column_mapping={"prompt": "user"},
)
eval_result = eval_task.evaluate()
```


```
eval_result.summary_metrics
```

#### Let's compare both eval results

We can do that by using the method `display_runs` for a given `eval task` object to see which prompt template performed best on our dataset.


```
df = vertexai.preview.get_experiment_df(experiment_name).T
df
```

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial.


```
import os

# Delete Experiments
delete_experiments = True
if delete_experiments or os.getenv("IS_TESTING"):
    experiments_list = aiplatform.Experiment.list()
    for experiment in experiments_list:
        experiment.delete()
```
