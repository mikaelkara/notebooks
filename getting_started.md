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

# Getting Started - Template

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/sample-apps/e2e-gen-ai-app-starter-pack/notebooks/getting_started.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fsample-apps%2Fe2e-gen-ai-app-starter-pack%2Fnotebooks%2Fgetting_started.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/sample-apps/e2e-gen-ai-app-starter-pack/notebooks/getting_started.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/sample-apps/e2e-gen-ai-app-starter-pack/notebooks/getting_started.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Elia Secchi](https://github.com/eliasecchig) |

## Overview

This tutorial walks you through the process of developing and assessing a chain - a sequence of steps that power an AI application. 
These operations may include interactions with language models, utilization of tools, or data preprocessing steps, aiming to solve a given use case e.g a chatbot that provides grounded information.

You'll learn how to:

1. Build chains using three different approaches:
   - [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
   - [LangGraph](https://python.langchain.com/docs/langgraph/)
   - A custom Python implementation. This is to enable implementation with other SDKs ( e.g [Vertex AI SDK](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk ), [LlamaIndex](https://www.llamaindex.ai/))  and to allow granular control on the sequence of steps in the chain
   
2. Evaluate the performance of your chains using [Vertex AI Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview)

Finally, the tutorial discusses next steps for deploying your chain in a production application

By the end of this tutorial, you'll have a solid foundation for developing and refining your own Generative AI chains.

## Get Started

### Install required packages using Poetry (Recommended)

This template uses [Poetry](https://python-poetry.org/) as tool to manage project dependencies. 
Poetry makes it easy to install and keep track of the packages your project needs.

To run this notebook with Poetry, follow these steps:
1. Make sure Poetry is installed. See the [relative guide for installation](https://python-poetry.org/docs/#installation).

2. Make sure that dependencies are installed. From your command line:

   ```bash
   poetry install --with streamlit,jupyter
   ```

3. Run Jupyter:

   ```bash
   poetry run jupyter
   ```
   
4. Open this notebook in the Jupyter interface.

### (Alternative) Install Vertex AI SDK and other required packages 


```
%pip install --upgrade --user --quiet langchain-core langchain-google-vertexai langchain-google-community langchain langgraph traceloop-sdk
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
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
# Add the parent directory to the Python path. This allows importing modules from the parent directory
import sys

sys.path.append("../")
```


```
from collections.abc import AsyncIterator
import json
from typing import Any, Literal

from app.eval.utils import generate_multiturn_history
from app.patterns.custom_rag_qa.templates import (
    inspect_conversation_template,
    rag_template,
    template_docs,
)
from app.patterns.custom_rag_qa.vector_store import get_vector_store
from app.utils.decorators import custom_chain
from app.utils.output_types import OnChatModelStreamEvent, OnToolEndEvent
from google.cloud import aiplatform
from langchain.schema import Document
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_community.vertex_rank import VertexAIRank
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import pandas as pd
from vertexai.evaluation import CustomMetric, EvalTask
import yaml
```

## Chain Interface

This section outlines a possible interface for the chain, which, if implemented, ensures compatibility with the FastAPI server application included in the template. However, it's important to note that you have the flexibility to explore and implement alternative interfaces that suit their specific needs and requirements.


### Input Interface

The chain must provide an `astream_events` method that accepts a dictionary with a "messages" key.
The "messages" value should be a list of LangChain [HumanMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.human.HumanMessage.html), [AIMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html) objects and [ToolMessage](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolMessage.html).

For example a possible input might be:

```py
{
    "messages": [
        HumanMessage("first"),
        AIMessage("a response"),
        HumanMessage("a follow up"),
    ]
}
```

Alternatively you can use the shortened form:

```py
{"messages": [("user", "first"), ("ai", "a response"), ("user", "a follow up")]}
```

### Output Interface

All chains use the [LangChain Stream Events (v2) API](https://python.langchain.com/docs/how_to/streaming/#using-stream-events). This API supports various use cases (simple chains, RAG, Agents). This API emits asynchronous events that can be used to stream the chain's output.

LangChain chains (LCEL, LangGraph) automatically implement the `astream_events` API. 

We provide examples of emitting `astream_events`-compatible events with custom Python code, allowing implementation with other SDKs (e.g., Vertex AI, LLamaIndex).

### Customizing I/O Interfaces

To modify the Input/Output interface, update `app/server.py` and related unit and integration tests.

## Events supported

The following list defines the events that are captured and supported by the Streamlit frontend.


```
SUPPORTED_EVENTS = [
    "on_tool_start",
    "on_tool_end",
    "on_retriever_start",
    "on_retriever_end",
    "on_chat_model_stream",
]
```

### Define the LLM
We set up the Large Language Model (LLM) for our conversational bot.


```
llm = ChatVertexAI(model_name="gemini-1.5-flash-002", temperature=0)
```

### Leverage LangChain LCEL

LangChain Expression Language (LCEL) provides a declarative approach to composing chains seamlessly. Key benefits include:

1. Rapid prototyping to production deployment without code changes
2. Scalability from simple "prompt + LLM" chains to complex, multi-step workflows
3. Enhanced readability and maintainability of chain logic

For comprehensive guidance on LCEL implementation, refer to the [official documentation](https://python.langchain.com/docs/expression_language/get_started).


```
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable culinary assistant specializing in providing"
            "detailed cooking recipes. Your responses should be informative, engaging, "
            "and tailored to the user's specific requests. Include ingredients, "
            "step-by-step instructions, cooking times, and any helpful tips or "
            "variations. If asked about dietary restrictions or substitutions, offer "
            "appropriate alternatives.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = template | llm
```

Let's test the chain with a dummy question:


```
input_message = {"messages": [("human", "Can you provide me a Lasagne recipe?")]}

async for event in chain.astream_events(input=input_message, version="v2"):
    if event["event"] in SUPPORTED_EVENTS:
        print(event["data"])
```

This methodology is used for the chain defined in the [`app/chain.py`](../app/chain.py) file.

We can also leverage the `invoke` method for synchronous invocation.


```
response = chain.invoke(input=input_message)
print(response.content)
```

### Use LangGraph

LangGraph is a framework for building stateful, multi-actor applications with Large Language Models (LLMs). 
It extends the LangChain library, allowing you to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner.


```
# 1. Define tools
@tool
def search(query: str):
    """Simulates a web search. Use it get information on weather. E.g what is the weather like in a region"""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]

# 2. Set up the language model
llm = llm.bind_tools(tools)


# 3. Define workflow components
def should_continue(state: MessagesState) -> Literal["tools", END]:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


def call_model(state: MessagesState, config: RunnableConfig):
    """Calls the language model and returns the response."""
    response = llm.invoke(state["messages"], config)
    return {"messages": response}


# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# 5. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 6. Compile the workflow
chain = workflow.compile()
```

Let's test the new chain with a dummy question:


```
input_message = {"messages": [("human", "What is the weather like in NY?")]}

async for event in chain.astream_events(input=input_message, version="v2"):
    if event["event"] in SUPPORTED_EVENTS:
        print(event["data"])
```

This methodology is used for the chain defined in the [`app/patterns/langgraph_dummy_agent/chain.py`](../app/patterns/langgraph_dummy_agent/chain.py) file.

### Use custom python code

You can also use pure python code to orchestrate the different steps of your chain and emit `astream_events` [API compatible events](https://python.langchain.com/docs/how_to/streaming/#using-stream-events). 

This offers full flexibility in how the different steps of a chain are orchestrated and allows you to include other SDK frameworks such as [Vertex AI SDK](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk ), [LlamaIndex](https://www.llamaindex.ai/).

We demonstrate this third methodology by implementing a RAG chain. The function `get_vector_store` provides a brute force Vector store (scikit-learn) initialized with data obtained from the [practictioners guide for MLOps](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf).


```
llm = ChatVertexAI(model_name="gemini-1.5-flash-002", temperature=0)
embedding = VertexAIEmbeddings(model_name="text-embedding-004")


vector_store = get_vector_store(embedding=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 20})
compressor = VertexAIRank(
    project_id=PROJECT_ID,
    location_id="global",
    ranking_config="default_ranking_config",
    title_field="id",
    top_n=5,
)


@tool
def retrieve_docs(query: str) -> list[Document]:
    """
    Useful for retrieving relevant documents based on a query.
    Use this when you need additional information to answer a question.

    Args:
        query (str): The user's question or search query.

    Returns:
        List[Document]: A list of the top-ranked Document objects, limited to TOP_K (5) results.
    """
    retrieved_docs = retriever.invoke(query)
    ranked_docs = compressor.compress_documents(documents=retrieved_docs, query=query)
    return ranked_docs


@tool
def should_continue() -> None:
    """
    Use this tool if you determine that you have enough context to respond to the questions of the user.
    """
    return None


# Set up conversation inspector
inspect_conversation = inspect_conversation_template | llm.bind_tools(
    [retrieve_docs, should_continue], tool_choice="any"
)

# Set up response chain
response_chain = rag_template | llm


@custom_chain
async def chain(
    input: dict[str, Any], **kwargs: Any
) -> AsyncIterator[OnToolEndEvent | OnChatModelStreamEvent]:
    """
    Implement a RAG QA chain with tool calls.

    This function is decorated with `custom_chain` to offer LangChain compatible
    astream_events, support for synchronous invocation through the `invoke` method,
    and OpenTelemetry tracing.
    """
    # Inspect conversation and determine next action
    inspection_result = inspect_conversation.invoke(input)
    tool_call_result = inspection_result.tool_calls[0]

    # Execute the appropriate tool based on the inspection result
    if tool_call_result["name"] == "retrieve_docs":
        # Retrieve relevant documents
        docs = retrieve_docs.invoke(tool_call_result["args"])
        # Format the retrieved documents
        formatted_docs = template_docs.format(docs=docs)
        # Create a ToolMessage with the formatted documents
        tool_message = ToolMessage(
            tool_call_id=tool_call_result["name"],
            name=tool_call_result["name"],
            content=formatted_docs,
            artifact=docs,
        )
    else:
        # If no documents need to be retrieved, continue with the conversation
        tool_message = should_continue.invoke(tool_call_result)

    # Update input messages with new information
    input["messages"] = input["messages"] + [inspection_result, tool_message]

    # Yield tool results metadata
    yield OnToolEndEvent(
        data={"input": tool_call_result["args"], "output": tool_message}
    )

    # Stream LLM response
    async for chunk in response_chain.astream(input=input):
        yield OnChatModelStreamEvent(data={"chunk": chunk})
```

The `@custom_chain` decorator defined in `app/utils/output_types.py`:
- Enables compatibility with the `astream_events` LangChain API interface by offering a `chain.astream_events` method.
- Provides an `invoke` method for synchronous invocation. This method can be utilized for evaluation purposes.
- Adds OpenTelemetry tracing functionality.

This methodology is used for the chain defined in `app/patterns/custom_rag_qa/chain.py` file.

Let's test the custom chain we just created. 


```
input_message = {"messages": [("human", "What is MLOps?")]}

async for event in chain.astream_events(input=input_message, version="v2"):
    if event["event"] in SUPPORTED_EVENTS:
        print(event["data"])
```

## Evaluation

Evaluation is the activity of assessing the quality of the model's outputs, to gauge its understanding and success in fulfilling the prompt's instructions.

In the context of Generative AI, evaluation extends beyond the evaluation of the model's outputs to include the evaluation of the chain's outputs and in some cases the evaluation of the intermediate steps (for example, the evaluation of the retriever's outputs).

### Vertex AI Evaluation
To evaluate the chain's outputs, we'll utilize [Vertex AI Evaluation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-overview) to assess our AI application's performance. 
Vertex AI Evaluation streamlines the evaluation process for generative AI by offering three key features:

- [Pre-built Metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval): It provides a library of ready-to-use metrics for common evaluation tasks, saving you time and effort in defining your own. These metrics cover a range of areas, simplifying the assessment of different aspects of your model's performance.
 
- [Custom Metrics](https://cloud.google.com/vertex-ai/generative-ai/docs/models/determine-eval): Beyond pre-built options, Vertex AI Evaluation allows you to define and implement custom metrics tailored to your specific needs and application requirements. 
 
- Strong Integration with [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments): Vertex AI Evaluation seamlessly integrates with Vertex AI Experiments, creating a unified workflow for tracking experiments and managing evaluation results.

### Evaluation Samples

**Note**: This notebook includes a section on evaluation, but it's a placeholder which should evolve based on the needs of your app. For a set of recommended samples on evaluation please visit the [official documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/evaluation-examples).

For a comprehensive solution to perform evaluation in Vertex AI, consider leveraging [Evals Playbook](https://github.com/GoogleCloudPlatform/applied-ai-engineering-samples/tree/main/genai-on-vertex-ai/gemini/evals_playbook), which provides recipes to streamline the experimentation and evaluation process. It showcases how you can define, track, compare, and iteratively refine experiments, customize evaluation runs and metrics and log prompts and responses.


## Evaluating a chain

Let's start by defining again a simple chain:


```
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable culinary assistant specializing in providing"
            "detailed cooking recipes. Your responses should be informative, engaging, "
            "and tailored to the user's specific requests. Include ingredients, "
            "step-by-step instructions, cooking times, and any helpful tips or "
            "variations. If asked about dietary restrictions or substitutions, offer "
            "appropriate alternatives.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = template | llm
```

We then import the ground truth data we will use for evaluation. Data is stored in [`app/eval/data/chats.yaml`](../app/eval/data/chats.yaml)
Note: You might need to adjust the path depending on where your Jupyter kernel was initialized.


```
y = yaml.safe_load(open("../app/eval/data/chats.yaml"))
df = pd.DataFrame(y)
df
```

We leverage the helper functions [`generate_multiturn_history`](../app/eval/utils.py) to prepare the data for evaluation.

You can see below the documentation for the function.


```
help(generate_multiturn_history)
```


```
df = generate_multiturn_history(df)
df
```

We prepare the batch prediction input by combining conversation history with the latest human message



```
batch_prediction_input = df.copy()
batch_prediction_input["all_messages"] = batch_prediction_input.apply(
    lambda row: row.get("conversation_history", []) + [row["human_message"]], axis=1
)
batch_prediction_input = (
    batch_prediction_input["all_messages"]
    .apply(lambda messages: {"messages": messages})
    .tolist()
)
```

To perform batch scoring, we leverage the [batch](https://python.langchain.com/v0.1/docs/expression_language/interface/#batch) method available for LangChain runnables.


```
responses = chain.batch(batch_prediction_input)
responses
```

We extract the user message and the reference (ground truth) message from dataframe so that we can use them for evaluation.


```
scored_data = df
scored_data["user"] = scored_data["human_message"].apply(lambda x: x["content"])
scored_data["reference"] = scored_data["ai_message"].apply(lambda x: x["content"])
scored_data["response"] = [item.content for item in responses]
scored_data["response_obj"] = [item for item in responses]
scored_data
```


```
# Note: if using LangGraph, the code above needs to be modified to grab the last message from the message history:
# scored_data["user"] = scored_data["human_message"].apply(lambda x: x["content"])
# scored_data["reference"] = scored_data["ai_message"].apply(lambda x: x["content"])
# scored_data["response"] = [item["messages"][-1].content for item in responses]
# scored_data["response_obj"] = [item["messages"][-1] for item in responses]
# scored_data
```

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
Original source data: {instance["response"]}

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
    "custom_faithfulness": 
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


```
experiment_name = "template-langchain-eval"  # @param {type:"string"}
```

We are now ready to run the evaluation. We will use different metrics, combining the custom metric we defined above with some pre-built metrics.

Results of the evaluation will be automatically tagged into the experiment_name we define.

You can click `View Experiment`, to see the experiment in Google Cloud Console.


```
metrics = ["fluency", "safety", custom_faithfulness_metric]

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

## Next Steps

Congratulations on completing the getting started tutorial! You've learned different methodologies to build a chain and how to evaluate it. 
Let's explore the next steps in your journey:

### 1. Prepare for Production

Once you're satisfied with your chain's evaluation results:

1. Write your chain into the [`app/chain.py` file](../app/chain.py).
2. Remove the `patterns` folder and its associated tests (these are for demonstration only).

### 2. Local Testing

Test your chain using the playground:

```bash
make playground
```

This launches af feature-rich playground, including chat curation, user feedback collection, multimodal input, and more!


### 3. Production Deployment

Once you are satisfied with the results, you can setup your CI/CD pipelines to deploy your chain to production.

Please refer to the [deployment guide](../deployment/README.md) for more information on how to do that.

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
