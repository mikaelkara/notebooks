# MLflow

>[MLflow](https://mlflow.org/) is a versatile, open-source platform for managing workflows and artifacts across the machine learning lifecycle. It has built-in integrations with many popular ML libraries, but can be used with any library, algorithm, or deployment tool. It is designed to be extensible, so you can write plugins to support new workflows, libraries, and tools.

In the context of LangChain integration, MLflow provides the following capabilities:

- **Experiment Tracking**: MLflow tracks and stores artifacts from your LangChain experiments, including models, code, prompts, metrics, and more.
- **Dependency Management**: MLflow automatically records model dependencies, ensuring consistency between development and production environments.
- **Model Evaluation** MLflow offers native capabilities for evaluating LangChain applications.
- **Tracing**: MLflow allows you to visually trace data flows through your LangChain chain, agent, retriever, or other components.


**Note**: The tracing capability is only available in MLflow versions 2.14.0 and later.

This notebook demonstrates how to track your LangChain experiments using MLflow. For more information about this feature and to explore tutorials and examples of using LangChain with MLflow, please refer to the [MLflow documentation for LangChain integration](https://mlflow.org/docs/latest/llms/langchain/index.html).

## Setup

Install MLflow Python package:


```python
%pip install google-search-results num
```


```python
%pip install mlflow -qU
```

This example utilizes the OpenAI LLM. Feel free to skip the command below and proceed with a different LLM if desired.


```python
%pip install langchain-openai -qU
```


```python
import os

# Set MLflow tracking URI if you have MLflow Tracking Server running
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["OPENAI_API_KEY"] = ""
```

To begin, let's create a dedicated MLflow experiment in order track our model and artifacts. While you can opt to skip this step and use the default experiment, we strongly recommend organizing your runs and artifacts into separate experiments to avoid clutter and maintain a clean, structured workflow.


```python
import mlflow

mlflow.set_experiment("LangChain MLflow Integration")
```

## Overview

Integrate MLflow with your LangChain Application using one of the following methods:

1. **Autologging**: Enable seamless tracking with the `mlflow.langchain.autolog()` command, our recommended first option for leveraging the LangChain MLflow integration.
2. **Manual Logging**: Use MLflow APIs to log LangChain chains and agents, providing fine-grained control over what to track in your experiment.
3. **Custom Callbacks**: Pass MLflow callbacks manually when invoking chains, allowing for semi-automated customization of your workload, such as tracking specific invocations.

## Scenario 1: MLFlow Autologging

To get started with autologging, simply call `mlflow.langchain.autolog()`. In this example, we set the `log_models` parameter to `True`, which allows the chain definition and its dependency libraries to be recorded as an MLflow model, providing a comprehensive tracking experience.


```python
import mlflow

mlflow.langchain.autolog(
    # These are optional configurations to control what information should be logged automatically (default: False)
    # For the full list of the arguments, refer to https://mlflow.org/docs/latest/llms/langchain/autologging.html#id1
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True,
)
```

### Define a Chain


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)
parser = StrOutputParser()

chain = prompt | llm | parser
```

### Invoke the Chain

Note that this step may take a few seconds longer than usual, as MLflow runs several background tasks in the background to log models, traces, and artifacts to the tracking server.


```python
test_input = {
    "input_language": "English",
    "output_language": "German",
    "input": "I love programming.",
}

chain.invoke(test_input)
```




    'Ich liebe das Programmieren.'



Take a moment to explore the MLflow Tracking UI, where you can gain a deeper understanding of what information are being logged.
* **Traces** -  Navigate to the "Traces" tab in the experiment and click the request ID link of the first row. The displayed trace tree visualizes the call stack of your chain invocation, providing you with a deep insight into how each component is executed within the chain.
* **MLflow Model** - As we set `log_model=True`, MLflow automatically creates an MLflow Run to track your chain definition. Navigate to the newest Run page and open the "Artifacts" tab, which lists file artifacts logged as an MLflow Model, including dependencies, input examples, model signatures, and more.


### Invoke the Logged Chain

Next, let's load the model back and verify that we can reproduce the same prediction, ensuring consistency and reliability.

There are two ways to load the model
1. `mlflow.langchain.load_model(MODEL_URI)` - This loads the model as the original LangChain object.
2. `mlflow.pyfunc.load_model(MODEL_URI)` - This loads the model within the `PythonModel` wrapper and encapsulates the prediction logic with the `predict()` API, which contains additional logic such as schema enforcement.


```python
# Replace YOUR_RUN_ID with the Run ID displayed on the MLflow UI
loaded_model = mlflow.langchain.load_model("runs:/{YOUR_RUN_ID}/model")
loaded_model.invoke(test_input)
```




    'Ich liebe Programmieren.'




```python
pyfunc_model = mlflow.pyfunc.load_model("runs:/{YOUR_RUN_ID}/model")
pyfunc_model.predict(test_input)
```




    ['Ich liebe das Programmieren.']



### Configure Autologging

The `mlflow.langchain.autolog()` function offers several parameters that allow for fine-grained control over the artifacts logged to MLflow. For a comprehensive list of available configurations, please refer to the latest [MLflow LangChain Autologging Documentation](https://mlflow.org/docs/latest/llms/langchain/autologging.html).

## Scenario 2: Manually Logging an Agent from Code


#### Prerequisites

This example uses `SerpAPI`, a search engine API, as a tool for the agent to retrieve Google Search results. LangChain is natively integrated with `SerpAPI`, allowing you to configure the tool for your agent with just one line of code.

To get started:

* Install the required Python package via pip: `pip install google-search-results numexpr`.
* Create an account at [SerpAPI's Official Website](https://serpapi.com/) and retrieve an API key.
* Set the API key in the environment variable: `os.environ["SERPAPI_API_KEY"] = "YOUR_API_KEY"`


### Define an Agent

In this example, we will log the agent definition **as code**, rather than directly feeding the Python object and saving it in a serialized format. This approach offers several benefits:

1. **No serialization required**: By saving the model as code, we avoid the need for serialization, which can be problematic when working with components that don't natively support it. This approach also eliminates the risk of incompatibility issues when deserializing the model in a different environment.
2. **Better transparency**: By inspecting the saved code file, you can gain valuable insights into what the model does. This is in contrast to serialized formats like pickle, where the model's behavior remains opaque until it's loaded back, potentially exposing security risks such as remote code execution.


First, create a separate `.py` file that defines the agent instance.

In the interest of time, you can run the following cell to generate a Python file `agent.py`, which contains the agent definition code. In actual dev scenario, you would define it in another notebook or hand-crafted python script.


```python
script_content = """
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI
import mlflow

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# IMPORTANT: call set_model() to register the instance to be logged.
mlflow.models.set_model(agent)
"""

with open("agent.py", "w") as f:
    f.write(script_content)
```

### Log the Agent

Return to the original notebook and run the following cell to log the agent you've defined in the `agent.py` file.



```python
question = "How long would it take to drive to the Moon with F1 racing cars?"

with mlflow.start_run(run_name="search-math-agent") as run:
    info = mlflow.langchain.log_model(
        lc_model="agent.py",  # Specify the relative code path to the agent definition
        artifact_path="model",
        input_example=question,
    )

print("The agent is successfully logged to MLflow!")
```

    The agent is successfully logged to MLflow!
    

Now, open the MLflow UI and navigate to the "Artifacts" tab in the Run detail page. You should see that the `agent.py` file has been successfully logged, along with other model artifacts, such as dependencies, input examples, and more.

### Invoke the Logged Agent

Now load the agent back and invoke it. There are two ways to load the model


```python
# Let's turn on the autologging with default configuration, so we can see the trace for the agent invocation.
mlflow.langchain.autolog()

# Load the model back
agent = mlflow.pyfunc.load_model(info.model_uri)

# Invoke
agent.predict(question)
```

    Downloading artifacts: 100%|██████████| 10/10 [00:00<00:00, 331.57it/s]
    




    ['It would take approximately 1194.5 hours to drive to the Moon with an F1 racing car.']



Navigate to the **"Traces"** tab in the experiment and click the request ID link of the first row. The trace visualizes how the agent operate multiple tasks within the single prediction call:
1. Determine what subtasks are required to answer the questions.
2. Search for the speed of an F1 racing car.
3. Search for the distance from Earth to Moon.
4. Compute the division using LLM.

## Scenario 3. Using MLflow Callbacks

**MLflow Callbacks** provide a semi-automated way to track your LangChain application in MLflow. There are two primary callbacks available:

1. **`MlflowLangchainTracer`:** Primarily used for generating traces, available in `mlflow >= 2.14.0`.
2. **`MLflowCallbackHandler`:** Logs metrics and artifacts to the MLflow tracking server.

### MlflowLangchainTracer

When the chain or agent is invoked with the `MlflowLangchainTracer` callback, MLflow will automatically generate a trace for the call stack and log it to the MLflow tracking server.  The outcome is exactly same as `mlflow.langchain.autolog()`, but this is particularly useful when you want to only trace specific invocation. Autologging is applied to all invocation in the same notebook/script, on the other hand.


```python
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

mlflow_tracer = MlflowLangchainTracer()

# This call generates a trace
chain.invoke(test_input, config={"callbacks": [mlflow_tracer]})

# This call does not generate a trace
chain.invoke(test_input)
```

#### Where to Pass the Callback
 LangChain supports two ways of passing callback instances: (1) Request time callbacks - pass them to the `invoke` method or bind with `with_config()` (2) Constructor callbacks - set them in the chain constructor. When using the `MlflowLangchainTracer` as a callback, you **must use request time callbacks**. Setting it in the constructor instead will only apply the callback to the top-level object, preventing it from being propagated to child components, resulting in incomplete traces. For more information on this behavior, please refer to [Callbacks Documentation](https://python.langchain.com/docs/concepts/callbacks) for more details.

```python
# OK
chain.invoke(test_input, config={"callbacks": [mlflow_tracer]})
chain.with_config(callbacks=[mlflow_tracer])
# NG
chain = TheNameOfSomeChain(callbacks=[mlflow_tracer])
```

#### Supported Methods

`MlflowLangchainTracer` supports the following invocation methods from the [Runnable Interfaces](https://python.langchain.com/v0.1/docs/expression_language/interface/).
-  Standard interfaces: `invoke`, `stream`, `batch`
-  Async interfaces: `astream`, `ainvoke`, `abatch`, `astream_log`, `astream_events`

Other methods are not guaranteed to be fully compatible.

### MlflowCallbackHandler

`MlflowCallbackHandler` is a callback handler that resides in the LangChain Community code base.

This callback can be passed for chain/agent invocation, but it must be explicitly finished by calling the `flush_tracker()` method.

When a chain is invoked with the callback, it performs the following actions:

1. Creates a new MLflow Run or retrieves an active one if available within the active MLflow Experiment.
2. Logs metrics such as the number of LLM calls, token usage, and other relevant metrics. If the chain/agent includes LLM call and you have `spacy` library installed, it logs text complexity metrics such as `flesch_kincaid_grade`.
3. Logs internal steps as a JSON file (this is a legacy version of traces).
4. Logs chain input and output as a Pandas Dataframe.
5. Calls the `flush_tracker()` method with a chain/agent instance, logging the chain/agent as an MLflow Model.



```python
from langchain_community.callbacks import MlflowCallbackHandler

mlflow_callback = MlflowCallbackHandler()

chain.invoke("What is LangChain callback?", config={"callbacks": [mlflow_callback]})

mlflow_callback.flush_tracker()
```

## References
To learn more about the feature and visit tutorials and examples of using LangChain with MLflow, please refer to the [MLflow documentation for LangChain integration](https://mlflow.org/docs/latest/llms/langchain/index.html).

`MLflow` also provides several [tutorials](https://mlflow.org/docs/latest/llms/langchain/index.html#getting-started-with-the-mlflow-langchain-flavor-tutorials-and-guides) and [examples](https://github.com/mlflow/mlflow/tree/master/examples/langchain) for the `LangChain` integration:
- [Quick Start](https://mlflow.org/docs/latest/llms/langchain/notebooks/langchain-quickstart.html)
- [RAG Tutorial](https://mlflow.org/docs/latest/llms/langchain/notebooks/langchain-retriever.html)
- [Agent Example](https://github.com/mlflow/mlflow/blob/master/examples/langchain/simple_agent.py)
