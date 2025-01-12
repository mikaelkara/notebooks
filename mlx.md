# MLX

This notebook shows how to get started using `MLX` LLM's as chat models.

In particular, we will:
1. Utilize the [MLXPipeline](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/mlx_pipeline.py), 
2. Utilize the `ChatMLX` class to enable any of these LLMs to interface with LangChain's [Chat Messages](https://python.langchain.com/docs/modules/model_io/chat/#messages) abstraction.
3. Demonstrate how to use an open-source LLM to power an `ChatAgent` pipeline



```python
%pip install --upgrade --quiet  mlx-lm transformers huggingface_hub
```

## 1. Instantiate an LLM

There are three LLM options to choose from.


```python
from langchain_community.llms.mlx_pipeline import MLXPipeline

llm = MLXPipeline.from_model_id(
    "mlx-community/quantized-gemma-2b-it",
    pipeline_kwargs={"max_tokens": 10, "temp": 0.1},
)
```

## 2. Instantiate the `ChatMLX` to apply chat templates

Instantiate the chat model and some messages to pass.


```python
from langchain_community.chat_models.mlx import ChatMLX
from langchain_core.messages import HumanMessage

messages = [
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

chat_model = ChatMLX(llm=llm)
```

Inspect how the chat messages are formatted for the LLM call.


```python
chat_model._to_chat_prompt(messages)
```

Call the model.


```python
res = chat_model.invoke(messages)
print(res.content)
```

## 3. Take it for a spin as an agent!

Here we'll test out `gemma-2b-it` as a zero-shot `ReAct` Agent. The example below is taken from [here](https://python.langchain.com/docs/modules/agents/agent_types/react#using-chat-models).

> Note: To run this section, you'll need to have a [SerpAPI Token](https://serpapi.com/) saved as an environment variable: `SERPAPI_API_KEY`


```python
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper
```

Configure the agent with a `react-json` style prompt and access to a search engine and calculator.


```python
# setup tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```


```python
agent_executor.invoke(
    {
        "input": "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
    }
)
```
