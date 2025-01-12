### Agents with Langchain integration with HuggingFace Chat models

Trying an example from this [HuffingFace Agents blog](https://huggingface.co/blog/open-source-llms-as-agents)

As a reference to ReAct prompting, see examples in the
[llm-prompts/7_how_to_use_react_prompt](../llm-prompts/7_how_to_use_react_prompt.ipynb) notebook using
in-context or few-short learning.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png">


```python
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace
```


```python
from dotenv import load_dotenv, find_dotenv
import warnings
import os
```


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
```


```python
from huggingface_hub import login
login()
```


    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…



```python
os.environ["HF_TOKEN"] = HF_TOKEN
```


```python
llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="mistralai/Mistral-7B-v0.1",
    task="text-generation",
     model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03
    }
)
```


```python
chat_model = ChatHuggingFace(llm=llm)
```


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


```python
# setup tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)
```


```python
# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
```


```python
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
```


```python
# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "Who is the current holder of the speed skating world record on 500 meters? What is her current age raised to the 0.43 power?"
    }
)
```
