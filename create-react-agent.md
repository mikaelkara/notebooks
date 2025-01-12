# How to use the prebuilt ReAct agent

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/">
                    Agent Architectures
                </a>                   
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#chat-models/">
                    Chat Models
                </a>
            </li>
            <li>
                <a href="https://python.langchain.com/docs/concepts/#tools">
                    Tools
                </a>
            </li>
        </ul>
    </p>
</div> 

In this how-to we'll create a simple [ReAct](https://arxiv.org/abs/2210.03629) agent app that can check the weather. The app consists of an agent (LLM) and tools. As we interact with the app, we will first call the agent (LLM) to decide if we should use tools. Then we will run a loop:  

1. If the agent said to take an action (i.e. call tool), we'll run the tools and pass the results back to the agent
2. If the agent did not ask to run tools, we will finish (respond to the user)

<div class="admonition warning">
    <p class="admonition-title">Prebuilt Agent</p>
    <p>
Please note that here will we use <a href="https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent">a prebuilt agent</a>. One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.
    </p>
</div>   

## Setup

First let's install the required packages and set our API keys


```python
%%capture --no-stderr
%pip install -U langgraph langchain-openai
```


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Code


```python
# First we initialize the model we want to use.
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)


# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)

from typing import Literal

from langchain_core.tools import tool


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]


# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(model, tools=tools)
```

## Usage

First, let's visualize the graph we just created


```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](output_9_0.jpg)
    



```python
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
```

Let's run the app with an input that needs a tool call


```python
inputs = {"messages": [("user", "what is the weather in sf")]}
print_stream(graph.stream(inputs, stream_mode="values"))
```

    ================================[1m Human Message [0m=================================
    
    what is the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_zVvnU9DKr6jsNnluFIl59mHb)
     Call ID: call_zVvnU9DKr6jsNnluFIl59mHb
      Args:
        city: sf
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is currently sunny.
    

Now let's try a question that doesn't need tools


```python
inputs = {"messages": [("user", "who built you?")]}
print_stream(graph.stream(inputs, stream_mode="values"))
```

    ================================[1m Human Message [0m=================================
    
    who built you?
    ==================================[1m Ai Message [0m==================================
    
    I was created by OpenAI, a research organization focused on developing and advancing artificial intelligence technology.
    
