# How to add memory to the prebuilt ReAct agent

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>            
                <a href="https://langchain-ai.github.io/langgraph/concepts/persistence/">
                    LangGraph Persistence
                </a>
            </li>
            <li>            
                <a href="https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpointer-interface">
                    Checkpointer interface
                </a>
            </li>
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

This guide will show how to add memory to the prebuilt ReAct agent. Please see [this tutorial](../create-react-agent) for how to get started with the prebuilt ReAct agent

We can add memory to the agent, by passing a [checkpointer](https://langchain-ai.github.io/langgraph/reference/checkpoints/) to the [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) function.

## Setup

First, let's install the required packages and set our API keys


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

# We can add "chat memory" to the graph with LangGraph's checkpointer
# to retain the chat context between interactions
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Define the graph

from langgraph.prebuilt import create_react_agent

graph = create_react_agent(model, tools=tools, checkpointer=memory)
```

## Usage

Let's interact with it multiple times to show that it can remember


```python
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
```


```python
config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [("user", "What's the weather in NYC?")]}

print_stream(graph.stream(inputs, config=config, stream_mode="values"))
```

    ================================[1m Human Message [0m=================================
    
    What's the weather in NYC?
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_xM1suIq26KXvRFqJIvLVGfqG)
     Call ID: call_xM1suIq26KXvRFqJIvLVGfqG
      Args:
        city: nyc
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It might be cloudy in nyc
    ==================================[1m Ai Message [0m==================================
    
    The weather in NYC might be cloudy.
    

Notice that when we pass the same the same thread ID, the chat history is preserved


```python
inputs = {"messages": [("user", "What's it known for?")]}
print_stream(graph.stream(inputs, config=config, stream_mode="values"))
```

    ================================[1m Human Message [0m=================================
    
    What's it known for?
    ==================================[1m Ai Message [0m==================================
    
    New York City (NYC) is known for a variety of iconic landmarks, cultural institutions, and vibrant neighborhoods. Some of the most notable aspects include:
    
    1. **Statue of Liberty**: A symbol of freedom and democracy.
    2. **Times Square**: Known for its bright lights, Broadway theaters, and bustling atmosphere.
    3. **Central Park**: A large urban park offering a green oasis in the middle of the city.
    4. **Empire State Building**: An iconic skyscraper with an observation deck offering panoramic views of the city.
    5. **Broadway**: Famous for its world-class theater productions.
    6. **Wall Street**: The financial hub of the United States.
    7. **Museums**: Including the Metropolitan Museum of Art, the Museum of Modern Art (MoMA), and the American Museum of Natural History.
    8. **Diverse Cuisine**: A melting pot of culinary experiences from around the world.
    9. **Cultural Diversity**: A rich tapestry of cultures, languages, and traditions.
    10. **Fashion**: A global fashion capital, home to New York Fashion Week.
    
    These are just a few highlights of what makes NYC a unique and vibrant city.
    


```python

```
