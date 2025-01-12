[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iRf6_ZXi2yPemZSJr2bMi4qQ7PM851ok?usp=sharing)

# Introduction - Fireworks x LangGraph

In this notebook, we demonstrate how to use the Fireworks function-calling model as a router across multiple models with specialized capabilities. Function-calling models have seen a rapid rise in usage due to their ability to easily utilize external tools. One such powerful tool is other LLMs. We have a variety of specialized OSS LLMs for [coding](https://www.deepseek.com/), [chatting in certain languages](https://github.com/QwenLM/Qwen), or just plain [HF Assistants](https://huggingface.co/chat/assistants).

The function-calling model allows us to:
1. Analyze the user query for intent.
2. Find the best model to answer the request, which could even be the function-calling model itself!
3. Construct the right query for the chosen LLM.
4. Profit!

This notebook is a sister notebook to LangChain, though we will use the newer and more controllable[LangGraph](https://www.langchain.com/langgraph) framework to construct an agent graph capable of chit-chatting and solving math equations using a calculator tool.

This agent chain will utilize [custom-defined tools](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/) capable of executing a math query using an LLM. The main routing LLM will be the Fireworks function-calling model.



---




```python
!pip3 install langgraph langchain-fireworks
```

    Requirement already satisfied: langgraph in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.2.45)
    Requirement already satisfied: langchain-fireworks in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.2.5)
    Requirement already satisfied: langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langgraph) (0.3.15)
    Requirement already satisfied: langgraph-checkpoint<3.0.0,>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langgraph) (2.0.2)
    Requirement already satisfied: langgraph-sdk<0.2.0,>=0.1.32 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langgraph) (0.1.35)
    Requirement already satisfied: aiohttp<4.0.0,>=3.9.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-fireworks) (3.10.5)
    Requirement already satisfied: fireworks-ai>=0.13.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-fireworks) (0.15.3)
    Requirement already satisfied: openai<2.0.0,>=1.10.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-fireworks) (1.47.0)
    Requirement already satisfied: requests<3,>=2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-fireworks) (2.32.3)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks) (2.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks) (24.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks) (6.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks) (1.11.1)
    Requirement already satisfied: httpx in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai>=0.13.0->langchain-fireworks) (0.27.2)
    Requirement already satisfied: httpx-sse in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai>=0.13.0->langchain-fireworks) (0.4.0)
    Requirement already satisfied: pydantic in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai>=0.13.0->langchain-fireworks) (2.9.2)
    Requirement already satisfied: Pillow in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai>=0.13.0->langchain-fireworks) (10.4.0)
    Requirement already satisfied: PyYAML>=5.3 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (6.0.2)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (1.33)
    Requirement already satisfied: langsmith<0.2.0,>=0.1.125 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (0.1.125)
    Requirement already satisfied: packaging<25,>=23.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (23.2)
    Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (8.5.0)
    Requirement already satisfied: typing-extensions>=4.7 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (4.12.2)
    Requirement already satisfied: msgpack<2.0.0,>=1.1.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langgraph-checkpoint<3.0.0,>=2.0.0->langgraph) (1.1.0)
    Requirement already satisfied: orjson>=3.10.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from langgraph-sdk<0.2.0,>=0.1.32->langgraph) (3.10.7)
    Requirement already satisfied: anyio<5,>=3.5.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai<2.0.0,>=1.10.0->langchain-fireworks) (4.6.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai<2.0.0,>=1.10.0->langchain-fireworks) (1.9.0)
    Requirement already satisfied: jiter<1,>=0.4.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai<2.0.0,>=1.10.0->langchain-fireworks) (0.4.2)
    Requirement already satisfied: sniffio in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai<2.0.0,>=1.10.0->langchain-fireworks) (1.3.1)
    Requirement already satisfied: tqdm>4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai<2.0.0,>=1.10.0->langchain-fireworks) (4.66.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests<3,>=2->langchain-fireworks) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests<3,>=2->langchain-fireworks) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests<3,>=2->langchain-fireworks) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests<3,>=2->langchain-fireworks) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx->fireworks-ai>=0.13.0->langchain-fireworks) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpcore==1.*->httpx->fireworks-ai>=0.13.0->langchain-fireworks) (0.14.0)
    Requirement already satisfied: jsonpointer>=1.9 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from jsonpatch<2.0,>=1.33->langchain-core!=0.3.0,!=0.3.1,!=0.3.10,!=0.3.11,!=0.3.12,!=0.3.13,!=0.3.14,!=0.3.2,!=0.3.3,!=0.3.4,!=0.3.5,!=0.3.6,!=0.3.7,!=0.3.8,!=0.3.9,<0.4.0,>=0.2.43->langgraph) (3.0.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic->fireworks-ai>=0.13.0->langchain-fireworks) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic->fireworks-ai>=0.13.0->langchain-fireworks) (2.23.4)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    

## Setup Dependencies

To accomplish the task in this notebook, we need to import some dependencies from LangChain to interface with the model. Specifically, we will use the [ChatFireworks](https://python.langchain.com/v0.2/docs/integrations/chat/fireworks/) implementation.

For solving our math equations, we will use the recently released [Firefunction V2](https://fireworks.ai/blog/firefunction-v2-launch-post) and interface with it using the [Fireworks Inference Service](https://fireworks.ai/models).

To use the Fireworks AI function-calling model, you must first obtain Fireworks API keys. If you don't already have one, you can get one by following the instructions [here](https://readme.fireworks.ai/docs/quickstart). When prompted below paste in your `FIREWORKS_API_KEY`.

**NOTE:** It's important to set the temperature to 0.0 for the function-calling model because we want reliable behavior in routing.



```python
import getpass
import os

from langchain_fireworks import ChatFireworks

# Replace 'YOUR_API_KEY_HERE' with your actual API key
FIREWORKS_API_KEY = "<YOUR_FIREWORKS_API_KEY>"

# Initialize a Fireworks chat model
llm = ChatFireworks(
  model="accounts/fireworks/models/firefunction-v2",
  temperature=0.0,
  max_tokens=256
  )
```

## Base LangGraph


```python
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

# This is the default state  same as "MessageState" TypedDict but allows us accessibility to
# custom keys to our state like user's details
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # user_id: int

graph = StateGraph(GraphsState)

def _call_model(state: GraphsState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph.add_edge(START, "modelNode")
graph.add_node("modelNode", _call_model)
graph.add_edge("modelNode", END)

graph_runnable = graph.compile()
```


```python
#We can visualize it using Mermaid
from IPython.display import Image, display

try:
    display(Image(graph_runnable.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_7_0.jpg)
    



```python
# simple Fireworks x LangGraph implementation
resp = graph_runnable.invoke({"messages": HumanMessage("What is your name?")})
resp["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    I'm an AI assistant, and I don't have a personal name. I'm here to help you with any questions or tasks you may have.
    

## Custom Tools

To seamlessly use the function-calling ability of the models, we can utilize the [Tool Node](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/) functionality built into LangGraph. This allows the model to select the appropriate tool based on the given options.

For this notebook, we will construct an `area_of_circle` function, which will use an LLM to calculate and return the area of a circle given a radius `r`.



```python
import math

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

@tool
def get_weather(location: str):
    """Call to get the fake current weather"""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

@tool
def area_of_circle(r: float):
    """Call to get the area of a circle in units squared"""
    return math.pi * r * r


tools = [get_weather, area_of_circle]
tool_node = ToolNode(tools)

model_with_tools = llm.bind_tools(tools)
```

now let's adjust the graph to include the ToolNode


```python
from typing import Literal
from langgraph.graph import START, END, StateGraph, MessagesState

#note for clarity, I am treating this cell as if `Base LangGraph` was not instantiated, but `Setup Dep` was.

def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_handler(state: MessagesState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(MessagesState)

workflow.add_edge(START, "modelNode")
workflow.add_node("modelNode", call_model)
workflow.add_conditional_edges(
    "modelNode",
    tool_handler,
)
workflow.add_node("tools", tool_node)
workflow.add_edge("tools", "modelNode")

app = workflow.compile()
```


```python
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_13_0.jpg)
    


### Test Chit Chat Ability

As we outlined in the beginning, the model should be able to both chit-chat, route queries to external tools when necessary or answer from internal knowledge.

Let's first start with a question that can be answered from internal knowledge.


```python
from langchain_core.messages import HumanMessage
app.invoke({"messages": HumanMessage("Who was the first President of the USA?")})["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    George Washington was the first President of the United States.
    

## Test Area of Circle and Weather in SF

Now let's test it's ability to detect a area of circle or weather questions & route the query accordingly.


```python
from langchain_core.messages import HumanMessage
while True:
  user = input("User (q/Q to quit): ")
  if user in {"q", "Q"}:
    print("AI: Byebye")
    break
  for output in app.stream({"messages": HumanMessage(user)}, stream_mode="updates"):
    last_message = next(iter(output.values()))['messages'][-1]
    last_message.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    I'm here to help! What would you like to know?
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_Vee0KBC1mhiypoeH6sWoyUYg)
     Call ID: call_Vee0KBC1mhiypoeH6sWoyUYg
      Args:
        location: SF
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's 60 degrees and foggy.
    ==================================[1m Ai Message [0m==================================
    
    It's 60 degrees and foggy.
    ==================================[1m Ai Message [0m==================================
    
    The weather in Berkeley in the spring can be quite pleasant, with average highs in the mid-60s to low 70s Fahrenheit (18-22°C). However, it can still be a bit chilly in the mornings and evenings, so it's a good idea to pack layers.
    
    For a wedding, you'll likely want to dress up a bit, so consider packing:
    
    * A dress or suit in a lightweight, breathable fabric (such as cotton or linen)
    * A light jacket or sweater for cooler moments
    * Comfortable shoes (you'll likely be standing and dancing for several hours)
    * A hat or fascinator to add a touch of elegance
    * A small umbrella or raincoat (spring showers are always a possibility)
    
    Of course, the specific dress code and weather forecast will depend on the wedding and the time of year, so be sure to check with the wedding party or the weather forecast before you pack.
    Tool Calls:
      get_weather (call_oB7GEbauiTaujQwReBZZQqvN)
     Call ID: call_oB7GEbauiTaujQwReBZZQqvN
      Args:
        location: Berkeley
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's 90 degrees and sunny.
    ==================================[1m Ai Message [0m==================================
    
    In that case, you may want to pack lighter, breathable clothing that will keep you cool in the warm weather. A lightweight dress or suit, a pair of sunglasses, and a hat to protect your face and head from the sun would be good choices. You may also want to consider packing a light scarf or shawl to add a touch of elegance to your outfit.
    
    Remember to stay hydrated by bringing a refillable water bottle with you, and don't forget to apply sunscreen to protect your skin from the sun's strong rays.
    
    I hope this helps, and I wish you a wonderful time at the wedding!
    ==================================[1m Ai Message [0m==================================
    
    I'm here to help! What would you like to know?
    ==================================[1m Ai Message [0m==================================
    
    I'm here to help! What would you like to know?
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[22], line 3
          1 from langchain_core.messages import HumanMessage
          2 while True:
    ----> 3   user = input("User (q/Q to quit): ")
          4   if user in {"q", "Q"}:
          5     print("AI: Byebye")
    

    File /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/ipykernel/kernelbase.py:1282, in Kernel.raw_input(self, prompt)
       1280     msg = "raw_input was called, but this frontend does not support input requests."
       1281     raise StdinNotImplementedError(msg)
    -> 1282 return self._input_request(
       1283     str(prompt),
       1284     self._parent_ident["shell"],
       1285     self.get_parent("shell"),
       1286     password=False,
       1287 )
    

    File /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/ipykernel/kernelbase.py:1325, in Kernel._input_request(self, prompt, ident, parent, password)
       1322 except KeyboardInterrupt:
       1323     # re-raise KeyboardInterrupt, to truncate traceback
       1324     msg = "Interrupted by user"
    -> 1325     raise KeyboardInterrupt(msg) from None
       1326 except Exception:
       1327     self.log.warning("Invalid Message:", exc_info=True)
    

    KeyboardInterrupt: Interrupted by user


# Conclusion

The fireworks function calling model can route request to external tools or internal knowledge appropriately - thus helping developers build co-operative agents.
