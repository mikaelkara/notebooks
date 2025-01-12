# Prompt Generation from User Requirements

In this example we will create a chat bot that helps a user generate a prompt.
It will first collect requirements from the user, and then will generate the prompt (and refine it based on user input).
These are split into two separate states, and the LLM decides when to transition between them.

A graphical representation of the system can be found below.

![prompt-generator.png](18f6888d-c412-4c53-ac3c-239fb90d2b6c.png)

## Setup

First, let's install our required packages and set our OpenAI API key (the LLM we will use)


```python
%%capture --no-stderr
% pip install -U langgraph langchain_openai
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

## Gather information

First, let's define the part of the graph that will gather user requirements. This will be an LLM call with a specific system message. It will have access to a tool that it can call when it is ready to generate the prompt.

<div class="admonition note">
    <p class="admonition-title">Using Pydantic with LangChain</p>
    <p>
        This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
    </p>
</div>


```python
from typing import List

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel
```


```python
template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


llm = ChatOpenAI(temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])


def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}
```

## Generate Prompt

We now set up the state that will generate the prompt.
This will require a separate system message, as well as a function to filter out all message PRIOR to the tool invocation (as that is when the previous state decided it was time to generate the prompt


```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# New system prompt
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""


# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}
```

## Define the state logic

This is the logic for what state the chatbot is in.
If the last message is a tool call, then we are in the state where the "prompt creator" (`prompt`) should respond.
Otherwise, if the last message is not a HumanMessage, then we know the human should respond next and so we are in the `END` state.
If the last message is a HumanMessage, then if there was a tool call previously we are in the `prompt` state.
Otherwise, we are in the "info gathering" (`info`) state.


```python
from typing import Literal

from langgraph.graph import END


def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"
```

## Create the graph

We can now the create the graph.
We will use a SqliteSaver to persist conversation history.


```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]


memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state: State):
    return {
        "messages": [
            ToolMessage(
                content="Prompt generated!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }


workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)
```


```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```


    
![jpeg](output_15_0.jpg)
    


## Use the graph

We can now use the created chatbot.


```python
import uuid

cached_human_responses = ["hi!", "rag prompt", "1 rag, 2 none, 3 no, 4 no", "red", "q"]
cached_response_index = 0
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
while True:
    try:
        user = input("User (q/Q to quit): ")
    except:
        user = cached_human_responses[cached_response_index]
        cached_response_index += 1
    print(f"User (q/Q to quit): {user}")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break
    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")
```

    User (q/Q to quit): hi!
    ==================================[1m Ai Message [0m==================================
    
    Hello! How can I assist you today?
    User (q/Q to quit): rag prompt
    ==================================[1m Ai Message [0m==================================
    
    Sure! I can help you create a prompt template. To get started, could you please provide me with the following information:
    
    1. What is the objective of the prompt?
    2. What variables will be passed into the prompt template?
    3. Any constraints for what the output should NOT do?
    4. Any requirements that the output MUST adhere to?
    
    Once I have this information, I can assist you in creating the prompt template.
    User (q/Q to quit): 1 rag, 2 none, 3 no, 4 no
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      PromptInstructions (call_tcz0foifsaGKPdZmsZxNnepl)
     Call ID: call_tcz0foifsaGKPdZmsZxNnepl
      Args:
        objective: rag
        variables: ['none']
        constraints: ['no']
        requirements: ['no']
    =================================[1m Tool Message [0m=================================
    
    Prompt generated!
    ==================================[1m Ai Message [0m==================================
    
    Please write a response using the RAG (Red, Amber, Green) rating system.
    Done!
    User (q/Q to quit): red
    ==================================[1m Ai Message [0m==================================
    
    Response: The status is RED.
    User (q/Q to quit): q
    AI: Byebye
    
