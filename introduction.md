# LangGraph Quick Start

In this comprehensive quick start, we will build a support chatbot in LangGraph that can:

- Answer common questions by searching the web
- Maintain conversation state across calls
- Route complex queries to a human for review
- Use custom state to control its behavior
- Rewind and explore alternative conversation paths

We'll start with a basic chatbot and progressively add more sophisticated capabilities, introducing key LangGraph concepts along the way.

## Setup

First, install the required packages:


```python
%%capture --no-stderr
%pip install -U langgraph langsmith langchain_anthropic
```

Next, set your API keys:


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

    ANTHROPIC_API_KEY:  ········
    

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Part 1: Build a Basic Chatbot

We'll first create a simple chatbot using LangGraph. This chatbot will respond directly to user messages. Though simple, it will illustrate the core concepts of building with LangGraph. By the end of this section, you will have a built rudimentary chatbot.

Start by creating a `StateGraph`. A `StateGraph` object defines the structure of our chatbot as a "state machine". We'll add `nodes` to represent the llm and functions our chatbot can call and `edges` to specify how the bot should transition between these functions.


```python
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
```

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
    The first thing you do when you define a graph is define the <code>State</code> of the graph. The <code>State</code> consists of the schema of the graph as well as <a href="https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers">reducer functions</a> which specify how to apply updates to the state. In our example <code>State</code> is a <code>TypedDict</code> with a single key: <code>messages</code>. The <code>messages</code> key is annotated with the <a href="https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages"><code>add_messages</code></a> reducer function, which tells LangGraph to append new messages to the existing list, rather than overwriting it. State keys without an annotation will be overwritten by each update, storing the most recent value. Check out <a href="https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages">this conceptual guide</a> to learn more about state, reducers and other low-level concepts.
    </p>
</div>

So now our graph knows two things:

1. Every `node` we define will receive the current `State` as input and return a value that updates that state.
2. `messages` will be _appended_ to the current list, rather than directly overwritten. This is communicated via the prebuilt [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages) function in the `Annotated` syntax.

Next, add a "`chatbot`" node. Nodes represent units of work. They are typically regular python functions.


```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
```

**Notice** how the `chatbot` node function takes the current `State` as input and returns a dictionary containing an updated `messages` list under the key "messages". This is the basic pattern for all LangGraph node functions.

The `add_messages` function in our `State` will append the llm's response messages to whatever messages are already in the state.

Next, add an `entry` point. This tells our graph **where to start its work** each time we run it.


```python
graph_builder.add_edge(START, "chatbot")
```

Similarly, set a `finish` point. This instructs the graph **"any time this node is run, you can exit."**


```python
graph_builder.add_edge("chatbot", END)
```

Finally, we'll want to be able to run our graph. To do so, call "`compile()`" on the graph builder. This creates a "`CompiledGraph`" we can use invoke on our state.


```python
graph = graph_builder.compile()
```

You can visualize the graph using the `get_graph` method and one of the "draw" methods, like `draw_ascii` or `draw_png`. The `draw` methods each require additional dependencies.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_17_0.jpg)
    


Now let's run the chatbot! 

**Tip:** You can exit the chat loop at any time by typing "quit", "exit", or "q".


```python
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

    Assistant: LangGraph is a library designed to help build stateful multi-agent applications using language models. It provides tools for creating workflows and state machines to coordinate multiple AI agents or language model interactions. LangGraph is built on top of LangChain, leveraging its components while adding graph-based coordination capabilities. It's particularly useful for developing more complex, stateful AI applications that go beyond simple query-response interactions.
    Goodbye!
    

**Congratulations!** You've built your first chatbot using LangGraph. This bot can engage in basic conversation by taking user input and generating responses using an LLM. You can inspect a [LangSmith Trace](https://smith.langchain.com/public/7527e308-9502-4894-b347-f34385740d5a/r) for the call above at the provided link.

However, you may have noticed that the bot's knowledge is limited to what's in its training data. In the next part, we'll add a web search tool to expand the bot's knowledge and make it more capable.

Below is the full code for this section for your reference:

<details>
<summary>Full Code</summary>
    <pre>
        
```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()
```

</pre>
</details>

## Part 2: Enhancing the Chatbot with Tools

To handle queries our chatbot can't answer "from memory", we'll integrate a web search tool. Our bot can use this tool to find relevant information and provide better responses.

#### Requirements

Before we start, make sure you have the necessary packages installed and API keys set up:

First, install the requirements to use the [Tavily Search Engine](https://python.langchain.com/docs/integrations/tools/tavily_search/), and set your [TAVILY_API_KEY](https://tavily.com/).


```python
%%capture --no-stderr
%pip install -U tavily-python langchain_community
```


```python
_set_env("TAVILY_API_KEY")
```

    TAVILY_API_KEY:  ········
    

Next, define the tool:


```python
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
tools = [tool]
tool.invoke("What's a 'node' in LangGraph?")
```




    [{'url': 'https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141',
      'content': 'Nodes: Nodes are the building blocks of your LangGraph. Each node represents a function or a computation step. You define nodes to perform specific tasks, such as processing input, making ...'},
     {'url': 'https://saksheepatil05.medium.com/demystifying-langgraph-a-beginner-friendly-dive-into-langgraph-concepts-5ffe890ddac0',
      'content': 'Nodes (Tasks): Nodes are like the workstations on the assembly line. Each node performs a specific task on the product. In LangGraph, nodes are Python functions that take the current state, do some work, and return an updated state. Next, we define the nodes, each representing a task in our sandwich-making process.'}]



The results are page summaries our chat bot can use to answer questions.


Next, we'll start defining our graph. The following is all **the same as in Part 1**, except we have added `bind_tools` on our LLM. This lets the LLM know the correct JSON format to use if it wants to use our search engine.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
```

Next we need to create a function to actually run the tools if they are called. We'll do this by adding the tools to a new node.

Below, we implement a `BasicToolNode` that checks the most recent message in the state and calls tools if the message contains `tool_calls`. It relies on the LLM's `tool_calling` support, which is available in Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.

We will later replace this with LangGraph's prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode) to speed things up, but building it ourselves first is instructive.


```python
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
```

With the tool node added, we can define the `conditional_edges`. 

Recall that **edges** route the control flow from one node to the next. **Conditional edges** usually contain "if" statements to route to different nodes depending on the current graph state. These functions receive the current graph `state` and return a string or list of strings indicating which node(s) to call next.

Below, call define a router function called `route_tools`, that checks for tool_calls in the chatbot's output. Provide this function to the graph by calling `add_conditional_edges`, which tells the graph that whenever the `chatbot` node completes to check this function to see where to go next. 

The condition will route to `tools` if tool calls are present and `END` if not.

Later, we will replace this with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition) to be more concise, but implementing it ourselves first makes things more clear. 


```python
from typing import Literal


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

**Notice** that conditional edges start from a single node. This tells the graph "any time the '`chatbot`' node runs, either go to 'tools' if it calls a tool, or end the loop if it responds directly. 

Like the prebuilt `tools_condition`, our function returns the `END` string if no tool calls are made. When the graph transitions to `END`, it has no more tasks to complete and ceases execution. Because the condition can return `END`, we don't need to explicitly set a `finish_point` this time. Our graph already has a way to finish!

Let's visualize the graph we've built. The following function has some additional dependencies to run that are unimportant for this tutorial.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_33_0.jpg)
    


Now we can ask the bot questions outside its training data.


```python
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

    Assistant: [{'text': "To provide you with accurate and up-to-date information about LangGraph, I'll need to search for the latest details. Let me do that for you.", 'type': 'text'}, {'id': 'toolu_01Q588CszHaSvvP2MxRq9zRD', 'input': {'query': 'LangGraph AI tool information'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Assistant: [{"url": "https://www.langchain.com/langgraph", "content": "LangGraph sets the foundation for how we can build and scale AI workloads \u2014 from conversational agents, complex task automation, to custom LLM-backed experiences that 'just work'. The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution ..."}, {"url": "https://github.com/langchain-ai/langgraph", "content": "Overview. LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures ..."}]
    Assistant: Based on the search results, I can provide you with information about LangGraph:
    
    1. Purpose:
       LangGraph is a library designed for building stateful, multi-actor applications with Large Language Models (LLMs). It's particularly useful for creating agent and multi-agent workflows.
    
    2. Developer:
       LangGraph is developed by LangChain, a company known for its tools and frameworks in the AI and LLM space.
    
    3. Key Features:
       - Cycles: LangGraph allows the definition of flows that involve cycles, which is essential for most agentic architectures.
       - Controllability: It offers enhanced control over the application flow.
       - Persistence: The library provides ways to maintain state and persistence in LLM-based applications.
    
    4. Use Cases:
       LangGraph can be used for various applications, including:
       - Conversational agents
       - Complex task automation
       - Custom LLM-backed experiences
    
    5. Integration:
       LangGraph works in conjunction with LangSmith, another tool by LangChain, to provide an out-of-the-box solution for building complex, production-ready features with LLMs.
    
    6. Significance:
       LangGraph is described as setting the foundation for building and scaling AI workloads. It's positioned as a key tool in the next chapter of LLM-based application development, particularly in the realm of agentic AI.
    
    7. Availability:
       LangGraph is open-source and available on GitHub, which suggests that developers can access and contribute to its codebase.
    
    8. Comparison to Other Frameworks:
       LangGraph is noted to offer unique benefits compared to other LLM frameworks, particularly in its ability to handle cycles, provide controllability, and maintain persistence.
    
    LangGraph appears to be a significant tool in the evolving landscape of LLM-based application development, offering developers new ways to create more complex, stateful, and interactive AI systems.
    Goodbye!
    

**Congrats!** You've created a conversational agent in langgraph that can use a search engine to retrieve updated information when needed. Now it can handle a wider range of user queries. To inspect all the steps your agent just took, check out this [LangSmith trace](https://smith.langchain.com/public/4fbd7636-25af-4638-9587-5a02fdbb0172/r).

Our chatbot still can't remember past interactions on its own, limiting its ability to have coherent, multi-turn conversations. In the next part, we'll add **memory** to address this.


The full code for the graph we've created in this section is reproduced below, replacing our `BasicToolNode` for the prebuilt [ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode), and our `route_tools` condition with the prebuilt [tools_condition](https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition)

<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()
```

</pre>
</details>

## Part 3: Adding Memory to the Chatbot

Our chatbot can now use tools to answer user questions, but it doesn't remember the context of previous interactions. This limits its ability to have coherent, multi-turn conversations.

LangGraph solves this problem through **persistent checkpointing**. If you provide a `checkpointer` when compiling the graph and a `thread_id` when calling your graph, LangGraph automatically saves the state after each step. When you invoke the graph again using the same `thread_id`, the graph loads its saved state, allowing the chatbot to pick up where it left off. 

We will see later that **checkpointing** is _much_ more powerful than simple chat memory - it lets you save and resume complex state at any time for error recovery, human-in-the-loop workflows, time travel interactions, and more. But before we get too ahead of ourselves, let's add checkpointing to enable multi-turn conversations.

To get started, create a `MemorySaver` checkpointer.


```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

**Notice** we're using an in-memory checkpointer. This is convenient for our tutorial (it saves it all in-memory). In a production application, you would likely change this to use `SqliteSaver` or `PostgresSaver` and connect to your own DB.

Next define the graph. Now that you've already built your own `BasicToolNode`, we'll replace it with LangGraph's prebuilt `ToolNode` and `tools_condition`, since these do some nice things like parallel API execution. Apart from that, the following is all copied from Part 2.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

Finally, compile the graph with the provided checkpointer.


```python
graph = graph_builder.compile(checkpointer=memory)
```

Notice the connectivity of the graph hasn't changed since Part 2. All we are doing is checkpointing the `State` as the graph works through each node.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_44_0.jpg)
    


Now you can interact with your bot! First, pick a thread to use as the key for this conversation.


```python
config = {"configurable": {"thread_id": "1"}}
```

Next, call your chat bot.


```python
user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Hi there! My name is Will.
    ==================================[1m Ai Message [0m==================================
    
    Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?
    

**Note:** The config was provided as the **second positional argument** when calling our graph. It importantly is _not_ nested within the graph inputs (`{'messages': []}`).

Let's ask a followup: see if it remembers your name.


```python
user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Remember my name?
    ==================================[1m Ai Message [0m==================================
    
    Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.
    

**Notice** that we aren't using an external list for memory: it's all handled by the checkpointer! You can inspect the full execution in this [LangSmith trace](https://smith.langchain.com/public/29ba22b5-6d40-4fbe-8d27-b369e3329c84/r) to see what's going on.

Don't believe me? Try this using a different config.


```python
# The only difference is we change the `thread_id` here to "2" instead of "1"
events = graph.stream(
    {"messages": [("user", user_input)]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Remember my name?
    ==================================[1m Ai Message [0m==================================
    
    I apologize, but I don't have any previous context or memory of your name. As an AI assistant, I don't retain information from past conversations. Each interaction starts fresh. Could you please tell me your name so I can address you properly in this conversation?
    

**Notice** that the **only** change we've made is to modify the `thread_id` in the config. See this call's [LangSmith trace](https://smith.langchain.com/public/51a62351-2f0a-4058-91cc-9996c5561428/r) for comparison. 

By now, we have made a few checkpoints across two different threads. But what goes into a checkpoint? To inspect a graph's `state` for a given config at any time, call `get_state(config)`.


```python
snapshot = graph.get_state(config)
snapshot
```




    StateSnapshot(values={'messages': [HumanMessage(content='Hi there! My name is Will.', additional_kwargs={}, response_metadata={}, id='8c1ca919-c553-4ebf-95d4-b59a2d61e078'), AIMessage(content="Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?", additional_kwargs={}, response_metadata={'id': 'msg_01WTQebPhNwmMrmmWojJ9KXJ', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 405, 'output_tokens': 32}}, id='run-58587b77-8c82-41e6-8a90-d62c444a261d-0', usage_metadata={'input_tokens': 405, 'output_tokens': 32, 'total_tokens': 437}), HumanMessage(content='Remember my name?', additional_kwargs={}, response_metadata={}, id='daba7df6-ad75-4d6b-8057-745881cea1ca'), AIMessage(content="Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.", additional_kwargs={}, response_metadata={'id': 'msg_01E41KitY74HpENRgXx94vag', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 444, 'output_tokens': 58}}, id='run-ffeaae5c-4d2d-4ddb-bd59-5d5cbf2a5af8-0', usage_metadata={'input_tokens': 444, 'output_tokens': 58, 'total_tokens': 502})]}, next=(), config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d06e-93e0-6acc-8004-f2ac846575d2'}}, metadata={'source': 'loop', 'writes': {'chatbot': {'messages': [AIMessage(content="Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.", additional_kwargs={}, response_metadata={'id': 'msg_01E41KitY74HpENRgXx94vag', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 444, 'output_tokens': 58}}, id='run-ffeaae5c-4d2d-4ddb-bd59-5d5cbf2a5af8-0', usage_metadata={'input_tokens': 444, 'output_tokens': 58, 'total_tokens': 502})]}}, 'step': 4, 'parents': {}}, created_at='2024-09-27T19:30:10.820758+00:00', parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d06e-859f-6206-8003-e1bd3c264b8f'}}, tasks=())




```python
snapshot.next  # (since the graph ended this turn, `next` is empty. If you fetch a state from within a graph invocation, next tells which node will execute next)
```




    ()



The snapshot above contains the current state values, corresponding config, and the `next` node to process. In our case, the graph has reached an `END` state, so `next` is empty.

**Congratulations!** Your chatbot can now maintain conversation state across sessions thanks to LangGraph's checkpointing system. This opens up exciting possibilities for more natural, contextual interactions. LangGraph's checkpointing even handles **arbitrarily complex graph states**, which is much more expressive and powerful than simple chat memory.

In the next part, we'll introduce human oversight to our bot to handle situations where it may need guidance or verification before proceeding.
  
Check out the code snippet below to review our graph from this section.

<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)
```
</pre>
</pre>
</details>

## Part 4: Human-in-the-loop

Agents can be unreliable and may need human input to successfully accomplish tasks. Similarly, for some actions, you may want to require human approval before running to ensure that everything is running as intended.

LangGraph supports `human-in-the-loop` workflows in a number of ways. In this section, we will use LangGraph's `interrupt_before` functionality to always break the tool node.

First, start from our existing code. The following is copied from Part 3.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

Now, compile the graph, specifying to `interrupt_before` the `tools` node.


```python
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ tools, if desired.
    # interrupt_after=["tools"]
)
```


```python
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I'm learning LangGraph. Could you do some research on it for me?
    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and comprehensive information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01R4ZFcb5hohpiVZwr88Bxhc', 'input': {'query': 'LangGraph framework for building language model applications'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01R4ZFcb5hohpiVZwr88Bxhc)
     Call ID: toolu_01R4ZFcb5hohpiVZwr88Bxhc
      Args:
        query: LangGraph framework for building language model applications
    

Let's inspect the graph state to confirm it worked.


```python
snapshot = graph.get_state(config)
snapshot.next
```




    ('tools',)



**Notice** that unlike last time, the "next" node is set to **'tools'**. We've interrupted here! Let's check the tool invocation.


```python
existing_message = snapshot.values["messages"][-1]
existing_message.tool_calls
```




    [{'name': 'tavily_search_results_json',
      'args': {'query': 'LangGraph framework for building language model applications'},
      'id': 'toolu_01R4ZFcb5hohpiVZwr88Bxhc',
      'type': 'tool_call'}]



This query seems reasonable. Nothing to filter here. The simplest thing the human can do is just let the graph continue executing. Let's do that below.

Next, continue the graph! Passing in `None` will just let the graph continue where it left off, without adding anything new to the state.


```python
# `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and comprehensive information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01R4ZFcb5hohpiVZwr88Bxhc', 'input': {'query': 'LangGraph framework for building language model applications'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01R4ZFcb5hohpiVZwr88Bxhc)
     Call ID: toolu_01R4ZFcb5hohpiVZwr88Bxhc
      Args:
        query: LangGraph framework for building language model applications
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787", "content": "LangChain is one of the leading frameworks for building applications powered by Lardge Language Models. With the LangChain Expression Language (LCEL), defining and executing step-by-step action sequences — also known as chains — becomes much simpler. In more technical terms, LangChain allows us to create DAGs (directed acyclic graphs). As LLM applications, particularly LLM agents, have ..."}, {"url": "https://github.com/langchain-ai/langgraph", "content": "Overview. LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures ..."}]
    ==================================[1m Ai Message [0m==================================
    
    Thank you for your patience. I've found some valuable information about LangGraph for you. Let me summarize the key points:
    
    1. LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs).
    
    2. It is particularly useful for creating agent and multi-agent workflows.
    
    3. LangGraph is built on top of LangChain, which is one of the leading frameworks for building LLM-powered applications.
    
    4. Key benefits of LangGraph compared to other LLM frameworks include:
       a) Cycles: It allows you to define flows that involve cycles, which is essential for most agent architectures.
       b) Controllability: Offers more control over the application flow.
       c) Persistence: Provides ways to maintain state across interactions.
    
    5. LangGraph works well with the LangChain Expression Language (LCEL), which simplifies the process of defining and executing step-by-step action sequences (chains).
    
    6. In technical terms, LangGraph enables the creation of Directed Acyclic Graphs (DAGs) for LLM applications.
    
    7. It's particularly useful for building more complex LLM agents and multi-agent systems.
    
    LangGraph seems to be an advanced tool that builds upon LangChain to provide more sophisticated capabilities for creating stateful and multi-actor LLM applications. It's especially valuable if you're looking to create complex agent systems or applications that require maintaining state across interactions.
    
    Is there any specific aspect of LangGraph you'd like to know more about? I'd be happy to dive deeper into any particular area of interest.
    

Review this call's [LangSmith trace](https://smith.langchain.com/public/4d7f8757-9d3b-43b9-88b6-aeab0595bc4c/r) to see the exact work that was done in the above call. Notice that the state is loaded in the first step so that your chatbot can continue where it left off.

**Congrats!** You've used an `interrupt` to add human-in-the-loop execution to your chatbot, allowing for human oversight and intervention when needed. This opens up the potential UIs you can create with your AI systems. Since we have already added a **checkpointer**, the graph can be paused **indefinitely** and resumed at any time as if nothing had happened.

Next, we'll explore how to further customize the bot's behavior using custom state updates.

Below is a copy of the code you used in this section. The only difference between this and the previous parts is the addition of the `interrupt_before` argument.

<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ actions, if desired.
    # interrupt_after=["tools"]
)
```
</pre>
</details>

## Part 5: Manually Updating the State

In the previous section, we showed how to interrupt a graph so that a human could inspect its actions. This lets the human `read` the state, but if they want to change their agent's course, they'll need to have `write` access.

Thankfully, LangGraph lets you **manually update state**! Updating the state lets you control the agent's trajectory by modifying its actions (even modifying the past!). This capability is particularly useful when you want to correct the agent's mistakes, explore alternative paths, or guide the agent towards a specific goal.

We'll show how to update a checkpointed state below. As before, first, define your graph. We'll reuse the exact same graph as before.


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt **after** actions, if desired.
    # interrupt_after=["tools"]
)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```


```python
snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
existing_message.pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and comprehensive information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_018YcbFR37CG8RRXnavH5fxZ', 'input': {'query': 'LangGraph: what is it, how is it used in AI development'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_018YcbFR37CG8RRXnavH5fxZ)
     Call ID: toolu_018YcbFR37CG8RRXnavH5fxZ
      Args:
        query: LangGraph: what is it, how is it used in AI development
    

So far, all of this is an _exact repeat_ of the previous section. The LLM just requested to use the search engine tool and our graph was interrupted. If we proceed as before, the tool will be called to search the web.

But what if the user wants to intercede? What if we think the chat bot doesn't need to use the tool? 

Let's directly provide the correct response!


```python
from langchain_core.messages import AIMessage, ToolMessage

answer = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs."
)
new_messages = [
    # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
    ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
    # And then directly "put words in the LLM's mouth" by populating its response.
    AIMessage(content=answer),
]

new_messages[-1].pretty_print()
graph.update_state(
    # Which state to update
    config,
    # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
    # to the existing state. We will review how to update existing messages in the next section!
    {"messages": new_messages},
)

print("\n\nLast 2 messages;")
print(graph.get_state(config).values["messages"][-2:])
```

    ==================================[1m Ai Message [0m==================================
    
    LangGraph is a library for building stateful, multi-actor applications with LLMs.
    
    
    Last 2 messages;
    [ToolMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', id='675f7618-367f-44b7-b80e-2834afb02ac5', tool_call_id='toolu_018YcbFR37CG8RRXnavH5fxZ'), AIMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', additional_kwargs={}, response_metadata={}, id='35fd5682-0c2a-4200-b192-71c59ac6d412')]
    

Now the graph is complete, since we've provided the final response message! Since state updates simulate a graph step, they even generate corresponding traces. Inspec the [LangSmith trace](https://smith.langchain.com/public/6d72aeb5-3bca-4090-8684-a11d5a36b10c/r) of the `update_state` call above to see what's going on.

**Notice** that our new messages are _appended_ to the messages already in the state. Remember how we defined the `State` type?

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

We annotated `messages` with the pre-built `add_messages` function. This instructs the graph to always append values to the existing list, rather than overwriting the list directly. The same logic is applied here, so the messages we passed to `update_state` were appended in the same way!

The `update_state` function operates as if it were one of the nodes in your graph! By default, the update operation uses the node that was last executed, but you can manually specify it below. Let's add an update and tell the graph to treat it as if it came from the "chatbot".


```python
graph.update_state(
    config,
    {"messages": [AIMessage(content="I'm an AI expert!")]},
    # Which node for this function to act as. It will automatically continue
    # processing as if this node just ran.
    as_node="chatbot",
)
```




    {'configurable': {'thread_id': '1',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef7d134-3958-6412-8002-3f4b4112062f'}}



Check out the [LangSmith trace](https://smith.langchain.com/public/2e4d92ca-c17c-49e0-92e5-3962390ded30/r) for this update call at the provided link. **Notice** from the trace that the graph continues into the `tools_condition` edge. We just told the graph to treat the update `as_node="chatbot"`. If we follow the diagram below and start from the `chatbot` node, we naturally end up in the `tools_condition` edge and then `__end__` since our updated message lacks tool calls.


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_77_0.jpg)
    


Inspect the current state as before to confirm the checkpoint reflects our manual updates.


```python
snapshot = graph.get_state(config)
print(snapshot.values["messages"][-3:])
print(snapshot.next)
```

    [ToolMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', id='675f7618-367f-44b7-b80e-2834afb02ac5', tool_call_id='toolu_018YcbFR37CG8RRXnavH5fxZ'), AIMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', additional_kwargs={}, response_metadata={}, id='35fd5682-0c2a-4200-b192-71c59ac6d412'), AIMessage(content="I'm an AI expert!", additional_kwargs={}, response_metadata={}, id='288e2f74-f1cb-4082-8c3c-af4695c83117')]
    ()
    

**Notice**: that we've continued to add AI messages to the state. Since we are acting as the `chatbot` and responding with an AIMessage that doesn't contain `tool_calls`, the graph knows that it has entered a finished state (`next` is empty).

#### What if you want to **overwrite** existing messages? 

The [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages) function we used to annotate our graph's `State` above controls how updates are made to the `messages` key. This function looks at any message IDs in the new `messages` list. If the ID matches a message in the existing state, [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/?h=add+messages#add_messages) overwrites the existing message with the new content. 

As an example, let's update the tool invocation to make sure we get good results from our search engine! First, start a new thread:


```python
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "2"}}  # we'll use thread_id = 2 here
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I'm learning LangGraph. Could you do some research on it for me?
    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and accurate information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01TfAeisrpx4ddgJpoAxqrVh', 'input': {'query': 'LangGraph framework for language models'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01TfAeisrpx4ddgJpoAxqrVh)
     Call ID: toolu_01TfAeisrpx4ddgJpoAxqrVh
      Args:
        query: LangGraph framework for language models
    

**Next,** let's update the tool invocation for our agent. Maybe we want to search for human-in-the-loop workflows in particular.


```python
from langchain_core.messages import AIMessage

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
print("Original")
print("Message ID", existing_message.id)
print(existing_message.tool_calls[0])
new_tool_call = existing_message.tool_calls[0].copy()
new_tool_call["args"]["query"] = "LangGraph human-in-the-loop workflow"
new_message = AIMessage(
    content=existing_message.content,
    tool_calls=[new_tool_call],
    # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
    id=existing_message.id,
)

print("Updated")
print(new_message.tool_calls[0])
print("Message ID", new_message.id)
graph.update_state(config, {"messages": [new_message]})

print("\n\nTool calls")
graph.get_state(config).values["messages"][-1].tool_calls
```

    Original
    Message ID run-342f3f54-356b-4cc1-b747-573f6aa31054-0
    {'name': 'tavily_search_results_json', 'args': {'query': 'LangGraph framework for language models'}, 'id': 'toolu_01TfAeisrpx4ddgJpoAxqrVh', 'type': 'tool_call'}
    Updated
    {'name': 'tavily_search_results_json', 'args': {'query': 'LangGraph human-in-the-loop workflow'}, 'id': 'toolu_01TfAeisrpx4ddgJpoAxqrVh', 'type': 'tool_call'}
    Message ID run-342f3f54-356b-4cc1-b747-573f6aa31054-0
    
    
    Tool calls
    




    [{'name': 'tavily_search_results_json',
      'args': {'query': 'LangGraph human-in-the-loop workflow'},
      'id': 'toolu_01TfAeisrpx4ddgJpoAxqrVh',
      'type': 'tool_call'}]



**Notice** that we've modified the AI's tool invocation to search for "LangGraph human-in-the-loop workflow" instead of the simple "LangGraph".

Check out the [LangSmith trace](https://smith.langchain.com/public/cd7c09a6-758d-41d4-8de1-64ab838b2338/r) to see the state update call - you can see our new message has successfully updated the previous AI message.

Resume the graph by streaming with an input of `None` and the existing config.


```python
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and accurate information, I'll use the Tavily search engine to look this up. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_01TfAeisrpx4ddgJpoAxqrVh', 'input': {'query': 'LangGraph framework for language models'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01TfAeisrpx4ddgJpoAxqrVh)
     Call ID: toolu_01TfAeisrpx4ddgJpoAxqrVh
      Args:
        query: LangGraph human-in-the-loop workflow
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.youtube.com/watch?v=9BPCV5TYPmg", "content": "In this video, I'll show you how to handle persistence with LangGraph, enabling a unique Human-in-the-Loop workflow. This approach allows a human to grant an..."}, {"url": "https://medium.com/@kbdhunga/implementing-human-in-the-loop-with-langgraph-ccfde023385c", "content": "Implementing a Human-in-the-Loop (HIL) framework in LangGraph with the Streamlit app provides a robust mechanism for user engagement and decision-making. By incorporating breakpoints and ..."}]
    ==================================[1m Ai Message [0m==================================
    
    Thank you for your patience. I've found some information about LangGraph, particularly focusing on its human-in-the-loop workflow capabilities. Let me summarize what I've learned for you:
    
    1. LangGraph Overview:
       LangGraph is a framework for building stateful, multi-actor applications with Large Language Models (LLMs). It's particularly useful for creating complex, interactive AI systems.
    
    2. Human-in-the-Loop (HIL) Workflow:
       One of the key features of LangGraph is its support for human-in-the-loop workflows. This means that it allows for human intervention and decision-making within AI-driven processes.
    
    3. Persistence Handling:
       LangGraph offers capabilities for handling persistence, which is crucial for maintaining state across interactions in a workflow.
    
    4. Implementation with Streamlit:
       There are examples of implementing LangGraph's human-in-the-loop functionality using Streamlit, a popular Python library for creating web apps. This combination allows for the creation of interactive user interfaces for AI applications.
    
    5. Breakpoints and User Engagement:
       LangGraph allows the incorporation of breakpoints in the workflow. These breakpoints are points where the system can pause and wait for human input or decision-making, enhancing user engagement and control over the AI process.
    
    6. Decision-Making Mechanism:
       The human-in-the-loop framework in LangGraph provides a robust mechanism for integrating user decision-making into AI workflows. This is particularly useful in scenarios where human judgment or expertise is needed to guide or validate AI actions.
    
    7. Flexibility and Customization:
       From the information available, it seems that LangGraph offers flexibility in how human-in-the-loop processes are implemented, allowing developers to customize the interaction points and the nature of human involvement based on their specific use case.
    
    LangGraph appears to be a powerful tool for developers looking to create more interactive and controllable AI applications, especially those that benefit from human oversight or input at crucial stages of the process.
    
    Would you like me to research any specific aspect of LangGraph in more detail, or do you have any questions about what I've found so far?
    

Check out the [trace](https://smith.langchain.com/public/2d633326-14ad-4248-a391-2757d01851c4/r/6464f2f2-edb4-4ef3-8f48-ee4e249f2ad0) to see the tool call and later LLM response. **Notice** that now the graph queries the search engine using our updated query term - we were able to manually override the LLM's search here!

All of this is reflected in the graph's checkpointed memory, meaning if we continue the conversation, it will recall all the _modified_ state.


```python
events = graph.stream(
    {
        "messages": (
            "user",
            "Remember what I'm learning about?",
        )
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Remember what I'm learning about?
    ==================================[1m Ai Message [0m==================================
    
    I apologize for my oversight. You're absolutely right to remind me. You mentioned that you're learning LangGraph. Thank you for bringing that back into focus. 
    
    Since you're in the process of learning LangGraph, it would be helpful to know more about your current level of understanding and what specific aspects of LangGraph you're most interested in or finding challenging. This way, I can provide more targeted information or explanations that align with your learning journey.
    
    Are there any particular areas of LangGraph you'd like to explore further? For example:
    
    1. Basic concepts and architecture of LangGraph
    2. Setting up and getting started with LangGraph
    3. Implementing specific features like the human-in-the-loop workflow
    4. Best practices for using LangGraph in projects
    5. Comparisons with other similar frameworks
    
    Or if you have any specific questions about what you've learned so far, I'd be happy to help clarify or expand on those topics. Please let me know what would be most useful for your learning process.
    

**Congratulations!** You've used `interrupt_before` and `update_state` to manually modify the state as a part of a human-in-the-loop workflow. Interruptions and state modifications let you control how the agent behaves. Combined with persistent checkpointing, it means you can `pause` an action and `resume` at any point. Your user doesn't have to be available when the graph interrupts!

The graph code for this section is identical to previous ones. The key snippets to remember are to add `.compile(..., interrupt_before=[...])` (or `interrupt_after`) if you want to explicitly pause the graph whenever it reaches a node. Then you can use `update_state` to modify the checkpoint and control how the graph should proceed.

## Part 6: Customizing State

So far, we've relied on a simple state (it's just a list of messages!). You can go far with this simple state, but if you want to define complex behavior without relying on the message list, you can add additional fields to the state. In this section, we will extend our chat bot with a new node to illustrate this.

In the examples above, we involved a human deterministically: the graph __always__ interrupted whenever an tool was invoked. Suppose we wanted our chat bot to have the choice of relying on a human.

One way to do this is to create a passthrough "human" node, before which the graph will always stop. We will only execute this node if the LLM invokes a "human" tool. For our convenience, we will include an "ask_human" flag in our graph state that we will flip if the LLM calls this tool.

Below, define this new graph, with an updated `State`


```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool
```

Next, define a schema to show the model to let it decide to request assistance.

<div class="admonition note">
    <p class="admonition-title">Using Pydantic with LangChain</p>
    <p>
        This notebook uses Pydantic v2 <code>BaseModel</code>, which requires <code>langchain-core >= 0.3</code>. Using <code>langchain-core < 0.3</code> will result in errors due to mixing of Pydantic v1 and v2 <code>BaseModels</code>.
    </p>
</div>


```python
from pydantic import BaseModel


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str
```

Next, define the chatbot node. The primary modification here is flip the `ask_human` flag if we see that the chat bot has invoked the `RequestAssistance` flag.


```python
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# We can bind the llm to a tool definition, a pydantic model, or a json schema
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}
```

Next, create the graph builder and add the chatbot and tools nodes to the graph, same as before.


```python
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))
```

Next, create the "human" `node`. This `node` function is mostly a placeholder in our graph that will trigger an interrupt. If the human does __not__ manually update the state during the `interrupt`, it inserts a tool message so the LLM knows the user was requested but didn't respond. This node also unsets the `ask_human` flag so the graph knows not to revisit the node unless further requests are made.


```python
from langchain_core.messages import AIMessage, ToolMessage


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)
```

Next, define the conditional logic. The `select_next_node` will route to the `human` node if the flag is set. Otherwise, it lets the prebuilt `tools_condition` function choose the next node.

Recall that the `tools_condition` function simply checks to see if the `chatbot` has responded with any `tool_calls` in its response message. If so, it routes to the `action` node. Otherwise, it ends the graph.


```python
def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)
```

Finally, add the simple directed edges and compile the graph. These edges instruct the graph to **always** flow from node `a`->`b` whenever `a` finishes executing.


```python
# The rest is the same
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    # We interrupt before 'human' here instead.
    interrupt_before=["human"],
)
```

If you have the visualization dependencies installed, you can see the graph structure below:


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_105_0.jpg)
    


The chat bot can either request help from a human (chatbot->select->human), invoke the search engine tool (chatbot->select->action), or directly respond (chatbot->select->__end__). Once an action or request has been made, the graph will transition back to the `chatbot` node to continue operations.

Let's see this graph in action. We will request for expert assistance to illustrate our graph.


```python
user_input = "I need some expert guidance for building this AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I need some expert guidance for building this AI agent. Could you request assistance for me?
    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I understand that you need expert guidance for building an AI agent. I'll use the RequestAssistance function to escalate your request to an expert who can provide you with the specialized knowledge and support you need. Let me do that for you right away.", 'type': 'text'}, {'id': 'toolu_01Mo3N2c1byuSZwT1vyJWRia', 'input': {'request': 'The user needs expert guidance for building an AI agent. They require specialized knowledge and support in AI development and implementation.'}, 'name': 'RequestAssistance', 'type': 'tool_use'}]
    Tool Calls:
      RequestAssistance (toolu_01Mo3N2c1byuSZwT1vyJWRia)
     Call ID: toolu_01Mo3N2c1byuSZwT1vyJWRia
      Args:
        request: The user needs expert guidance for building an AI agent. They require specialized knowledge and support in AI development and implementation.
    

**Notice:** the LLM has invoked the "`RequestAssistance`" tool we provided it, and the interrupt has been set. Let's inspect the graph state to confirm.


```python
snapshot = graph.get_state(config)
snapshot.next
```




    ('human',)



The graph state is indeed **interrupted** before the `'human'` node. We can act as the "expert" in this scenario and manually update the state by adding a new ToolMessage with our input.

Next, respond to the chatbot's request by:
1. Creating a `ToolMessage` with our response. This will be passed back to the `chatbot`.
2. Calling `update_state` to manually update the graph state.


```python
ai_message = snapshot.values["messages"][-1]
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)
tool_message = create_response(human_response, ai_message)
graph.update_state(config, {"messages": [tool_message]})
```




    {'configurable': {'thread_id': '1',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef7d092-bb30-6bee-8002-015e7e1c56c0'}}



You can inspect the state to confirm our response was added.


```python
graph.get_state(config).values["messages"]
```




    [HumanMessage(content='I need some expert guidance for building this AI agent. Could you request assistance for me?', additional_kwargs={}, response_metadata={}, id='3f28f959-9ab7-489a-9c58-7ed1b49cedf3'),
     AIMessage(content=[{'text': "Certainly! I understand that you need expert guidance for building an AI agent. I'll use the RequestAssistance function to escalate your request to an expert who can provide you with the specialized knowledge and support you need. Let me do that for you right away.", 'type': 'text'}, {'id': 'toolu_01Mo3N2c1byuSZwT1vyJWRia', 'input': {'request': 'The user needs expert guidance for building an AI agent. They require specialized knowledge and support in AI development and implementation.'}, 'name': 'RequestAssistance', 'type': 'tool_use'}], additional_kwargs={}, response_metadata={'id': 'msg_01VRnZvVbgsVRbQaQuvsziDx', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 516, 'output_tokens': 130}}, id='run-4e3f7906-5887-40d9-9267-5beefe7b3b76-0', tool_calls=[{'name': 'RequestAssistance', 'args': {'request': 'The user needs expert guidance for building an AI agent. They require specialized knowledge and support in AI development and implementation.'}, 'id': 'toolu_01Mo3N2c1byuSZwT1vyJWRia', 'type': 'tool_call'}], usage_metadata={'input_tokens': 516, 'output_tokens': 130, 'total_tokens': 646}),
     ToolMessage(content="We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.", id='8583b899-d898-4051-9f36-f5e5d11e9a37', tool_call_id='toolu_01Mo3N2c1byuSZwT1vyJWRia')]



Next, **resume** the graph by invoking it with `None` as the inputs.


```python
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    =================================[1m Tool Message [0m=================================
    
    We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
    =================================[1m Tool Message [0m=================================
    
    We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
    ==================================[1m Ai Message [0m==================================
    
    Thank you for your patience. I've escalated your request to our expert team, and they have provided some initial guidance. Here's what they suggest:
    
    The experts recommend that you check out LangGraph for building your AI agent. They mention that LangGraph is a more reliable and extensible option compared to simple autonomous agents.
    
    LangGraph is likely a framework or tool designed specifically for creating complex AI agents. It seems to offer advantages in terms of reliability and extensibility, which are crucial factors when developing sophisticated AI systems.
    
    To further assist you, I can provide some additional context and next steps:
    
    1. Research LangGraph: Look up documentation, tutorials, and examples of LangGraph to understand its features and how it can help you build your AI agent.
    
    2. Compare with other options: While the experts recommend LangGraph, it might be useful to understand how it compares to other AI agent development frameworks or tools you might have been considering.
    
    3. Assess your requirements: Consider your specific needs for the AI agent you want to build. Think about the tasks it needs to perform, the level of complexity required, and how LangGraph's features align with these requirements.
    
    4. Start with a small project: If you decide to use LangGraph, consider beginning with a small, manageable project to familiarize yourself with the framework.
    
    5. Seek community support: Look for LangGraph user communities, forums, or discussion groups where you can ask questions and get additional support as you build your agent.
    
    6. Consider additional training: Depending on your current skill level, you might want to look into courses or workshops that focus on AI agent development, particularly those that cover LangGraph.
    
    Do you have any specific questions about LangGraph or AI agent development that you'd like me to try to answer? Or would you like me to search for more detailed information about LangGraph and its features?
    

**Notice** that the chat bot has incorporated the updated state in its final response. Since **everything** was checkpointed, the "expert" human in the loop could perform the update at any time without impacting the graph's execution.

**Congratulations!** you've now added an additional node to your assistant graph to let the chat bot decide for itself whether or not it needs to interrupt execution. You did so by updating the graph `State` with a new `ask_human` field and modifying the interruption logic when compiling the graph. This lets you dynamically include a human in the loop while maintaining full **memory** every time you execute the graph.

We're almost done with the tutorial, but there is one more concept we'd like to review before finishing that connects `checkpointing` and `state updates`. 

This section's code is reproduced below for your reference.

<details>
<summary>Full Code</summary>
    <pre>

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# We can bind the llm to a tool definition, a pydantic model, or a json schema
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
)
```
</pre>
</details>

## Part 7: Time Travel

In a typical chat bot workflow, the user interacts with the bot 1 or more times to accomplish a task. In the previous sections, we saw how to add memory and a human-in-the-loop to be able to checkpoint our graph state and manually override the state to control future responses.

But what if you want to let your user start from a previous response and "branch off" to explore a separate outcome? Or what if you want users to be able to "rewind" your assistant's work to fix some mistakes or try a different strategy (common in applications like autonomous software engineers)?

You can create both of these experiences and more using LangGraph's built-in "time travel" functionality. 

In this section, you will "rewind" your graph by fetching a checkpoint using the graph's `get_state_history` method. You can then resume execution at this previous point in time.

First, recall our chatbot graph. We don't need to make **any** changes from before:


```python
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage

# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
# We can bind the llm to a tool definition, a pydantic model, or a json schema
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
)
```


```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```


    
![jpeg](output_119_0.jpg)
    


Let's have our graph take a couple steps. Every step will be checkpointed in its state history:


```python
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            ("user", "I'm learning LangGraph. Could you do some research on it for me?")
        ]
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    I'm learning LangGraph. Could you do some research on it for me?
    ==================================[1m Ai Message [0m==================================
    
    [{'text': "Certainly! I'd be happy to research LangGraph for you. To get the most up-to-date and accurate information, I'll use the Tavily search function to gather details about LangGraph. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_019HPZEw6v1eSLBXnwxk6MZm', 'input': {'query': 'LangGraph framework for language models'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_019HPZEw6v1eSLBXnwxk6MZm)
     Call ID: toolu_019HPZEw6v1eSLBXnwxk6MZm
      Args:
        query: LangGraph framework for language models
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141", "content": "LangGraph is a powerful tool for building stateful, multi-actor applications with Large Language Models (LLMs). It extends the LangChain library, allowing you to coordinate multiple chains (or ..."}, {"url": "https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787", "content": "LangChain is one of the leading frameworks for building applications powered by Lardge Language Models. With the LangChain Expression Language (LCEL), defining and executing step-by-step action sequences — also known as chains — becomes much simpler. In more technical terms, LangChain allows us to create DAGs (directed acyclic graphs)."}]
    ==================================[1m Ai Message [0m==================================
    
    Thank you for your patience. I've gathered some information about LangGraph for you. Let me summarize the key points:
    
    1. What is LangGraph?
       LangGraph is a powerful tool designed for building stateful, multi-actor applications using Large Language Models (LLMs). It's an extension of the LangChain library, which is already a popular framework for developing LLM-powered applications.
    
    2. Purpose and Functionality:
       - LangGraph allows developers to coordinate multiple chains or actors within a single application.
       - It enhances the capabilities of LangChain by introducing more complex, stateful workflows.
    
    3. Relation to LangChain:
       - LangGraph builds upon LangChain, which is one of the leading frameworks for creating LLM-powered applications.
       - LangChain itself uses the LangChain Expression Language (LCEL) to define and execute step-by-step action sequences, also known as chains.
       - LangChain allows the creation of DAGs (Directed Acyclic Graphs), which represent the flow of operations in an application.
    
    4. Key Features:
       - Stateful Applications: Unlike simple query-response models, LangGraph allows the creation of applications that maintain state across interactions.
       - Multi-Actor Systems: It supports coordinating multiple AI "actors" or components within a single application, enabling more complex interactions and workflows.
    
    5. Use Cases:
       While not explicitly mentioned in the search results, LangGraph is typically used for creating more sophisticated AI applications such as:
       - Multi-turn conversational agents
       - Complex task-planning systems
       - Applications requiring memory and context management across multiple steps or actors
    
    Learning LangGraph can be a valuable skill, especially if you're interested in developing advanced applications with LLMs that go beyond simple question-answering or text generation tasks. It allows for the creation of more dynamic, interactive, and stateful AI systems.
    
    Is there any specific aspect of LangGraph you'd like to know more about, or do you have any questions about how it compares to or works with LangChain?
    


```python
events = graph.stream(
    {
        "messages": [
            ("user", "Ya that's helpful. Maybe I'll build an autonomous agent with it!")
        ]
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    Ya that's helpful. Maybe I'll build an autonomous agent with it!
    ==================================[1m Ai Message [0m==================================
    
    [{'text': "That's an excellent idea! Building an autonomous agent with LangGraph is a great way to explore its capabilities and learn about advanced AI application development. LangGraph's features make it well-suited for creating autonomous agents. Let me provide some additional insights and encouragement for your project.", 'type': 'text'}, {'id': 'toolu_017t6BS5rNCzFWcpxRizDKjE', 'input': {'query': 'building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_017t6BS5rNCzFWcpxRizDKjE)
     Call ID: toolu_017t6BS5rNCzFWcpxRizDKjE
      Args:
        query: building autonomous agents with LangGraph examples and tutorials
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://medium.com/@lucas.dahan/hands-on-langgraph-building-a-multi-agent-assistant-06aa68ed942f", "content": "Building the Graph. With our agents defined, we'll create a graph.py file to orchestrate their interactions. The basic graph structure in LangGraph is really simple, here we are going to use ..."}, {"url": "https://medium.com/@cplog/building-tool-calling-conversational-ai-with-langchain-and-langgraph-a-beginners-guide-8d6986cc589e", "content": "Introduction to AI Agent with LangChain and LangGraph: A Beginner’s Guide Two powerful tools revolutionizing this field are LangChain and LangGraph. In this guide, we’ll explore how these technologies can be combined to build a sophisticated AI assistant capable of handling complex conversations and tasks. Tool calling is a standout feature in agentic design, allowing the LLM to interact with external systems or perform specific tasks via the @tool decorator. While the Assistant class presented here is one approach, the flexibility of tool calling and LangGraph allows for a wide range of designs. With LangChain and LangGraph, you can build a powerful, flexible AI assistant capable of handling complex tasks and conversations. Tool calling significantly enhances the AI’s capabilities by enabling interaction with external systems."}]
    ==================================[1m Ai Message [0m==================================
    
    Your enthusiasm for building an autonomous agent with LangGraph is fantastic! This project will not only help you learn more about LangGraph but also give you hands-on experience with cutting-edge AI development. Here are some insights and tips to get you started:
    
    1. Multi-Agent Systems:
       LangGraph excels at creating multi-agent systems. You could design your autonomous agent as a collection of specialized sub-agents, each handling different aspects of tasks or knowledge domains.
    
    2. Graph Structure:
       The basic graph structure in LangGraph is straightforward. You'll create a graph.py file to orchestrate the interactions between your agents or components.
    
    3. Tool Calling:
       A key feature you can incorporate is tool calling. This allows your LLM-based agent to interact with external systems or perform specific tasks. You can implement this using the @tool decorator in your code.
    
    4. Flexibility in Design:
       LangGraph offers great flexibility in designing your agent. While there are example structures like the Assistant class, you have the freedom to create a wide range of designs tailored to your specific needs.
    
    5. Complex Conversations and Tasks:
       Your autonomous agent can be designed to handle sophisticated conversations and complex tasks. This is where LangGraph's stateful nature really shines, allowing your agent to maintain context over extended interactions.
    
    6. Integration with LangChain:
       Since LangGraph builds upon LangChain, you can leverage features from both. This combination allows for powerful, flexible AI assistants capable of managing intricate workflows.
    
    7. External System Interaction:
       Consider incorporating external APIs or databases to enhance your agent's capabilities. This could include accessing real-time data, performing calculations, or interacting with other services.
    
    8. Tutorial Resources:
       There are tutorials available that walk through the process of building AI assistants with LangChain and LangGraph. These can be excellent starting points for your project.
    
    To get started, you might want to:
    1. Set up your development environment with LangChain and LangGraph.
    2. Define the core functionalities you want your autonomous agent to have.
    3. Design the overall structure of your agent, possibly as a multi-agent system.
    4. Implement basic interactions and gradually add more complex features like tool calling and state management.
    5. Test your agent thoroughly with various scenarios to ensure robust performance.
    
    Remember, building an autonomous agent is an iterative process. Start with a basic version and progressively enhance its capabilities. This approach will help you understand the intricacies of LangGraph while creating a sophisticated AI application.
    
    Do you have any specific ideas about what kind of tasks or domain you want your autonomous agent to specialize in? This could help guide the design and implementation process.
    

Now that we've had the agent take a couple steps, we can `replay` the full state history to see everything that occurred.


```python
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
```

    Num Messages:  8 Next:  ()
    --------------------------------------------------------------------------------
    Num Messages:  7 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  6 Next:  ('tools',)
    --------------------------------------------------------------------------------
    Num Messages:  5 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  4 Next:  ('__start__',)
    --------------------------------------------------------------------------------
    Num Messages:  4 Next:  ()
    --------------------------------------------------------------------------------
    Num Messages:  3 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  2 Next:  ('tools',)
    --------------------------------------------------------------------------------
    Num Messages:  1 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  0 Next:  ('__start__',)
    --------------------------------------------------------------------------------
    

**Notice** that checkpoints are saved for every step of the graph. This __spans invocations__ so you can rewind across a full thread's history. We've picked out `to_replay` as a state to resume from. This is the state after the `chatbot` node in the second graph invocation above.

Resuming from this point should call the **action** node next.


```python
print(to_replay.next)
print(to_replay.config)
```

    ('tools',)
    {'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef7d094-2634-687c-8006-49ddde5b2f1c'}}
    

**Notice** that the checkpoint's config (`to_replay.config`) contains a `checkpoint_id` **timestamp**. Providing this `checkpoint_id` value tells LangGraph's checkpointer to **load** the state from that moment in time. Let's try it below:


```python
# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    [{'text': "That's an excellent idea! Building an autonomous agent with LangGraph is a great way to explore its capabilities and learn about advanced AI application development. LangGraph's features make it well-suited for creating autonomous agents. Let me provide some additional insights and encouragement for your project.", 'type': 'text'}, {'id': 'toolu_017t6BS5rNCzFWcpxRizDKjE', 'input': {'query': 'building autonomous agents with LangGraph examples and tutorials'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_017t6BS5rNCzFWcpxRizDKjE)
     Call ID: toolu_017t6BS5rNCzFWcpxRizDKjE
      Args:
        query: building autonomous agents with LangGraph examples and tutorials
    =================================[1m Tool Message [0m=================================
    Name: tavily_search_results_json
    
    [{"url": "https://blog.langchain.dev/how-to-build-the-ultimate-ai-automation-with-multi-agent-collaboration/", "content": "Learn how to create an autonomous research assistant using LangGraph, an extension of LangChain for agent and multi-agent flows. Follow the steps to define the graph state, initialize the graph, and run the agents for planning, research, review, writing and publishing."}, {"url": "https://medium.com/@lucas.dahan/hands-on-langgraph-building-a-multi-agent-assistant-06aa68ed942f", "content": "Building the Graph. With our agents defined, we'll create a graph.py file to orchestrate their interactions. The basic graph structure in LangGraph is really simple, here we are going to use ..."}]
    ==================================[1m Ai Message [0m==================================
    
    Great choice! Building an autonomous agent with LangGraph is an excellent way to dive deep into its capabilities. Based on the additional information I've found, here are some insights and steps to help you get started:
    
    1. LangGraph for Autonomous Agents:
       LangGraph is particularly well-suited for creating autonomous agents, especially those involving multi-agent collaboration. It allows you to create complex, stateful workflows that can simulate autonomous behavior.
    
    2. Example Project: Autonomous Research Assistant
       One popular example is building an autonomous research assistant. This type of project can help you understand the core concepts of LangGraph while creating something useful.
    
    3. Key Steps in Building an Autonomous Agent:
       a. Define the Graph State: This involves setting up the structure that will hold the agent's state and context.
       b. Initialize the Graph: Set up the initial conditions and parameters for your agent.
       c. Create Multiple Agents: For a complex system, you might create several specialized agents, each with a specific role (e.g., planning, research, review, writing).
       d. Orchestrate Interactions: Use LangGraph to manage how these agents interact and collaborate.
    
    4. Components of an Autonomous Agent:
       - Planning Agent: Determines the overall strategy and steps.
       - Research Agent: Gathers necessary information.
       - Review Agent: Evaluates and refines the work.
       - Writing Agent: Produces the final output.
       - Publishing Agent: Handles the final distribution or application of results.
    
    5. Implementation Tips:
       - Start with a simple graph structure in LangGraph.
       - Define clear roles and responsibilities for each agent or component.
       - Use LangGraph's features to manage state and context across the different stages of your agent's workflow.
    
    6. Learning Resources:
       - Look for tutorials and examples specifically on building multi-agent systems with LangGraph.
       - The LangChain documentation and community forums can be valuable resources, as LangGraph builds upon LangChain.
    
    7. Potential Applications:
       - Autonomous research assistants
       - Complex task automation systems
       - Interactive storytelling agents
       - Autonomous problem-solving systems
    
    Building an autonomous agent with LangGraph is an exciting project that will give you hands-on experience with advanced concepts in AI application development. It's a great way to learn about state management, multi-agent coordination, and complex workflow design in AI systems.
    
    As you embark on this project, remember to start small and gradually increase complexity. You might begin with a simple autonomous agent that performs a specific task, then expand its capabilities and add more agents or components as you become more comfortable with LangGraph.
    
    Do you have a specific type of autonomous agent in mind, or would you like some suggestions for beginner-friendly autonomous agent projects to start with?
    

Notice that the graph resumed execution from the `**action**` node. You can tell this is the case since the first value printed above is the response from our search engine tool.

**Congratulations!** You've now used time-travel checkpoint traversal in LangGraph. Being able to rewind and explore alternative paths opens up a world of possibilities for debugging, experimentation, and interactive applications.

## Conclusion

Congrats! You've completed the intro tutorial and built a chat bot in LangGraph that supports tool calling, persistent memory, human-in-the-loop interactivity, and even time-travel!

The [LangGraph documentation](https://langchain-ai.github.io/langgraph/) is a great resource for diving deeper into the library's capabilities.
