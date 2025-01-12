---
keywords: [create_react_agent, create_react_agent()]
---
# How to migrate from legacy LangChain agents to LangGraph

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Agents](/docs/concepts/agents)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Tool calling](/docs/how_to/tool_calling/)

:::

Here we focus on how to move from legacy LangChain agents to more flexible [LangGraph](https://langchain-ai.github.io/langgraph/) agents.
LangChain agents (the [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor) in particular) have multiple configuration parameters.
In this notebook we will show how those parameters map to the LangGraph react agent executor using the [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) prebuilt helper method.


:::note
In LangGraph, the graph replaces LangChain's agent executor. It manages the agent's cycles and tracks the scratchpad as messages within its state. The LangChain "agent" corresponds to the state_modifier and LLM you've provided.
:::


#### Prerequisites

This how-to guide uses OpenAI as the LLM. Install the dependencies to run.


```python
%%capture --no-stderr
%pip install -U langgraph langchain langchain-openai
```

Then, set your OpenAI API key.


```python
import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key:\n")
```

## Basic Usage

For basic creation and usage of a tool-calling ReAct-style agent, the functionality is the same. First, let's define a model and tool(s), then we'll use those to create an agent.


```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


query = "what is the value of magic_function(3)?"
```

For the LangChain [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), we define a prompt with a placeholder for the agent's scratchpad. The agent can be invoked as follows:


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})
```




    {'input': 'what is the value of magic_function(3)?',
     'output': 'The value of `magic_function(3)` is 5.'}



LangGraph's [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) manages a state that is defined by a list of messages. It will continue to process the list until there are no tool calls in the agent's output. To kick it off, we input a list of messages. The output will contain the entire state of the graph-- in this case, the conversation history.




```python
from langgraph.prebuilt import create_react_agent

langgraph_agent_executor = create_react_agent(model, tools)


messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
{
    "input": query,
    "output": messages["messages"][-1].content,
}
```




    {'input': 'what is the value of magic_function(3)?',
     'output': 'The value of `magic_function(3)` is 5.'}




```python
message_history = messages["messages"]

new_query = "Pardon?"

messages = langgraph_agent_executor.invoke(
    {"messages": message_history + [("human", new_query)]}
)
{
    "input": new_query,
    "output": messages["messages"][-1].content,
}
```




    {'input': 'Pardon?',
     'output': 'The result of applying `magic_function` to the input value 3 is 5.'}



## Prompt Templates

With legacy LangChain agents you have to pass in a prompt template. You can use this to control the agent.

With LangGraph [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent), by default there is no prompt. You can achieve similar control over the agent in a few ways:

1. Pass in a system message as input
2. Initialize the agent with a system message
3. Initialize the agent with a function to transform messages before passing to the model.

Let's take a look at all of these below. We will pass in custom instructions to get the agent to respond in Spanish.

First up, using `AgentExecutor`:


```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})
```




    {'input': 'what is the value of magic_function(3)?',
     'output': 'El valor de magic_function(3) es 5.'}



Now, let's pass a custom system message to [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent).

LangGraph's prebuilt `create_react_agent` does not take a prompt template directly as a parameter, but instead takes a [`state_modifier`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) parameter. This modifies the graph state before the llm is called, and can be one of four values:

- A `SystemMessage`, which is added to the beginning of the list of messages.
- A `string`, which is converted to a `SystemMessage` and added to the beginning of the list of messages.
- A `Callable`, which should take in full graph state. The output is then passed to the language model.
- Or a [`Runnable`](/docs/concepts/lcel), which should take in full graph state. The output is then passed to the language model.

Here's how it looks in action:


```python
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

system_message = "You are a helpful assistant. Respond only in Spanish."
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

langgraph_agent_executor = create_react_agent(
    model, tools, state_modifier=system_message
)


messages = langgraph_agent_executor.invoke({"messages": [("user", query)]})
```

We can also pass in an arbitrary function. This function should take in a list of messages and output a list of messages.
We can do all types of arbitrary formatting of messages here. In this cases, let's just add a SystemMessage to the start of the list of messages.


```python
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("placeholder", "{messages}"),
    ]
)


def _modify_state_messages(state: AgentState):
    return prompt.invoke({"messages": state["messages"]}).to_messages() + [
        ("user", "Also say 'Pandamonium!' after the answer.")
    ]


langgraph_agent_executor = create_react_agent(
    model, tools, state_modifier=_modify_state_messages
)


messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
print(
    {
        "input": query,
        "output": messages["messages"][-1].content,
    }
)
```

    {'input': 'what is the value of magic_function(3)?', 'output': 'El valor de magic_function(3) es 5. ¡Pandamonium!'}
    

## Memory

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could add chat [Memory](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.memory) so it can engage in a multi-turn conversation.


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
memory = InMemoryChatMessageHistory(session_id="test-session")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # First put the history
        ("placeholder", "{chat_history}"),
        # Then the new input
        ("human", "{input}"),
        # Finally the scratchpad
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "test-session"}}
print(
    agent_with_chat_history.invoke(
        {"input": "Hi, I'm polly! What's the output of magic_function of 3?"}, config
    )["output"]
)
print("---")
print(agent_with_chat_history.invoke({"input": "Remember my name?"}, config)["output"])
print("---")
print(
    agent_with_chat_history.invoke({"input": "what was that output again?"}, config)[
        "output"
    ]
)
```

    The output of the magic function when the input is 3 is 5.
    ---
    Yes, you mentioned your name is Polly.
    ---
    The output of the magic function when the input is 3 is 5.
    

### In LangGraph

Memory is just [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/), aka [checkpointing](https://langchain-ai.github.io/langgraph/reference/checkpoints/).

Add a `checkpointer` to the agent and you get chat memory for free.


```python
from langgraph.checkpoint.memory import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent

system_message = "You are a helpful assistant."
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

memory = MemorySaver()
langgraph_agent_executor = create_react_agent(
    model, tools, state_modifier=system_message, checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread"}}
print(
    langgraph_agent_executor.invoke(
        {
            "messages": [
                ("user", "Hi, I'm polly! What's the output of magic_function of 3?")
            ]
        },
        config,
    )["messages"][-1].content
)
print("---")
print(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "Remember my name?")]}, config
    )["messages"][-1].content
)
print("---")
print(
    langgraph_agent_executor.invoke(
        {"messages": [("user", "what was that output again?")]}, config
    )["messages"][-1].content
)
```

    The output of the magic function for the input 3 is 5.
    ---
    Yes, you mentioned that your name is Polly.
    ---
    The output of the magic function for the input 3 was 5.
    

## Iterating through steps

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could iterate over the steps using the [stream](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream) (or async `astream`) methods or the [iter](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter) method. LangGraph supports stepwise iteration using [stream](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream) 


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for step in agent_executor.stream({"input": query}):
    print(step)
```

    {'actions': [ToolAgentAction(tool='magic_function', tool_input={'input': 3}, log="\nInvoking: `magic_function` with `{'input': 3}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7'}, id='run-7a3a5ada-52ec-4df0-bf7d-81e5051b01b4', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='call_yyetzabaDBRX9Ml2KyqfKzZM')], 'messages': [AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7'}, id='run-7a3a5ada-52ec-4df0-bf7d-81e5051b01b4', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'index': 0, 'type': 'tool_call_chunk'}])]}
    {'steps': [AgentStep(action=ToolAgentAction(tool='magic_function', tool_input={'input': 3}, log="\nInvoking: `magic_function` with `{'input': 3}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7'}, id='run-7a3a5ada-52ec-4df0-bf7d-81e5051b01b4', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_yyetzabaDBRX9Ml2KyqfKzZM', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='call_yyetzabaDBRX9Ml2KyqfKzZM'), observation=5)], 'messages': [FunctionMessage(content='5', additional_kwargs={}, response_metadata={}, name='magic_function')]}
    {'output': 'The value of `magic_function(3)` is 5.', 'messages': [AIMessage(content='The value of `magic_function(3)` is 5.', additional_kwargs={}, response_metadata={})]}
    

### In LangGraph

In LangGraph, things are handled natively using [stream](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.stream) or the asynchronous `astream` method.


```python
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}"),
    ]
)


def _modify_state_messages(state: AgentState):
    return prompt.invoke({"messages": state["messages"]}).to_messages()


langgraph_agent_executor = create_react_agent(
    model, tools, state_modifier=_modify_state_messages
)

for step in langgraph_agent_executor.stream(
    {"messages": [("human", query)]}, stream_mode="updates"
):
    print(step)
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IHTMrjvIHn8gFOX42FstIpr9', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 61, 'total_tokens': 75, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-1a6970da-163a-4e4d-b9b7-7e73b1057f42-0', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_IHTMrjvIHn8gFOX42FstIpr9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 61, 'output_tokens': 14, 'total_tokens': 75, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]}}
    {'tools': {'messages': [ToolMessage(content='5', name='magic_function', id='51a9d3e4-734d-426f-a5a1-c6597e4efe25', tool_call_id='call_IHTMrjvIHn8gFOX42FstIpr9')]}}
    {'agent': {'messages': [AIMessage(content='The value of `magic_function(3)` is 5.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 84, 'total_tokens': 98, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a20a4ee344', 'finish_reason': 'stop', 'logprobs': None}, id='run-73001576-a3dc-4552-8d81-c9ce8aec05b3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 14, 'total_tokens': 98, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]}}
    

## `return_intermediate_steps`

### In LangChain

Setting this parameter on AgentExecutor allows users to access intermediate_steps, which pairs agent actions (e.g., tool invocations) with their outcomes.



```python
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
result = agent_executor.invoke({"input": query})
print(result["intermediate_steps"])
```

    [(ToolAgentAction(tool='magic_function', tool_input={'input': 3}, log="\nInvoking: `magic_function` with `{'input': 3}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_njTvl2RsVf4q1aMUxoYnJuK1', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7'}, id='run-c9dfe3ab-2db6-4592-851e-89e056aeab32', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_njTvl2RsVf4q1aMUxoYnJuK1', 'type': 'tool_call'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_njTvl2RsVf4q1aMUxoYnJuK1', 'index': 0, 'type': 'tool_call_chunk'}])], tool_call_id='call_njTvl2RsVf4q1aMUxoYnJuK1'), 5)]
    

### In LangGraph

By default the [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) in LangGraph appends all messages to the central state. Therefore, it is easy to see any intermediate steps by just looking at the full state.


```python
from langgraph.prebuilt import create_react_agent

langgraph_agent_executor = create_react_agent(model, tools=tools)

messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})

messages
```




    {'messages': [HumanMessage(content='what is the value of magic_function(3)?', additional_kwargs={}, response_metadata={}, id='1abb52c2-4bc2-4d82-bd32-5a24c3976b0f'),
      AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_XfQD6C7rAalcmicQubkhJVFq', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 55, 'total_tokens': 69, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a20a4ee344', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-34f02786-5b5c-4bb1-bd9e-406c81944a24-0', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_XfQD6C7rAalcmicQubkhJVFq', 'type': 'tool_call'}], usage_metadata={'input_tokens': 55, 'output_tokens': 14, 'total_tokens': 69, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}),
      ToolMessage(content='5', name='magic_function', id='cbc9fadf-1962-4ed7-b476-348c774652be', tool_call_id='call_XfQD6C7rAalcmicQubkhJVFq'),
      AIMessage(content='The value of `magic_function(3)` is 5.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 78, 'total_tokens': 92, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'stop', 'logprobs': None}, id='run-547e03d2-872d-4008-a38d-b7f739a77df5-0', usage_metadata={'input_tokens': 78, 'output_tokens': 14, 'total_tokens': 92, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]}



## `max_iterations`

### In LangChain

`AgentExecutor` implements a `max_iterations` parameter, allowing users to abort a run that exceeds a specified number of iterations.


```python
@tool
def magic_function(input: str) -> str:
    """Applies a magic function to an input."""
    return "Sorry, there was an error. Please try again."


tools = [magic_function]
```


```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
)

agent_executor.invoke({"input": query})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mLo siento, no puedo decirte directamente el valor de `magic_function(3)`. Si deseas, puedo usar la función mágica para calcularlo. ¿Te gustaría que lo hiciera?[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'what is the value of magic_function(3)?',
     'output': 'Lo siento, no puedo decirte directamente el valor de `magic_function(3)`. Si deseas, puedo usar la función mágica para calcularlo. ¿Te gustaría que lo hiciera?'}



### In LangGraph

In LangGraph this is controlled via `recursion_limit` configuration parameter.

Note that in `AgentExecutor`, an "iteration" includes a full turn of tool invocation and execution. In LangGraph, each step contributes to the recursion limit, so we will need to multiply by two (and add one) to get equivalent results.

If the recursion limit is reached, LangGraph raises a specific exception type, that we can catch and manage similarly to AgentExecutor.


```python
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

RECURSION_LIMIT = 2 * 3 + 1

langgraph_agent_executor = create_react_agent(model, tools=tools)

try:
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
except GraphRecursionError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
```

    content='what is the value of magic_function(3)?' additional_kwargs={} response_metadata={} id='c2489fe8-e69c-4163-876d-3cce26b28521'
    content='' additional_kwargs={'tool_calls': [{'id': 'call_OyNTcO6SDAvZcBlIEknPRrTR', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 55, 'total_tokens': 69, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-b65504bb-fa23-4f8a-8d6c-7edb6d16e7ff-0' tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_OyNTcO6SDAvZcBlIEknPRrTR', 'type': 'tool_call'}] usage_metadata={'input_tokens': 55, 'output_tokens': 14, 'total_tokens': 69, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
    content='Sorry, there was an error. Please try again.' name='magic_function' id='f00e0bff-54fe-4726-a1a7-127a59d8f7ed' tool_call_id='call_OyNTcO6SDAvZcBlIEknPRrTR'
    content="It seems there was an error when trying to compute the value of the magic function with input 3. Let's try again." additional_kwargs={'tool_calls': [{'id': 'call_Q020rQoJh4cnh8WglIMnDm4z', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 40, 'prompt_tokens': 88, 'total_tokens': 128, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-556d8cb2-b47a-4826-b17d-b520982c2475-0' tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_Q020rQoJh4cnh8WglIMnDm4z', 'type': 'tool_call'}] usage_metadata={'input_tokens': 88, 'output_tokens': 40, 'total_tokens': 128, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
    content='Sorry, there was an error. Please try again.' name='magic_function' id='777212cd-8381-44db-9762-3f81951ea73e' tool_call_id='call_Q020rQoJh4cnh8WglIMnDm4z'
    content="It seems there is a persistent issue in computing the value of the magic function with the input 3. Unfortunately, I can't provide the value at this time. If you have any other questions or need further assistance, feel free to ask!" additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 49, 'prompt_tokens': 150, 'total_tokens': 199, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'stop', 'logprobs': None} id='run-92ec0b90-bc8e-4851-9139-f1d976145ab7-0' usage_metadata={'input_tokens': 150, 'output_tokens': 49, 'total_tokens': 199, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
    

## `max_execution_time`

### In LangChain

`AgentExecutor` implements a `max_execution_time` parameter, allowing users to abort a run that exceeds a total time limit.


```python
import time


@tool
def magic_function(input: str) -> str:
    """Applies a magic function to an input."""
    time.sleep(2.5)
    return "Sorry, there was an error. Please try again."


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=2,
    verbose=True,
)

agent_executor.invoke({"input": query})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mLo siento, no tengo la capacidad de evaluar directamente una función llamada "magic_function" con el valor 3. Sin embargo, si me proporcionas más detalles sobre qué hace la función o cómo está definida, podría intentar ayudarte a comprender su comportamiento o resolverlo de otra manera.[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'what is the value of magic_function(3)?',
     'output': 'Lo siento, no tengo la capacidad de evaluar directamente una función llamada "magic_function" con el valor 3. Sin embargo, si me proporcionas más detalles sobre qué hace la función o cómo está definida, podría intentar ayudarte a comprender su comportamiento o resolverlo de otra manera.'}



### In LangGraph

With LangGraph's react agent, you can control timeouts on two levels. 

You can set a `step_timeout` to bound each **step**:


```python
from langgraph.prebuilt import create_react_agent

langgraph_agent_executor = create_react_agent(model, tools=tools)
# Set the max timeout for each step here
langgraph_agent_executor.step_timeout = 2

try:
    for chunk in langgraph_agent_executor.stream({"messages": [("human", query)]}):
        print(chunk)
        print("------")
except TimeoutError:
    print({"input": query, "output": "Agent stopped due to a step timeout."})
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_UuxSgpGaqzX84sNlKzCVOiRO', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 55, 'total_tokens': 69, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-24c94cbd-2962-48cf-a447-af888eb6ef86-0', tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_UuxSgpGaqzX84sNlKzCVOiRO', 'type': 'tool_call'}], usage_metadata={'input_tokens': 55, 'output_tokens': 14, 'total_tokens': 69, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]}}
    ------
    {'input': 'what is the value of magic_function(3)?', 'output': 'Agent stopped due to a step timeout.'}
    

The other way to set a single max timeout for an entire run is to directly use the python stdlib [asyncio](https://docs.python.org/3/library/asyncio.html) library.


```python
import asyncio

from langgraph.prebuilt import create_react_agent

langgraph_agent_executor = create_react_agent(model, tools=tools)


async def stream(langgraph_agent_executor, inputs):
    async for chunk in langgraph_agent_executor.astream(
        {"messages": [("human", query)]}
    ):
        print(chunk)
        print("------")


try:
    task = asyncio.create_task(
        stream(langgraph_agent_executor, {"messages": [("human", query)]})
    )
    await asyncio.wait_for(task, timeout=3)
except asyncio.TimeoutError:
    print("Task Cancelled.")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_km17xvoY7wJ5yNnXhb5V9D3I', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 55, 'total_tokens': 69, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_45c6de4934', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b44a04e5-9b68-4020-be36-98de1593eefc-0', tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_km17xvoY7wJ5yNnXhb5V9D3I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 55, 'output_tokens': 14, 'total_tokens': 69, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})]}}
    ------
    Task Cancelled.
    

## `early_stopping_method`

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could configure an [early_stopping_method](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.early_stopping_method) to either return a string saying "Agent stopped due to iteration limit or time limit." (`"force"`) or prompt the LLM a final time to respond (`"generate"`).


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return "Sorry there was an error, please try again."


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, early_stopping_method="force", max_iterations=1
)

result = agent_executor.invoke({"input": query})
print("Output with early_stopping_method='force':")
print(result["output"])
```

    Output with early_stopping_method='force':
    Agent stopped due to max iterations.
    

### In LangGraph

In LangGraph, you can explicitly handle the response behavior outside the agent, since the full state can be accessed.


```python
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

RECURSION_LIMIT = 2 * 1 + 1

langgraph_agent_executor = create_react_agent(model, tools=tools)

try:
    for chunk in langgraph_agent_executor.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
except GraphRecursionError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
```

    content='what is the value of magic_function(3)?' additional_kwargs={} response_metadata={} id='81fd2e50-1e6a-4871-87aa-b7c1225913a4'
    content='' additional_kwargs={'tool_calls': [{'id': 'call_aaEzj3aO1RTnB0uoc9rYUIhi', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 55, 'total_tokens': 69, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-476bc4b1-b7bf-4607-a31c-ddf09dc814c5-0' tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_aaEzj3aO1RTnB0uoc9rYUIhi', 'type': 'tool_call'}] usage_metadata={'input_tokens': 55, 'output_tokens': 14, 'total_tokens': 69, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
    content='Sorry there was an error, please try again.' name='magic_function' id='dcbe7e3e-0ed4-467d-a729-2f45916ff44f' tool_call_id='call_aaEzj3aO1RTnB0uoc9rYUIhi'
    content="It seems there was an error when trying to compute the value of `magic_function(3)`. Let's try that again." additional_kwargs={'tool_calls': [{'id': 'call_jr4R8uJn2pdXF5GZC2Dg3YWS', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}], 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 40, 'prompt_tokens': 87, 'total_tokens': 127, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_a7d06e42a7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-d94b8932-6e9e-4ab1-99f7-7dca89887ffe-0' tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_jr4R8uJn2pdXF5GZC2Dg3YWS', 'type': 'tool_call'}] usage_metadata={'input_tokens': 87, 'output_tokens': 40, 'total_tokens': 127, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
    {'input': 'what is the value of magic_function(3)?', 'output': 'Agent stopped due to max iterations.'}
    

## `trim_intermediate_steps`

### In LangChain

With LangChain's [AgentExecutor](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), you could trim the intermediate steps of long-running agents using [trim_intermediate_steps](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.trim_intermediate_steps), which is either an integer (indicating the agent should keep the last N steps) or a custom function.

For instance, we could trim the value so the agent only sees the most recent intermediate step.


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


magic_step_num = 1


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    global magic_step_num
    print(f"Call number: {magic_step_num}")
    magic_step_num += 1
    return input + magic_step_num


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)


def trim_steps(steps: list):
    # Let's give the agent amnesia
    return []


agent_executor = AgentExecutor(
    agent=agent, tools=tools, trim_intermediate_steps=trim_steps
)


query = "Call the magic function 4 times in sequence with the value 3. You cannot call it multiple times at once."

for step in agent_executor.stream({"input": query}):
    pass
```

    Call number: 1
    Call number: 2
    Call number: 3
    Call number: 4
    Call number: 5
    Call number: 6
    Call number: 7
    Call number: 8
    Call number: 9
    Call number: 10
    Call number: 11
    Call number: 12
    Call number: 13
    Call number: 14
    

    Stopping agent prematurely due to triggering stop condition
    

    Call number: 15
    

### In LangGraph

We can use the [`state_modifier`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) just as before when passing in [prompt templates](#prompt-templates).


```python
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

magic_step_num = 1


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    global magic_step_num
    print(f"Call number: {magic_step_num}")
    magic_step_num += 1
    return input + magic_step_num


tools = [magic_function]


def _modify_state_messages(state: AgentState):
    # Give the agent amnesia, only keeping the original user query
    return [("system", "You are a helpful assistant"), state["messages"][0]]


langgraph_agent_executor = create_react_agent(
    model, tools, state_modifier=_modify_state_messages
)

try:
    for step in langgraph_agent_executor.stream(
        {"messages": [("human", query)]}, stream_mode="updates"
    ):
        pass
except GraphRecursionError as e:
    print("Stopping agent prematurely due to triggering stop condition")
```

    Call number: 1
    Call number: 2
    Call number: 3
    Call number: 4
    Call number: 5
    Call number: 6
    Call number: 7
    Call number: 8
    Call number: 9
    Call number: 10
    Call number: 11
    Call number: 12
    Stopping agent prematurely due to triggering stop condition
    

## Next steps

You've now learned how to migrate your LangChain agent executors to LangGraph.

Next, check out other [LangGraph how-to guides](https://langchain-ai.github.io/langgraph/how-tos/).
