# How to add summary of the conversation history

One of the most common use cases for persistence is to use it to keep track of conversation history. This is great - it makes it easy to continue conversations. As conversations get longer and longer, however, this conversation history can build up and take up more and more of the context window. This can often be undesirable as it leads to more expensive and longer calls to the LLM, and potentially ones that error. One way to work around that is to create a summary of the conversation to date, and use that with the past N messages. This guide will go through an example of how to do that.

This will involve a few steps:
- Check if the conversation is too long (can be done by checking number of messages or length of messages)
- If yes, the create summary (will need a prompt for this)
- Then remove all except the last N messages

A big part of this is deleting old messages. For an in depth guide on how to do that, see [this guide](../delete-messages)

## Setup

First, let's set up the packages we're going to want to use


```python
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_anthropic
```

Next, we need to set API keys for Anthropic (the LLM we will use)


```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

<div class="admonition tip">
    <p class="admonition-title">Set up <a href="https://smith.langchain.com">LangSmith</a> for LangGraph development</p>
    <p style="padding-top: 5px;">
        Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started <a href="https://docs.smith.langchain.com">here</a>. 
    </p>
</div>

## Build the chatbot

Let's now build the chatbot.


```python
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END

memory = MemorySaver()


# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str


# We will use this model for both the conversation and the summarization
model = ChatAnthropic(model_name="claude-3-haiku-20240307")


# Define the logic to call the model
def call_model(state: State):
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END


def summarize_conversation(state: State):
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", END)

# Finally, we compile it!
app = workflow.compile(checkpointer=memory)
```

## Using the graph


```python
def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])
```


```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="hi! I'm bob")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the celtics!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

    ================================[1m Human Message [0m=================================
    
    hi! I'm bob
    ==================================[1m Ai Message [0m==================================
    
    It's nice to meet you, Bob! I'm an AI assistant created by Anthropic. How can I help you today?
    ================================[1m Human Message [0m=================================
    
    what's my name?
    ==================================[1m Ai Message [0m==================================
    
    Your name is Bob, as you told me at the beginning of our conversation.
    ================================[1m Human Message [0m=================================
    
    i like the celtics!
    ==================================[1m Ai Message [0m==================================
    
    That's great, the Celtics are a fun team to follow! Basketball is an exciting sport. Do you have a favorite Celtics player or a favorite moment from a Celtics game you've watched? I'd be happy to discuss the team and the sport with you.
    

We can see that so far no summarization has happened - this is because there are only six messages in the list.


```python
values = app.get_state(config).values
values
```




    {'messages': [HumanMessage(content="hi! I'm bob", id='6534853d-b8a7-44b9-837b-eb7abaf7ebf7'),
      AIMessage(content="It's nice to meet you, Bob! I'm an AI assistant created by Anthropic. How can I help you today?", response_metadata={'id': 'msg_015wCFew2vwMQJcpUh2VZ5ah', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 12, 'output_tokens': 30}}, id='run-0d33008b-1094-4f5e-94ce-293283fc3024-0'),
      HumanMessage(content="what's my name?", id='0a4f203a-b95a-42a9-b1c5-bb20f68b3251'),
      AIMessage(content='Your name is Bob, as you told me at the beginning of our conversation.', response_metadata={'id': 'msg_01PLp8wg2xDsJbNR9uCtxcGz', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 50, 'output_tokens': 19}}, id='run-3815dd4d-ee0c-4fc2-9889-f6dd40325961-0'),
      HumanMessage(content='i like the celtics!', id='ac128172-42d1-4390-b7cc-7bcb2d22ee48'),
      AIMessage(content="That's great, the Celtics are a fun team to follow! Basketball is an exciting sport. Do you have a favorite Celtics player or a favorite moment from a Celtics game you've watched? I'd be happy to discuss the team and the sport with you.", response_metadata={'id': 'msg_01CSg5avZEx6CKcZsSvSVXpr', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 78, 'output_tokens': 61}}, id='run-698faa28-0f72-495f-8ebe-e948664d2200-0')]}



Now let's send another message in


```python
input_message = HumanMessage(content="i like how much they win")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

    ================================[1m Human Message [0m=================================
    
    i like how much they win
    ==================================[1m Ai Message [0m==================================
    
    That's understandable, the Celtics have been one of the more successful NBA franchises over the years. Their history of winning championships is very impressive. It's always fun to follow a team that regularly competes for titles. What do you think has been the key to the Celtics' sustained success? Is there a particular era or team that stands out as your favorite?
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    Here is a summary of our conversation so far:
    
    - You introduced yourself as Bob and said you like the Boston Celtics basketball team.
    - I acknowledged that it's nice to meet you, Bob, and noted that you had shared your name earlier in the conversation.
    - You expressed that you like how much the Celtics win, and I agreed that their history of sustained success and championship pedigree is impressive.
    - I asked if you have a favorite Celtics player or moment that stands out to you, and invited further discussion about the team and the sport of basketball.
    - The overall tone has been friendly and conversational, with me trying to engage with your interest in the Celtics by asking follow-up questions.
    

If we check the state now, we can see that we have a summary of the conversation, as well as the last two messages


```python
values = app.get_state(config).values
values
```




    {'messages': [HumanMessage(content='i like how much they win', id='bb916ce7-534c-4d48-9f92-e269f9dc4859'),
      AIMessage(content="That's understandable, the Celtics have been one of the more successful NBA franchises over the years. Their history of winning championships is very impressive. It's always fun to follow a team that regularly competes for titles. What do you think has been the key to the Celtics' sustained success? Is there a particular era or team that stands out as your favorite?", response_metadata={'id': 'msg_01B7TMagaM8xBnYXLSMwUDAG', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 148, 'output_tokens': 82}}, id='run-c5aa9a8f-7983-4a7f-9c1e-0c0055334ac1-0')],
     'summary': "Here is a summary of our conversation so far:\n\n- You introduced yourself as Bob and said you like the Boston Celtics basketball team.\n- I acknowledged that it's nice to meet you, Bob, and noted that you had shared your name earlier in the conversation.\n- You expressed that you like how much the Celtics win, and I agreed that their history of sustained success and championship pedigree is impressive.\n- I asked if you have a favorite Celtics player or moment that stands out to you, and invited further discussion about the team and the sport of basketball.\n- The overall tone has been friendly and conversational, with me trying to engage with your interest in the Celtics by asking follow-up questions."}



We can now resume having a conversation! Note that even though we only have the last two messages, we can still ask it questions about things mentioned earlier in the conversation (because we summarized those)


```python
input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

    ================================[1m Human Message [0m=================================
    
    what's my name?
    ==================================[1m Ai Message [0m==================================
    
    In our conversation so far, you introduced yourself as Bob. I acknowledged that earlier when you had shared your name.
    


```python
input_message = HumanMessage(content="what NFL team do you think I like?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

    ================================[1m Human Message [0m=================================
    
    what NFL team do you think I like?
    ==================================[1m Ai Message [0m==================================
    
    I don't actually have any information about what NFL team you might like. In our conversation so far, you've only mentioned that you're a fan of the Boston Celtics basketball team. I don't have any prior knowledge about your preferences for NFL teams. Unless you provide me with that information, I don't have a basis to guess which NFL team you might be a fan of.
    


```python
input_message = HumanMessage(content="i like the patriots!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

    ================================[1m Human Message [0m=================================
    
    i like the patriots!
    ==================================[1m Ai Message [0m==================================
    
    Okay, got it! Thanks for sharing that you're also a fan of the New England Patriots in the NFL. That makes sense, given your interest in other Boston sports teams like the Celtics. The Patriots have also had a very successful run over the past couple of decades, winning multiple Super Bowls. It's fun to follow winning franchises like the Celtics and Patriots. Do you have a favorite Patriots player or moment that stands out to you?
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    ================================[1m Remove Message [0m================================
    
    
    Okay, extending the summary with the new information:
    
    - You initially introduced yourself as Bob and said you like the Boston Celtics basketball team. 
    - I acknowledged that and we discussed your appreciation for the Celtics' history of winning.
    - You then asked what your name was, and I reminded you that you had introduced yourself as Bob earlier in the conversation.
    - You followed up by asking what NFL team I thought you might like, and I explained that I didn't have any prior information about your NFL team preferences.
    - You then revealed that you are also a fan of the New England Patriots, which made sense given your Celtics fandom.
    - I responded positively to this new information, noting the Patriots' own impressive success and dynasty over the past couple of decades.
    - I then asked if you have a particular favorite Patriots player or moment that stands out to you, continuing the friendly, conversational tone.
    
    Overall, the discussion has focused on your sports team preferences, with you sharing that you are a fan of both the Celtics and the Patriots. I've tried to engage with your interests and ask follow-up questions to keep the dialogue flowing.
    
