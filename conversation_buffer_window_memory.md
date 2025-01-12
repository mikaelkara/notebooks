# Migrating off ConversationBufferWindowMemory or ConversationTokenBufferMemory

Follow this guide if you're trying to migrate off one of the old memory classes listed below:


| Memory Type                      | Description                                                                                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ConversationBufferWindowMemory` | Keeps the last `n` messages of the conversation. Drops the oldest messages when there are more than `n` messages.                                                                      |
| `ConversationTokenBufferMemory`  | Keeps only the most recent messages in the conversation under the constraint that the total number of tokens in the conversation does not exceed a certain limit. |

`ConversationBufferWindowMemory` and `ConversationTokenBufferMemory` apply additional processing on top of the raw conversation history to trim the conversation history to a size that fits inside the context window of a chat model. 

This processing functionality can be accomplished using LangChain's built-in [trim_messages](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html) function.

:::important

We’ll begin by exploring a straightforward method that involves applying processing logic to the entire conversation history.

While this approach is easy to implement, it has a downside: as the conversation grows, so does the latency, since the logic is re-applied to all previous exchanges in the conversation at each turn.

More advanced strategies focus on incrementally updating the conversation history to avoid redundant processing.

For instance, the langgraph [how-to guide on summarization](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) demonstrates
how to maintain a running summary of the conversation while discarding older messages, ensuring they aren't re-processed during later turns.
:::

## Set up


```python
%%capture --no-stderr
%pip install --upgrade --quiet langchain-openai langchain
```


```python
import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass()
```

## Legacy usage with LLMChain / Conversation Chain

<details open>


```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate(
    [
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

# highlight-start
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
# highlight-end

legacy_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=prompt,
    # highlight-next-line
    memory=memory,
)

legacy_result = legacy_chain.invoke({"text": "my name is bob"})
print(legacy_result)

legacy_result = legacy_chain.invoke({"text": "what was my name"})
print(legacy_result)
```

    {'text': 'Nice to meet you, Bob! How can I assist you today?', 'chat_history': []}
    {'text': 'Your name is Bob. How can I assist you further, Bob?', 'chat_history': [HumanMessage(content='my name is bob', additional_kwargs={}, response_metadata={}), AIMessage(content='Nice to meet you, Bob! How can I assist you today?', additional_kwargs={}, response_metadata={})]}
    

</details>

## Reimplementing ConversationBufferWindowMemory logic

Let's first create appropriate logic to process the conversation history, and then we'll see how to integrate it into an application. You can later replace this basic setup with more advanced logic tailored to your specific needs.

We'll use `trim_messages` to implement logic that keeps the last `n` messages of the conversation. It will drop the oldest messages when the number of messages exceeds `n`.

In addition, we will also keep the system message if it's present -- when present, it's the first message in a conversation that includes instructions for the chat model.


```python
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("why is 42 always the answer?"),
    AIMessage(
        "Because it’s the only number that’s constantly right, even when it doesn’t add up!"
    ),
    HumanMessage("What did the cow say?"),
]
```


```python
from langchain_core.messages import trim_messages

selected_messages = trim_messages(
    messages,
    token_counter=len,  # <-- len will simply count the number of messages rather than tokens
    max_tokens=5,  # <-- allow up to 5 messages.
    strategy="last",
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    # start_on="human" makes sure we produce a valid chat history
    start_on="human",
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
    allow_partial=False,
)

for msg in selected_messages:
    msg.pretty_print()
```

    ================================[1m System Message [0m================================
    
    you're a good assistant, you always respond with a joke.
    ==================================[1m Ai Message [0m==================================
    
    Hmmm let me think.
    
    Why, he's probably chasing after the last cup of coffee in the office!
    ================================[1m Human Message [0m=================================
    
    why is 42 always the answer?
    ==================================[1m Ai Message [0m==================================
    
    Because it’s the only number that’s constantly right, even when it doesn’t add up!
    ================================[1m Human Message [0m=================================
    
    What did the cow say?
    

## Reimplementing ConversationTokenBufferMemory logic

Here, we'll use `trim_messages` to keeps the system message and the most recent messages in the conversation under the constraint that the total number of tokens in the conversation does not exceed a certain limit. 



```python
from langchain_core.messages import trim_messages

selected_messages = trim_messages(
    messages,
    # Please see API reference for trim_messages for other ways to specify a token counter.
    token_counter=ChatOpenAI(model="gpt-4o"),
    max_tokens=80,  # <-- token limit
    # The start_on is specified
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    # start_on="human" makes sure we produce a valid chat history
    start_on="human",
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
    strategy="last",
)

for msg in selected_messages:
    msg.pretty_print()
```

    ================================[1m System Message [0m================================
    
    you're a good assistant, you always respond with a joke.
    ================================[1m Human Message [0m=================================
    
    why is 42 always the answer?
    ==================================[1m Ai Message [0m==================================
    
    Because it’s the only number that’s constantly right, even when it doesn’t add up!
    ================================[1m Human Message [0m=================================
    
    What did the cow say?
    

## Modern usage with LangGraph

The example below shows how to use LangGraph to add simple conversation pre-processing logic.

:::note

If you want to avoid running the computation on the entire conversation history each time, you can follow
the [how-to guide on summarization](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) that demonstrates
how to discard older messages, ensuring they aren't re-processed during later turns.

:::

<details open>


```python
import uuid

from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define a chat model
model = ChatOpenAI()


# Define the function that calls the model
def call_model(state: MessagesState):
    # highlight-start
    selected_messages = trim_messages(
        state["messages"],
        token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        max_tokens=5,  # <-- allow up to 5 messages.
        strategy="last",
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )

    # highlight-end
    response = model.invoke(selected_messages)
    # We return a list, because this will get added to the existing list
    return {"messages": response}


# Define the two nodes we will cycle between
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)


# Adding memory is straight forward in langgraph!
# highlight-next-line
memory = MemorySaver()

app = workflow.compile(
    # highlight-next-line
    checkpointer=memory
)


# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
thread_id = uuid.uuid4()
# highlight-next-line
config = {"configurable": {"thread_id": thread_id}}

input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Here, let's confirm that the AI remembers our name!
config = {"configurable": {"thread_id": thread_id}}
input_message = HumanMessage(content="what was my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    hi! I'm bob
    ==================================[1m Ai Message [0m==================================
    
    Hello Bob! How can I assist you today?
    ================================[1m Human Message [0m=================================
    
    what was my name?
    ==================================[1m Ai Message [0m==================================
    
    Your name is Bob. How can I help you, Bob?
    

</details>

## Usage with a pre-built langgraph agent

This example shows usage of an Agent Executor with a pre-built agent constructed using the [create_tool_calling_agent](https://api.python.langchain.com/en/latest/agents/langchain.agents.tool_calling_agent.base.create_tool_calling_agent.html) function.

If you are using one of the [old LangChain pre-built agents](https://python.langchain.com/v0.1/docs/modules/agents/agent_types/), you should be able
to replace that code with the new [langgraph pre-built agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/) which leverages
native tool calling capabilities of chat models and will likely work better out of the box.

<details open>


```python
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


@tool
def get_user_age(name: str) -> str:
    """Use this tool to find the user's age."""
    # This is a placeholder for the actual implementation
    if "bob" in name.lower():
        return "42 years old"
    return "41 years old"


memory = MemorySaver()
model = ChatOpenAI()


# highlight-start
def state_modifier(state) -> list[BaseMessage]:
    """Given the agent state, return a list of messages for the chat model."""
    # We're using the message processor defined above.
    return trim_messages(
        state["messages"],
        token_counter=len,  # <-- len will simply count the number of messages rather than tokens
        max_tokens=5,  # <-- allow up to 5 messages.
        strategy="last",
        # Most chat models expect that chat history starts with either:
        # (1) a HumanMessage or
        # (2) a SystemMessage followed by a HumanMessage
        # start_on="human" makes sure we produce a valid chat history
        start_on="human",
        # Usually, we want to keep the SystemMessage
        # if it's present in the original history.
        # The SystemMessage has special instructions for the model.
        include_system=True,
        allow_partial=False,
    )


# highlight-end

app = create_react_agent(
    model,
    tools=[get_user_age],
    checkpointer=memory,
    # highlight-next-line
    state_modifier=state_modifier,
)

# The thread id is a unique key that identifies
# this particular conversation.
# We'll just generate a random uuid here.
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Tell the AI that our name is Bob, and ask it to use a tool to confirm
# that it's capable of working like an agent.
input_message = HumanMessage(content="hi! I'm bob. What is my age?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# Confirm that the chat bot has access to previous conversation
# and can respond to the user saying that the user's name is Bob.
input_message = HumanMessage(content="do you remember my name?")

for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    hi! I'm bob. What is my age?
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_user_age (call_jsMvoIFv970DhqqLCJDzPKsp)
     Call ID: call_jsMvoIFv970DhqqLCJDzPKsp
      Args:
        name: bob
    =================================[1m Tool Message [0m=================================
    Name: get_user_age
    
    42 years old
    ==================================[1m Ai Message [0m==================================
    
    Bob, you are 42 years old.
    ================================[1m Human Message [0m=================================
    
    do you remember my name?
    ==================================[1m Ai Message [0m==================================
    
    Yes, your name is Bob.
    

</details>

## LCEL: Add a preprocessing step

The simplest way to add complex conversation management is by introducing a pre-processing step in front of the chat model and pass the full conversation history to the pre-processing step.

This approach is conceptually simple and will work in many situations; for example, if using a [RunnableWithMessageHistory](/docs/how_to/message_history/) instead of wrapping the chat model, wrap the chat model with the pre-processor.

The obvious downside of this approach is that latency starts to increase as the conversation history grows because of two reasons:

1. As the conversation gets longer, more data may need to be fetched from whatever store your'e using to store the conversation history (if not storing it in memory).
2. The pre-processing logic will end up doing a lot of redundant computation, repeating computation from previous steps of the conversation.

:::caution

If you want to use a chat model's tool calling capabilities, remember to bind the tools to the model before adding the history pre-processing step to it!

:::

<details open>


```python
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI()


@tool
def what_did_the_cow_say() -> str:
    """Check to see what the cow said."""
    return "foo"


# highlight-start
message_processor = trim_messages(  # Returns a Runnable if no messages are provided
    token_counter=len,  # <-- len will simply count the number of messages rather than tokens
    max_tokens=5,  # <-- allow up to 5 messages.
    strategy="last",
    # The start_on is specified
    # to make sure we do not generate a sequence where
    # a ToolMessage that contains the result of a tool invocation
    # appears before the AIMessage that requested a tool invocation
    # as this will cause some chat models to raise an error.
    start_on=("human", "ai"),
    include_system=True,  # <-- Keep the system message
    allow_partial=False,
)
# highlight-end

# Note that we bind tools to the model first!
model_with_tools = model.bind_tools([what_did_the_cow_say])

# highlight-next-line
model_with_preprocessor = message_processor | model_with_tools

full_history = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("why is 42 always the answer?"),
    AIMessage(
        "Because it’s the only number that’s constantly right, even when it doesn’t add up!"
    ),
    HumanMessage("What did the cow say?"),
]


# We pass it explicity to the model_with_preprocesor for illustrative purposes.
# If you're using `RunnableWithMessageHistory` the history will be automatically
# read from the source the you configure.
model_with_preprocessor.invoke(full_history).pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      what_did_the_cow_say (call_urHTB5CShhcKz37QiVzNBlIS)
     Call ID: call_urHTB5CShhcKz37QiVzNBlIS
      Args:
    

</details>

If you need to implement more efficient logic and want to use `RunnableWithMessageHistory` for now the way to achieve this
is to subclass from [BaseChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html) and
define appropriate logic for `add_messages` (that doesn't simply append the history, but instead re-writes it).

Unless you have a good reason to implement this solution, you should instead use LangGraph.

## Next steps

Explore persistence with LangGraph:

* [LangGraph quickstart tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
* [How to add persistence ("memory") to your graph](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
* [How to manage conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
* [How to add summary of the conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/)

Add persistence with simple LCEL (favor langgraph for more complex use cases):

* [How to add message history](/docs/how_to/message_history/)

Working with message history:

* [How to trim messages](/docs/how_to/trim_messages)
* [How to filter messages](/docs/how_to/filter_messages/)
* [How to merge message runs](/docs/how_to/merge_message_runs/)



```python

```
