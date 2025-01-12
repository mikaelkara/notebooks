# How to convert LangGraph calls to LangGraph Cloud calls

So you're used to interacting with your graph locally, but now you've deployed it with LangGraph cloud. How do you change all the places in your codebase where you call LangGraph directly to call LangGraph Cloud? This notebook contains side-by-side comparisons so you can easily transition from calling LangGraph to calling LangGraph Cloud.

## Setup

We'll be using a simple ReAct agent for this how-to guide. You will also need to set up a project with `agent.py` and `langgraph.json` files. See [quick start](https://langchain-ai.github.io/langgraph/cloud/quick_start/#develop) for setting this up.


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

    OPENAI_API_KEY:  ········
    


```python
# this is all that's needed for the agent.py
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


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

model = ChatOpenAI(model_name="gpt-4o", temperature=0)
graph = create_react_agent(model, tools)
```

Now we'll set up the langgraph client. The client assumes the LangGraph Cloud server is running on `localhost:8123`


```python
from langgraph_sdk import get_client

client = get_client()
```

## Invoking the graph

Below examples show how to mirror `.invoke() / .ainvoke()` methods of LangGraph's `CompiledGraph` runnable, i.e. create a blocking graph execution

### With LangGraph


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
invoke_output = await graph.ainvoke(inputs)
```


```python
for m in invoke_output["messages"]:
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_GOKlsBY2XKm7pZnmAzJweYDU)
     Call ID: call_GOKlsBY2XKm7pZnmAzJweYDU
      Args:
        city: sf
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is currently sunny.
    

### With LangGraph Cloud


```python
# NOTE: We're not specifying the thread here -- this allows us to create a thread just for this run
wait_output = await client.runs.wait(None, "agent", input=inputs)
```


```python
# we'll use this for pretty message formatting
from langchain_core.messages import convert_to_messages
```


```python
for m in convert_to_messages(wait_output["messages"]):
    m.pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_pQJsT9uLG3nVppN8Dt2OhnFx)
     Call ID: call_pQJsT9uLG3nVppN8Dt2OhnFx
      Args:
        city: sf
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is currently sunny.
    

## Streaming

Below examples show how to mirror `.stream() / .astream()` methods for streaming partial graph execution results.  
Note: LangGraph's `stream_mode=values/updates/debug` behave nearly identically in LangGraph Cloud (with the exception of additional streamed chunks with `metadata` / `end` events types)

### With LangGraph


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph.astream(inputs, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_302y9671bqMkMcpLZOWLNAnq)
     Call ID: call_302y9671bqMkMcpLZOWLNAnq
      Args:
        city: sf
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is currently sunny.
    

### With LangGraph Cloud


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in client.runs.stream(
    None, "agent", input=inputs, stream_mode="values"
):
    if chunk.event == "values":
        messages = convert_to_messages(chunk.data["messages"])
        messages[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_NYVNSiBeF0oTAYnaDrlEAG7a)
     Call ID: call_NYVNSiBeF0oTAYnaDrlEAG7a
      Args:
        city: sf
    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is currently sunny.
    

## Persistence

In LangGraph, you need to provide a `checkpointer` object when compiling your graph to persist state across interactions with your graph (i.e. threads). In LangGraph Cloud, you don't need to create a checkpointer -- the server already implements one for you. You can also directly manage the threads from a client.

### With LangGraph


```python
from langgraph.checkpoint.memory import MemorySaver
```


```python
checkpointer = MemorySaver()
graph_with_memory = create_react_agent(model, tools, checkpointer=checkpointer)
```


```python
inputs = {"messages": [("human", "what's the weather in nyc")]}
invoke_output = await graph_with_memory.ainvoke(
    inputs, config={"configurable": {"thread_id": "1"}}
)
invoke_output["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    The weather in NYC might be cloudy.
    


```python
inputs = {"messages": [("human", "what's it known for?")]}
invoke_output = await graph_with_memory.ainvoke(
    inputs, config={"configurable": {"thread_id": "1"}}
)
invoke_output["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    New York City (NYC) is known for a variety of iconic landmarks, cultural institutions, and vibrant neighborhoods. Some of the most notable things NYC is known for include:
    
    1. **Statue of Liberty**: A symbol of freedom and democracy.
    2. **Times Square**: Famous for its bright lights, Broadway theaters, and bustling atmosphere.
    3. **Central Park**: A large urban park offering a green oasis in the middle of the city.
    4. **Empire State Building**: An iconic skyscraper with an observation deck offering panoramic views of the city.
    5. **Broadway**: Renowned for its world-class theater productions.
    6. **Wall Street**: The financial hub of the United States.
    7. **Museums**: Including the Metropolitan Museum of Art, the Museum of Modern Art (MoMA), and the American Museum of Natural History.
    8. **Diverse Cuisine**: A melting pot of culinary experiences from around the world.
    9. **Cultural Diversity**: A rich tapestry of cultures, languages, and traditions.
    10. **Fashion**: A global fashion capital, home to numerous designers and fashion events.
    
    These are just a few highlights, but NYC offers countless other attractions and experiences.
    


```python
inputs = {"messages": [("human", "what's it known for?")]}
invoke_output = await graph_with_memory.ainvoke(
    inputs, config={"configurable": {"thread_id": "2"}}
)
invoke_output["messages"][-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    Could you please specify what "it" refers to? Are you asking about a specific city, person, object, or something else?
    


```python
# get the state of the thread
checkpointer.get({"configurable": {"thread_id": "2"}})
```




    {'v': 1,
     'ts': '2024-06-22T02:31:49.722569+00:00',
     'id': '1ef303f9-4149-6b56-8001-a80d1e3c9dc6',
     'channel_values': {'messages': [HumanMessage(content="what's it known for?", id='ea0d1672-05e9-4d77-9dff-b33bd5c824e7'),
       AIMessage(content='Could you please specify what "it" refers to? Are you asking about a specific city, person, object, or something else?', response_metadata={'token_usage': {'completion_tokens': 28, 'prompt_tokens': 57, 'total_tokens': 85}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_3e7d703517', 'finish_reason': 'stop', 'logprobs': None}, id='run-f0381dc0-d891-4203-8f77-3155ba17998c-0', usage_metadata={'input_tokens': 57, 'output_tokens': 28, 'total_tokens': 85})],
      'agent': 'agent'},
     'channel_versions': {'__start__': 2,
      'messages': 3,
      'start:agent': 3,
      'agent': 3},
     'versions_seen': {'__start__': {'__start__': 1},
      'agent': {'start:agent': 2},
      'tools': {}},
     'pending_sends': []}



### With LangGraph Cloud

Let's now reproduce the same using LangGraph Cloud. Note that instead of using a checkpointer we just create a new thread on the backend and pass the ID to the API


```python
thread = await client.threads.create()
```


```python
inputs = {"messages": [("human", "what's the weather in nyc")]}
wait_output = await client.runs.wait(thread["thread_id"], "agent", input=inputs)
convert_to_messages(wait_output["messages"])[-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    The weather in NYC might be cloudy.
    


```python
inputs = {"messages": [("human", "what's it known for?")]}
wait_output = await client.runs.wait(thread["thread_id"], "agent", input=inputs)
convert_to_messages(wait_output["messages"])[-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    New York City (NYC) is known for a variety of iconic landmarks, cultural institutions, and vibrant neighborhoods. Some of the most notable things NYC is known for include:
    
    1. **Statue of Liberty**: A symbol of freedom and democracy, located on Liberty Island.
    2. **Times Square**: Known for its bright lights, Broadway theaters, and bustling atmosphere.
    3. **Central Park**: A large urban park offering a green oasis in the middle of the city.
    4. **Empire State Building**: An iconic skyscraper with an observation deck offering panoramic views of the city.
    5. **Broadway**: Famous for its world-class theater productions and musicals.
    6. **Wall Street**: The financial hub of the United States, home to the New York Stock Exchange.
    7. **Museums**: Including the Metropolitan Museum of Art, the Museum of Modern Art (MoMA), and the American Museum of Natural History.
    8. **Diverse Cuisine**: A melting pot of culinary experiences, from street food to Michelin-starred restaurants.
    9. **Cultural Diversity**: A rich tapestry of cultures and communities from around the world.
    10. **Skyscrapers**: A skyline filled with iconic buildings and modern architecture.
    
    NYC is also known for its influence in fashion, media, and entertainment, making it one of the most dynamic and influential cities in the world.
    


```python
thread = await client.threads.create()
```


```python
inputs = {"messages": [("human", "what's it known for?")]}
wait_output = await client.runs.wait(thread["thread_id"], "agent", input=inputs)
convert_to_messages(wait_output["messages"])[-1].pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    Could you please specify what "it" refers to? Are you asking about a specific city, person, object, or something else?
    


```python
# get the state of the thread
await client.threads.get_state(thread["thread_id"])
```




    {'values': {'messages': [{'content': "what's it known for?",
        'additional_kwargs': {},
        'response_metadata': {},
        'type': 'human',
        'name': None,
        'id': '381cd144-b360-4c7d-8177-e2634446993c',
        'example': False},
       {'content': 'Could you please specify what "it" refers to? Are you asking about a specific city, person, object, or something else?',
        'additional_kwargs': {'refusal': None},
        'response_metadata': {'token_usage': {'completion_tokens': 28,
          'prompt_tokens': 57,
          'total_tokens': 85,
          'completion_tokens_details': {'reasoning_tokens': 0}},
         'model_name': 'gpt-4o-2024-05-13',
         'system_fingerprint': 'fp_3537616b13',
         'finish_reason': 'stop',
         'logprobs': None},
        'type': 'ai',
        'name': None,
        'id': 'run-b23c6a05-d6c7-46fd-8b09-443a334feb6b-0',
        'example': False,
        'tool_calls': [],
        'invalid_tool_calls': [],
        'usage_metadata': {'input_tokens': 57,
         'output_tokens': 28,
         'total_tokens': 85}}]},
     'next': [],
     'tasks': [],
     'metadata': {'step': 1,
      'run_id': '1ef78309-0a79-6667-a00b-1a04034cabff',
      'source': 'loop',
      'writes': {'agent': {'messages': [{'id': 'run-b23c6a05-d6c7-46fd-8b09-443a334feb6b-0',
          'name': None,
          'type': 'ai',
          'content': 'Could you please specify what "it" refers to? Are you asking about a specific city, person, object, or something else?',
          'example': False,
          'tool_calls': [],
          'usage_metadata': {'input_tokens': 57,
           'total_tokens': 85,
           'output_tokens': 28},
          'additional_kwargs': {'refusal': None},
          'response_metadata': {'logprobs': None,
           'model_name': 'gpt-4o-2024-05-13',
           'token_usage': {'total_tokens': 85,
            'prompt_tokens': 57,
            'completion_tokens': 28,
            'completion_tokens_details': {'reasoning_tokens': 0}},
           'finish_reason': 'stop',
           'system_fingerprint': 'fp_3537616b13'},
          'invalid_tool_calls': []}]}},
      'parents': {},
      'user_id': '',
      'graph_id': 'agent',
      'thread_id': '14162c42-ab85-404f-a2f9-e7207493f74b',
      'created_by': 'system',
      'run_attempt': 1,
      'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'},
     'created_at': '2024-09-21T15:45:46.150887+00:00',
     'checkpoint_id': '1ef78309-131f-673f-8001-55ea44389e6a',
     'parent_checkpoint_id': '1ef78309-0aab-6271-8000-caaf708a2185'}



## Breakpoints

### With LangGraph


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in graph_with_memory.astream(
    inputs,
    stream_mode="values",
    interrupt_before=["tools"],
    config={"configurable": {"thread_id": "3"}},
):
    chunk["messages"][-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_cYp3BijeW2JNQ9RqJRdkrbMu)
     Call ID: call_cYp3BijeW2JNQ9RqJRdkrbMu
      Args:
        city: sf
    


```python
async for chunk in graph_with_memory.astream(
    None,
    stream_mode="values",
    interrupt_before=["tools"],
    config={"configurable": {"thread_id": "3"}},
):
    chunk["messages"][-1].pretty_print()
```

    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is sunny!
    

### With LangGraph Cloud

Similar to the persistence example, we need to create a thread so we can persist state and continue from the breakpoint.


```python
thread = await client.threads.create()

async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input=inputs,
    stream_mode="values",
    interrupt_before=["tools"],
):
    if chunk.event == "values":
        messages = convert_to_messages(chunk.data["messages"])
        messages[-1].pretty_print()
```

    ================================[1m Human Message [0m=================================
    
    what's the weather in sf
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_weather (call_MVQEJtPYAj1nJ7J6YaCeLX8a)
     Call ID: call_MVQEJtPYAj1nJ7J6YaCeLX8a
      Args:
        city: sf
    


```python
async for chunk in client.runs.stream(
    thread["thread_id"],
    "agent",
    input=None,
    stream_mode="values",
    interrupt_before=["tools"],
):
    if chunk.event == "values":
        messages = convert_to_messages(chunk.data["messages"])
        messages[-1].pretty_print()
```

    =================================[1m Tool Message [0m=================================
    Name: get_weather
    
    It's always sunny in sf
    ==================================[1m Ai Message [0m==================================
    
    The weather in San Francisco is currently sunny.
    

## Steaming events

For streaming events, in LangGraph you need to use `.astream` method on the `CompiledGraph`. In LangGraph Cloud this is done via passing `stream_mode="events"`

### With LangGraph


```python
from langchain_core.messages import AIMessageChunk

inputs = {"messages": [("human", "what's the weather in sf")]}
first = True
async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
    if msg.content:
        print(msg.content, end="|", flush=True)

    if isinstance(msg, AIMessageChunk):
        if first:
            gathered = msg
            first = False
        else:
            gathered = gathered + msg

        if msg.tool_call_chunks:
            print(gathered.tool_calls)
```

    [{'name': 'get_weather', 'args': {}, 'id': 'call_G6boP6Hj21glqPTqtFdTllUd', 'type': 'tool_call'}]
    [{'name': 'get_weather', 'args': {}, 'id': 'call_G6boP6Hj21glqPTqtFdTllUd', 'type': 'tool_call'}]
    [{'name': 'get_weather', 'args': {}, 'id': 'call_G6boP6Hj21glqPTqtFdTllUd', 'type': 'tool_call'}]
    [{'name': 'get_weather', 'args': {'city': ''}, 'id': 'call_G6boP6Hj21glqPTqtFdTllUd', 'type': 'tool_call'}]
    [{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_G6boP6Hj21glqPTqtFdTllUd', 'type': 'tool_call'}]
    [{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_G6boP6Hj21glqPTqtFdTllUd', 'type': 'tool_call'}]
    It's always sunny in sf|The| weather| in| San| Francisco| is| currently| sunny|.|

### With LangGraph Cloud


```python
inputs = {"messages": [("human", "what's the weather in sf")]}
async for chunk in client.runs.stream(
    None, "agent", input=inputs, stream_mode="events"
):
    if chunk.event == "events" and chunk.data["event"] == "on_chat_model_stream":
        print(chunk.data["data"]["chunk"])
```

    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': 'call_JYWaAecaAV92cOlZwRHi9B7M', 'function': {'arguments': '', 'name': 'get_weather'}, 'type': 'function'}]}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [{'name': 'get_weather', 'args': '', 'id': 'call_JYWaAecaAV92cOlZwRHi9B7M', 'error': None}], 'usage_metadata': None, 'tool_call_chunks': [{'name': 'get_weather', 'args': '', 'id': 'call_JYWaAecaAV92cOlZwRHi9B7M', 'index': 0}]}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': '{"', 'name': None}, 'type': None}]}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [{'name': '', 'args': {}, 'id': None}], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': [{'name': None, 'args': '{"', 'id': None, 'index': 0}]}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': 'city', 'name': None}, 'type': None}]}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [{'name': None, 'args': 'city', 'id': None, 'error': None}], 'usage_metadata': None, 'tool_call_chunks': [{'name': None, 'args': 'city', 'id': None, 'index': 0}]}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': '":"', 'name': None}, 'type': None}]}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [{'name': None, 'args': '":"', 'id': None, 'error': None}], 'usage_metadata': None, 'tool_call_chunks': [{'name': None, 'args': '":"', 'id': None, 'index': 0}]}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': 'sf', 'name': None}, 'type': None}]}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [{'name': None, 'args': 'sf', 'id': None, 'error': None}], 'usage_metadata': None, 'tool_call_chunks': [{'name': None, 'args': 'sf', 'id': None, 'index': 0}]}
    {'content': '', 'additional_kwargs': {'tool_calls': [{'index': 0, 'id': None, 'function': {'arguments': '"}', 'name': None}, 'type': None}]}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [{'name': None, 'args': '"}', 'id': None, 'error': None}], 'usage_metadata': None, 'tool_call_chunks': [{'name': None, 'args': '"}', 'id': None, 'index': 0}]}
    {'content': '', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'tool_calls'}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-855fec3d-15df-4ae8-b74d-208a0e463be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': '', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': 'The', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' weather', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' in', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' San', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' Francisco', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' is', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' currently', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' sunny', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': '.', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' Enjoy', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' the', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': ' sunshine', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': '!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    {'content': '', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop'}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-19a0bdff-8724-4730-8052-c3ac89525461', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}
    
