# How to use Postgres checkpointer for persistence

<div class="admonition tip">
    <p class="admonition-title">Prerequisites</p>
    <p>
        This guide assumes familiarity with the following:
        <ul>
            <li>
                <a href="https://langchain-ai.github.io/langgraph/concepts/persistence/">
                    Persistence
                </a>
            </li>       
            <li>
                <a href="https://www.postgresql.org/about/">
                    Postgresql
                </a>
            </li>        
        </ul>
    </p>
</div> 

When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.

This how-to guide shows how to use `Postgres` as the backend for persisting checkpoint state using the [`langgraph-checkpoint-postgres`](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-postgres) library.

For demonstration purposes we add persistence to the [pre-built create react agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent). 

In general, you can add a checkpointer to any custom graph that you build like this:

```python
from langgraph.graph import StateGraph

builder = StateGraph(....)
# ... define the graph
checkpointer = # postgres checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

## Setup

You will need access to a postgres instance. There are many resources online that can help
you set up a postgres instance.

Next, let's install the required packages and set our API keys


```python
%%capture --no-stderr
%pip install -U psycopg psycopg-pool langgraph langgraph-checkpoint-postgres
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

## Define model and tools for the graph


```python
from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


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
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

## Use sync connection

This sets up a synchronous connection to the database. 

Synchronous connections execute operations in a blocking manner, meaning each operation waits for completion before moving to the next one. The `DB_URI` is the database connection URI, with the protocol used for connecting to a PostgreSQL database, authentication, and host where database is running. The connection_kwargs dictionary defines additional parameters for the database connection.


```python
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
```


```python
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
```

### With a connection pool

This manages a pool of reusable database connections: 
- Advantages: Efficient resource utilization, improved performance for frequent connections
- Best for: Applications with many short-lived database operations



```python
from psycopg_pool import ConnectionPool

with ConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = PostgresSaver(pool)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
    checkpoint = checkpointer.get(config)
```


```python
res
```




    {'messages': [HumanMessage(content="what's the weather in sf", id='735b7deb-b0fe-4ad5-8920-2a3c69bbe9f7'),
      AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c56b3e04-08a9-4a59-b3f5-ee52d0ef0656-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
      ToolMessage(content="It's always sunny in sf", name='get_weather', id='0644bf7b-4d1b-4ebe-afa1-d2169ccce582', tool_call_id='call_lJHMDYgfgRdiEAGfFsEhqqKV'),
      AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-1ed9b8d0-9b50-4b87-b3a2-9860f51e9fd1-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}




```python
checkpoint
```




    {'v': 1,
     'id': '1ef559b7-3b19-6ce8-8003-18d0f60634be',
     'ts': '2024-08-08T15:32:42.108605+00:00',
     'current_tasks': {},
     'pending_sends': [],
     'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8',
       'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'},
      'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'},
      '__input__': {},
      '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}},
     'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
      'tools': '00000000000000000000000000000005.',
      'messages': '00000000000000000000000000000005.b9adc75836c78af94af1d6811340dd13',
      '__start__': '00000000000000000000000000000002.',
      'start:agent': '00000000000000000000000000000003.',
      'branch:agent:should_continue:tools': '00000000000000000000000000000004.'},
     'channel_values': {'agent': 'agent',
      'messages': [HumanMessage(content="what's the weather in sf", id='735b7deb-b0fe-4ad5-8920-2a3c69bbe9f7'),
       AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c56b3e04-08a9-4a59-b3f5-ee52d0ef0656-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
       ToolMessage(content="It's always sunny in sf", name='get_weather', id='0644bf7b-4d1b-4ebe-afa1-d2169ccce582', tool_call_id='call_lJHMDYgfgRdiEAGfFsEhqqKV'),
       AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-1ed9b8d0-9b50-4b87-b3a2-9860f51e9fd1-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}



### With a connection

This creates a single, dedicated connection to the database:
- Advantages: Simple to use, suitable for longer transactions
- Best for: Applications with fewer, longer-lived database operations


```python
from psycopg import Connection


with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = PostgresSaver(conn)
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # checkpointer.setup()
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuple = checkpointer.get_tuple(config)
```


```python
checkpoint_tuple
```




    CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4650-6bfc-8003-1c5488f19318'}}, checkpoint={'v': 1, 'id': '1ef559b7-4650-6bfc-8003-1c5488f19318', 'ts': '2024-08-08T15:32:43.284551+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.af9f229d2c4e14f4866eb37f72ec39f6', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in sf", id='7a14f96c-2d88-454f-9520-0e0287a4abbb'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_NcL4dBTYu4kSPGMKdxztdpjN', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-39adbf2c-36ef-40f6-9cad-8e1f8167fc19-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_NcL4dBTYu4kSPGMKdxztdpjN', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='c9f82354-3225-40a8-bf54-81f3e199043b', tool_call_id='call_NcL4dBTYu4kSPGMKdxztdpjN'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-83888be3-d681-42ca-ad67-e2f5ee8550de-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 94, 'prompt_tokens': 84, 'completion_tokens': 10}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-83888be3-d681-42ca-ad67-e2f5ee8550de-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4087-681a-8002-88a5738f76f1'}}, pending_writes=[])



### With a connection string

This creates a connection based on a connection string:
- Advantages: Simplicity, encapsulates connection details
- Best for: Quick setup or when connection details are provided as a string


```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "3"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuples = list(checkpointer.list(config))
```


```python
checkpoint_tuples
```




    [CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-5024-6476-8003-cf0a750e6b37'}}, checkpoint={'v': 1, 'id': '1ef559b7-5024-6476-8003-cf0a750e6b37', 'ts': '2024-08-08T15:32:44.314900+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.3f8b8d9923575b911e17157008ab75ac', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_507c9469a1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='cac4f90a-dc3e-4bfa-940f-1c630289a583', tool_call_id='call_9y3q1BiwW7zGh2gk2faInTRk'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-97d3fb7a-3d2e-4090-84f4-dafdfe44553f-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 94, 'prompt_tokens': 84, 'completion_tokens': 10}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-97d3fb7a-3d2e-4090-84f4-dafdfe44553f-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b3d-6430-8002-b5c99d2eb4db'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b3d-6430-8002-b5c99d2eb4db'}}, checkpoint={'v': 1, 'id': '1ef559b7-4b3d-6430-8002-b5c99d2eb4db', 'ts': '2024-08-08T15:32:43.800857+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'messages': '00000000000000000000000000000004.1195f50946feaedb0bae1fdbfadc806b', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'tools': 'tools', 'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_507c9469a1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='cac4f90a-dc3e-4bfa-940f-1c630289a583', tool_call_id='call_9y3q1BiwW7zGh2gk2faInTRk')]}}, metadata={'step': 2, 'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content="It's always sunny in sf", name='get_weather', id='cac4f90a-dc3e-4bfa-940f-1c630289a583', tool_call_id='call_9y3q1BiwW7zGh2gk2faInTRk')]}}}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b30-6078-8001-eaf8c9bd8844'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b30-6078-8001-eaf8c9bd8844'}}, checkpoint={'v': 1, 'id': '1ef559b7-4b30-6078-8001-eaf8c9bd8844', 'ts': '2024-08-08T15:32:43.795440+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'messages': '00000000000000000000000000000003.bab5fb3a70876f600f5f2fd46945ce5f', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_507c9469a1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})], 'branch:agent:should_continue:tools': 'agent'}}, metadata={'step': 1, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"sf"}'}}]}, response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 71, 'prompt_tokens': 57, 'completion_tokens': 14}, 'finish_reason': 'tool_calls', 'system_fingerprint': 'fp_507c9469a1'}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})]}}}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46d7-6116-8000-8976b7c89a2f'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46d7-6116-8000-8976b7c89a2f'}}, checkpoint={'v': 1, 'id': '1ef559b7-46d7-6116-8000-8976b7c89a2f', 'ts': '2024-08-08T15:32:43.339573+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'messages': '00000000000000000000000000000002.ba0c90d32863686481f7fe5eab9ecdf0', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84')], 'start:agent': '__start__'}}, metadata={'step': 0, 'source': 'loop', 'writes': None}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46ce-6c64-bfff-ef7fe2663573'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46ce-6c64-bfff-ef7fe2663573'}}, checkpoint={'v': 1, 'id': '1ef559b7-46ce-6c64-bfff-ef7fe2663573', 'ts': '2024-08-08T15:32:43.336188+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'channel_values': {'__start__': {'messages': [['human', "what's the weather in sf"]]}}}, metadata={'step': -1, 'source': 'input', 'writes': {'messages': [['human', "what's the weather in sf"]]}}, parent_config=None, pending_writes=None)]



## Use async connection

This sets up an asynchronous connection to the database. 

Async connections allow non-blocking database operations. This means other parts of your application can continue running while waiting for database operations to complete. It's particularly useful in high-concurrency scenarios or when dealing with I/O-bound operations.

### With a connection pool


```python
from psycopg_pool import AsyncConnectionPool

async with AsyncConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = AsyncPostgresSaver(pool)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    await checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "4"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )

    checkpoint = await checkpointer.aget(config)
```


```python
checkpoint
```




    {'v': 1,
     'id': '1ef559b7-5cc9-6460-8003-8655824c0944',
     'ts': '2024-08-08T15:32:45.640793+00:00',
     'current_tasks': {},
     'pending_sends': [],
     'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8',
       'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'},
      'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'},
      '__input__': {},
      '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}},
     'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
      'tools': '00000000000000000000000000000005.',
      'messages': '00000000000000000000000000000005.d869fc7231619df0db74feed624efe41',
      '__start__': '00000000000000000000000000000002.',
      'start:agent': '00000000000000000000000000000003.',
      'branch:agent:should_continue:tools': '00000000000000000000000000000004.'},
     'channel_values': {'agent': 'agent',
      'messages': [HumanMessage(content="what's the weather in nyc", id='d883b8a0-99de-486d-91a2-bcfa7f25dc05'),
       AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_H6TAYfyd6AnaCrkQGs6Q2fVp', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-6f542f84-ad73-444c-8ef7-b5ea75a2e09b-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_H6TAYfyd6AnaCrkQGs6Q2fVp', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}),
       ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='c0e52254-77a4-4ea9-a2b7-61dd2d65ec68', tool_call_id='call_H6TAYfyd6AnaCrkQGs6Q2fVp'),
       AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-977140d4-7582-40c3-b2b6-31b542c430a3-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}



### With a connection


```python
from psycopg import AsyncConnection

async with await AsyncConnection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = AsyncPostgresSaver(conn)
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "5"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuple = await checkpointer.aget_tuple(config)
```


```python
checkpoint_tuple
```




    CheckpointTuple(config={'configurable': {'thread_id': '5', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-65b4-60ca-8003-1ef4b620559a'}}, checkpoint={'v': 1, 'id': '1ef559b7-65b4-60ca-8003-1ef4b620559a', 'ts': '2024-08-08T15:32:46.575814+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.1557a6006d58f736d5cb2dd5c5f10111', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in nyc", id='935e7732-b288-49bd-9ec2-1f7610cc38cb'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_94KtjtPmsiaj7T8yXvL7Ef31', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-790c929a-7982-49e7-af67-2cbe4a86373b-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_94KtjtPmsiaj7T8yXvL7Ef31', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='b2dc1073-abc4-4492-8982-434a7e32e445', tool_call_id='call_94KtjtPmsiaj7T8yXvL7Ef31'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-7e8a7f16-d8e1-457a-89f3-192102396449-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 97, 'prompt_tokens': 88, 'completion_tokens': 9}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-7e8a7f16-d8e1-457a-89f3-192102396449-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}}, parent_config={'configurable': {'thread_id': '5', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-62ae-6128-8002-c04af82bcd41'}}, pending_writes=[])



### With a connection string


```python
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "6"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]
```


```python
checkpoint_tuples
```




    [CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-723c-67de-8003-63bd4eab35af'}}, checkpoint={'v': 1, 'id': '1ef559b7-723c-67de-8003-63bd4eab35af', 'ts': '2024-08-08T15:32:47.890003+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.b6fe2a26011590cfe8fd6a39151a9e92', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='798c520f-4f9a-4f6d-a389-da721eb4d4ce', tool_call_id='call_QIFCuh4zfP9owpjToycJiZf7'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-4a34e05d-8bcf-41ad-adc3-715919fde64c-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 97, 'prompt_tokens': 88, 'completion_tokens': 9}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-4a34e05d-8bcf-41ad-adc3-715919fde64c-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6bf5-63c6-8002-ed990dbbc96e'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6bf5-63c6-8002-ed990dbbc96e'}}, checkpoint={'v': 1, 'id': '1ef559b7-6bf5-63c6-8002-ed990dbbc96e', 'ts': '2024-08-08T15:32:47.231667+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'messages': '00000000000000000000000000000004.c9074f2a41f05486b5efb86353dc75c0', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'tools': 'tools', 'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='798c520f-4f9a-4f6d-a389-da721eb4d4ce', tool_call_id='call_QIFCuh4zfP9owpjToycJiZf7')]}}, metadata={'step': 2, 'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='798c520f-4f9a-4f6d-a389-da721eb4d4ce', tool_call_id='call_QIFCuh4zfP9owpjToycJiZf7')]}}}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6be0-6926-8001-1a8ce73baf9e'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6be0-6926-8001-1a8ce73baf9e'}}, checkpoint={'v': 1, 'id': '1ef559b7-6be0-6926-8001-1a8ce73baf9e', 'ts': '2024-08-08T15:32:47.223198+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'messages': '00000000000000000000000000000003.097b5407d709b297591f1ef5d50c8368', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})], 'branch:agent:should_continue:tools': 'agent'}}, metadata={'step': 1, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"nyc"}'}}]}, response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 73, 'prompt_tokens': 58, 'completion_tokens': 15}, 'finish_reason': 'tool_calls', 'system_fingerprint': 'fp_48196bc67a'}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})]}}}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-663d-60b4-8000-10a8922bffbf'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-663d-60b4-8000-10a8922bffbf'}}, checkpoint={'v': 1, 'id': '1ef559b7-663d-60b4-8000-10a8922bffbf', 'ts': '2024-08-08T15:32:46.631935+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'messages': '00000000000000000000000000000002.2a79db8da664e437bdb25ea804457ca7', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396')], 'start:agent': '__start__'}}, metadata={'step': 0, 'source': 'loop', 'writes': None}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6637-6d4e-bfff-6cecf690c3cb'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6637-6d4e-bfff-6cecf690c3cb'}}, checkpoint={'v': 1, 'id': '1ef559b7-6637-6d4e-bfff-6cecf690c3cb', 'ts': '2024-08-08T15:32:46.629806+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'channel_values': {'__start__': {'messages': [['human', "what's the weather in nyc"]]}}}, metadata={'step': -1, 'source': 'input', 'writes': {'messages': [['human', "what's the weather in nyc"]]}}, parent_config=None, pending_writes=None)]


