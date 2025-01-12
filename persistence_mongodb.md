# How to create a custom checkpointer using MongoDB

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
                <a href="https://www.mongodb.com/">
                    MongoDB
                </a>
            </li>        
        </ul>
    </p>
</div> 

When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions. 

This reference implementation shows how to use MongoDB as the backend for persisting checkpoint state. Make sure that you have MongoDB running on port `27017` for going through this guide.

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        This is a **reference** implementation. You can implement your own checkpointer using a different database or modify this one as long as it conforms to the <a href="https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.base.BaseCheckpointSaver">BaseCheckpointSaver</a> interface.
    </p>
</div>

For demonstration purposes we add persistence to the [pre-built create react agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent).

In general, you can add a checkpointer to any custom graph that you build like this:

```python
from langgraph.graph import StateGraph

builder = StateGraph(....)
# ... define the graph
checkpointer = # mongodb checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

## Setup

First let's install the required packages and set our API keys


```python
%%capture --no-stderr
%pip install -U pymongo motor langgraph
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

## Checkpointer implementation

### MongoDBSaver

Below is an implementation of MongoDBSaver (for synchronous use of graph, i.e. `.invoke()`, `.stream()`). MongoDBSaver implements four methods that are required for any checkpointer:

- `.put` - Store a checkpoint with its configuration and metadata.
- `.put_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- `.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`).
- `.list` - List checkpoints that match a given configuration and filter criteria.


```python
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient, UpdateOne
from pymongo.database import Database as MongoDatabase

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)


class MongoDBSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a MongoDB database."""

    client: MongoClient
    db: MongoDatabase

    def __init__(
        self,
        client: MongoClient,
        db_name: str,
    ) -> None:
        super().__init__()
        self.client = client
        self.db = self.client[db_name]

    @classmethod
    @contextmanager
    def from_conn_info(
        cls, *, host: str, port: int, db_name: str
    ) -> Iterator["MongoDBSaver"]:
        client = None
        try:
            client = MongoClient(host=host, port=port)
            yield MongoDBSaver(client, db_name)
        finally:
            if client:
                client.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the MongoDB database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            query = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        result = self.db["checkpoints"].find(query).sort("checkpoint_id", -1).limit(1)
        for doc in result:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            serialized_writes = self.db["checkpoint_writes"].find(config_values)
            pending_writes = [
                (
                    doc["task_id"],
                    doc["channel"],
                    self.serde.loads_typed((doc["type"], doc["value"])),
                )
                for doc in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the MongoDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        query = {}
        if config is not None:
            query = {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
            }

        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before is not None:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        result = self.db["checkpoints"].find(query).sort("checkpoint_id", -1)

        if limit is not None:
            result = result.limit(limit)
        for doc in result:
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the MongoDB database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.serde.dumps(metadata),
        }
        upsert_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        # Perform your operations here
        self.db["checkpoints"].update_one(upsert_query, {"$set": doc}, upsert=True)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the MongoDB database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        operations = []
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
            }
            type_, serialized_value = self.serde.dumps_typed(value)
            operations.append(
                UpdateOne(
                    upsert_query,
                    {
                        "$set": {
                            "channel": channel,
                            "type": type_,
                            "value": serialized_value,
                        }
                    },
                    upsert=True,
                )
            )
        self.db["checkpoint_writes"].bulk_write(operations)
```

### AsyncMongoDBSaver

Below is a reference implementation of AsyncMongoDBSaver (for asynchronous use of graph, i.e. `.ainvoke()`, `.astream()`). AsyncMongoDBSaver implements four methods that are required for any async checkpointer:

- `.aput` - Store a checkpoint with its configuration and metadata.
- `.aput_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- `.aget_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`).
- `.alist` - List checkpoints that match a given configuration and filter criteria.


```python
class AsyncMongoDBSaver(BaseCheckpointSaver):
    """A checkpoint saver that stores checkpoints in a MongoDB database asynchronously."""

    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase

    def __init__(
        self,
        client: AsyncIOMotorClient,
        db_name: str,
    ) -> None:
        super().__init__()
        self.client = client
        self.db = self.client[db_name]

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls, *, host: str, port: int, db_name: str
    ) -> AsyncIterator["AsyncMongoDBSaver"]:
        client = None
        try:
            client = AsyncIOMotorClient(host=host, port=port)
            yield AsyncMongoDBSaver(client, db_name)
        finally:
            if client:
                client.close()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from the database asynchronously.

        This method retrieves a checkpoint tuple from the MongoDB database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        if checkpoint_id := get_checkpoint_id(config):
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        else:
            query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }

        result = self.db["checkpoints"].find(query).sort("checkpoint_id", -1).limit(1)
        async for doc in result:
            config_values = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            serialized_writes = self.db["checkpoint_writes"].find(config_values)
            pending_writes = [
                (
                    doc["task_id"],
                    doc["channel"],
                    self.serde.loads_typed((doc["type"], doc["value"])),
                )
                async for doc in serialized_writes
            ]
            return CheckpointTuple(
                {"configurable": config_values},
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
                pending_writes,
            )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from the database asynchronously.

        This method retrieves a list of checkpoint tuples from the MongoDB database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        query = {}
        if config is not None:
            query = {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
            }

        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before is not None:
            query["checkpoint_id"] = {"$lt": before["configurable"]["checkpoint_id"]}

        result = self.db["checkpoints"].find(query).sort("checkpoint_id", -1)

        if limit is not None:
            result = result.limit(limit)
        async for doc in result:
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": doc["checkpoint_id"],
                    }
                },
                checkpoint,
                self.serde.loads(doc["metadata"]),
                (
                    {
                        "configurable": {
                            "thread_id": doc["thread_id"],
                            "checkpoint_ns": doc["checkpoint_ns"],
                            "checkpoint_id": doc["parent_checkpoint_id"],
                        }
                    }
                    if doc.get("parent_checkpoint_id")
                    else None
                ),
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to the MongoDB database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        doc = {
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized_checkpoint,
            "metadata": self.serde.dumps(metadata),
        }
        upsert_query = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        # Perform your operations here
        await self.db["checkpoints"].update_one(
            upsert_query, {"$set": doc}, upsert=True
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        operations = []
        for idx, (channel, value) in enumerate(writes):
            upsert_query = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "idx": idx,
            }
            type_, serialized_value = self.serde.dumps_typed(value)
            operations.append(
                UpdateOne(
                    upsert_query,
                    {
                        "$set": {
                            "channel": channel,
                            "type": type_,
                            "value": serialized_value,
                        }
                    },
                    upsert=True,
                )
            )
        await self.db["checkpoint_writes"].bulk_write(operations)
```

## Setup model and tools for the graph


```python
from typing import Literal
from langchain_core.runnables import ConfigurableField
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
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

## Use sync connection


```python
with MongoDBSaver.from_conn_info(
    host="localhost", port=27017, db_name="checkpoints"
) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    latest_checkpoint = checkpointer.get(config)
    latest_checkpoint_tuple = checkpointer.get_tuple(config)
    checkpoint_tuples = list(checkpointer.list(config))
```


```python
latest_checkpoint
```




    {'v': 1,
     'ts': '2024-08-09T16:19:39.102711+00:00',
     'id': '1ef566b2-d2a8-6cdc-8003-cc4d1980d188',
     'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f4227353-e0e5-43a9-984a-e4b9e2d8e7b8'),
       AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd1d3187-470f-4ebd-938f-527a61824045-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
       ToolMessage(content="It's always sunny in sf", name='get_weather', id='2d124101-696d-450f-bc9f-d8fdcc564101', tool_call_id='call_Y7PzHb7LrIdiTnO5UiSfelt3'),
       AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-87c76dd2-33f4-433e-986a-9405cfe88c88-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})],
      'agent': 'agent'},
     'channel_versions': {'__start__': 2,
      'messages': 5,
      'start:agent': 3,
      'agent': 5,
      'branch:agent:should_continue:tools': 4,
      'tools': 5},
     'versions_seen': {'__input__': {},
      '__start__': {'__start__': 1},
      'agent': {'start:agent': 2, 'tools': 4},
      'tools': {'branch:agent:should_continue:tools': 3}},
     'pending_sends': [],
     'current_tasks': {}}




```python
latest_checkpoint_tuple
```




    CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-d2a8-6cdc-8003-cc4d1980d188'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:39.102711+00:00', 'id': '1ef566b2-d2a8-6cdc-8003-cc4d1980d188', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f4227353-e0e5-43a9-984a-e4b9e2d8e7b8'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd1d3187-470f-4ebd-938f-527a61824045-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='2d124101-696d-450f-bc9f-d8fdcc564101', tool_call_id='call_Y7PzHb7LrIdiTnO5UiSfelt3'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-87c76dd2-33f4-433e-986a-9405cfe88c88-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})], 'agent': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 5, 'start:agent': 3, 'agent': 5, 'branch:agent:should_continue:tools': 4, 'tools': 5}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2, 'tools': 4}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-87c76dd2-33f4-433e-986a-9405cfe88c88-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-cdf7-6b98-8002-997748cc5052'}}, pending_writes=[])




```python
checkpoint_tuples
```




    [CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-d2a8-6cdc-8003-cc4d1980d188'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:39.102711+00:00', 'id': '1ef566b2-d2a8-6cdc-8003-cc4d1980d188', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f4227353-e0e5-43a9-984a-e4b9e2d8e7b8'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd1d3187-470f-4ebd-938f-527a61824045-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='2d124101-696d-450f-bc9f-d8fdcc564101', tool_call_id='call_Y7PzHb7LrIdiTnO5UiSfelt3'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-87c76dd2-33f4-433e-986a-9405cfe88c88-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})], 'agent': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 5, 'start:agent': 3, 'agent': 5, 'branch:agent:should_continue:tools': 4, 'tools': 5}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2, 'tools': 4}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-87c76dd2-33f4-433e-986a-9405cfe88c88-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-cdf7-6b98-8002-997748cc5052'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-cdf7-6b98-8002-997748cc5052'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:38.610752+00:00', 'id': '1ef566b2-cdf7-6b98-8002-997748cc5052', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f4227353-e0e5-43a9-984a-e4b9e2d8e7b8'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd1d3187-470f-4ebd-938f-527a61824045-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='2d124101-696d-450f-bc9f-d8fdcc564101', tool_call_id='call_Y7PzHb7LrIdiTnO5UiSfelt3')], 'tools': 'tools'}, 'channel_versions': {'__start__': 2, 'messages': 4, 'start:agent': 3, 'agent': 4, 'branch:agent:should_continue:tools': 4, 'tools': 4}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content="It's always sunny in sf", name='get_weather', id='2d124101-696d-450f-bc9f-d8fdcc564101', tool_call_id='call_Y7PzHb7LrIdiTnO5UiSfelt3')]}}, 'step': 2}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-cde3-6c60-8001-28d4cc36978d'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-cde3-6c60-8001-28d4cc36978d'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:38.602590+00:00', 'id': '1ef566b2-cde3-6c60-8001-28d4cc36978d', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f4227353-e0e5-43a9-984a-e4b9e2d8e7b8'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd1d3187-470f-4ebd-938f-527a61824045-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})], 'agent': 'agent', 'branch:agent:should_continue:tools': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 3, 'start:agent': 3, 'agent': 3, 'branch:agent:should_continue:tools': 3}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-cd1d3187-470f-4ebd-938f-527a61824045-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_Y7PzHb7LrIdiTnO5UiSfelt3', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})]}}, 'step': 1}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-c72c-6fca-8000-aac6e4f4b809'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-c72c-6fca-8000-aac6e4f4b809'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:37.898584+00:00', 'id': '1ef566b2-c72c-6fca-8000-aac6e4f4b809', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f4227353-e0e5-43a9-984a-e4b9e2d8e7b8')], 'start:agent': '__start__'}, 'channel_versions': {'__start__': 2, 'messages': 2, 'start:agent': 2}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': None, 'step': 0}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-c72a-6af4-bfff-919b9dc6abfe'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b2-c72a-6af4-bfff-919b9dc6abfe'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:37.897642+00:00', 'id': '1ef566b2-c72a-6af4-bfff-919b9dc6abfe', 'channel_values': {'messages': [], '__start__': {'messages': [['human', "what's the weather in sf"]]}}, 'channel_versions': {'__start__': 1}, 'versions_seen': {'__input__': {}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'input', 'writes': {'messages': [['human', "what's the weather in sf"]]}, 'step': -1}, parent_config=None, pending_writes=None)]



## Use async connection


```python
async with AsyncMongoDBSaver.from_conn_info(
    host="localhost", port=27017, db_name="checkpoints"
) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )

    latest_checkpoint = await checkpointer.aget(config)
    latest_checkpoint_tuple = await checkpointer.aget_tuple(config)
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]
```


```python
latest_checkpoint
```




    {'v': 1,
     'ts': '2024-08-09T16:19:48.212051+00:00',
     'id': '1ef566b3-2988-664c-8003-5974c59c6bda',
     'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='1ae4b12f-b1cb-4d55-a754-42cf1c2fbcd5'),
       AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b5da58b5-8f75-485d-af29-bfdeb09b0d94-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}),
       ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='56d4e46b-6cb3-4efe-b369-27b666e62348', tool_call_id='call_IJvXEELx7Ir3kASCqr9dbvhU'),
       AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-dcacbc70-b213-4ddc-ac08-c0d17b2766d8-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})],
      'agent': 'agent'},
     'channel_versions': {'__start__': 2,
      'messages': 5,
      'start:agent': 3,
      'agent': 5,
      'branch:agent:should_continue:tools': 4,
      'tools': 5},
     'versions_seen': {'__input__': {},
      '__start__': {'__start__': 1},
      'agent': {'start:agent': 2, 'tools': 4},
      'tools': {'branch:agent:should_continue:tools': 3}},
     'pending_sends': [],
     'current_tasks': {}}




```python
latest_checkpoint_tuple
```




    CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-2988-664c-8003-5974c59c6bda'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:48.212051+00:00', 'id': '1ef566b3-2988-664c-8003-5974c59c6bda', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='1ae4b12f-b1cb-4d55-a754-42cf1c2fbcd5'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b5da58b5-8f75-485d-af29-bfdeb09b0d94-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='56d4e46b-6cb3-4efe-b369-27b666e62348', tool_call_id='call_IJvXEELx7Ir3kASCqr9dbvhU'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-dcacbc70-b213-4ddc-ac08-c0d17b2766d8-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})], 'agent': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 5, 'start:agent': 3, 'agent': 5, 'branch:agent:should_continue:tools': 4, 'tools': 5}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2, 'tools': 4}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-dcacbc70-b213-4ddc-ac08-c0d17b2766d8-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-23c9-64ea-8002-036c32979035'}}, pending_writes=[])




```python
checkpoint_tuples
```




    [CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-2988-664c-8003-5974c59c6bda'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:48.212051+00:00', 'id': '1ef566b3-2988-664c-8003-5974c59c6bda', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='1ae4b12f-b1cb-4d55-a754-42cf1c2fbcd5'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b5da58b5-8f75-485d-af29-bfdeb09b0d94-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='56d4e46b-6cb3-4efe-b369-27b666e62348', tool_call_id='call_IJvXEELx7Ir3kASCqr9dbvhU'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-dcacbc70-b213-4ddc-ac08-c0d17b2766d8-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})], 'agent': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 5, 'start:agent': 3, 'agent': 5, 'branch:agent:should_continue:tools': 4, 'tools': 5}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2, 'tools': 4}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-dcacbc70-b213-4ddc-ac08-c0d17b2766d8-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-23c9-64ea-8002-036c32979035'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-23c9-64ea-8002-036c32979035'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:47.609498+00:00', 'id': '1ef566b3-23c9-64ea-8002-036c32979035', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='1ae4b12f-b1cb-4d55-a754-42cf1c2fbcd5'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b5da58b5-8f75-485d-af29-bfdeb09b0d94-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='56d4e46b-6cb3-4efe-b369-27b666e62348', tool_call_id='call_IJvXEELx7Ir3kASCqr9dbvhU')], 'tools': 'tools'}, 'channel_versions': {'__start__': 2, 'messages': 4, 'start:agent': 3, 'agent': 4, 'branch:agent:should_continue:tools': 4, 'tools': 4}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='56d4e46b-6cb3-4efe-b369-27b666e62348', tool_call_id='call_IJvXEELx7Ir3kASCqr9dbvhU')]}}, 'step': 2}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-23b5-6de6-8001-a39c8ce6fd93'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-23b5-6de6-8001-a39c8ce6fd93'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:47.601527+00:00', 'id': '1ef566b3-23b5-6de6-8001-a39c8ce6fd93', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='1ae4b12f-b1cb-4d55-a754-42cf1c2fbcd5'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b5da58b5-8f75-485d-af29-bfdeb09b0d94-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})], 'agent': 'agent', 'branch:agent:should_continue:tools': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 3, 'start:agent': 3, 'agent': 3, 'branch:agent:should_continue:tools': 3}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b5da58b5-8f75-485d-af29-bfdeb09b0d94-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_IJvXEELx7Ir3kASCqr9dbvhU', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})]}}, 'step': 1}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-1d6c-6a02-8000-158f156ffce3'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-1d6c-6a02-8000-158f156ffce3'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:46.942389+00:00', 'id': '1ef566b3-1d6c-6a02-8000-158f156ffce3', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='1ae4b12f-b1cb-4d55-a754-42cf1c2fbcd5')], 'start:agent': '__start__'}, 'channel_versions': {'__start__': 2, 'messages': 2, 'start:agent': 2}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': None, 'step': 0}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-1d67-61e2-bfff-d91abbcc3a09'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef566b3-1d67-61e2-bfff-d91abbcc3a09'}}, checkpoint={'v': 1, 'ts': '2024-08-09T16:19:46.940133+00:00', 'id': '1ef566b3-1d67-61e2-bfff-d91abbcc3a09', 'channel_values': {'messages': [], '__start__': {'messages': [['human', "what's the weather in nyc"]]}}, 'channel_versions': {'__start__': 1}, 'versions_seen': {'__input__': {}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'input', 'writes': {'messages': [['human', "what's the weather in nyc"]]}, 'step': -1}, parent_config=None, pending_writes=None)]


