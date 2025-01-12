# How to create a custom checkpointer using Redis

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
                <a href="https://redis.io/">
                    Redis
                </a>
            </li>        
        </ul>
    </p>
</div> 

When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.

This reference implementation shows how to use Redis as the backend for persisting checkpoint state. Make sure that you have Redis running on port `6379` for going through this guide.

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
checkpointer = # redis checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

## Setup

First, let's install the required packages and set our API keys


```python
%%capture --no-stderr
%pip install -U redis langgraph langchain_openai
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

### Define imports and helper functions

First, let's define some imports and shared utilities for both `RedisSaver` and `AsyncRedisSaver`


```python
"""Implementation of a langgraph checkpoint saver using Redis."""
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Tuple,
)

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

REDIS_KEY_SEPARATOR = ":"


# Utilities shared by both RedisSaver and AsyncRedisSaver


def _make_redis_checkpoint_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return REDIS_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_redis_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: Optional[int],
) -> str:
    if idx is None:
        return REDIS_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )

    return REDIS_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_redis_checkpoint_key(redis_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    if namespace != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_redis_checkpoint_writes_key(redis_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    if namespace != "writes":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _filter_keys(
    keys: List[str], before: Optional[RunnableConfig], limit: Optional[int]
) -> list:
    """Filter and sort Redis keys based on optional criteria."""
    if before:
        keys = [
            k
            for k in keys
            if _parse_redis_checkpoint_key(k.decode())["checkpoint_id"]
            < before["configurable"]["checkpoint_id"]
        ]

    keys = sorted(
        keys,
        key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        reverse=True,
    )
    if limit:
        keys = keys[:limit]
    return keys


def _dump_writes(serde: SerializerProtocol, writes: tuple[str, Any]) -> list[dict]:
    """Serialize pending writes."""
    serialized_writes = []
    for channel, value in writes:
        type_, serialized_value = serde.dumps_typed(value)
        serialized_writes.append(
            {"channel": channel, "type": type_, "value": serialized_value}
        )
    return serialized_writes


def _load_writes(
    serde: SerializerProtocol, task_id_to_data: dict[tuple[str, str], dict]
) -> list[PendingWrite]:
    """Deserialize pending writes."""
    writes = [
        (
            task_id,
            data[b"channel"].decode(),
            serde.loads_typed((data[b"type"].decode(), data[b"value"])),
        )
        for (task_id, _), data in task_id_to_data.items()
    ]
    return writes


def _parse_redis_checkpoint_data(
    serde: SerializerProtocol,
    key: str,
    data: dict,
    pending_writes: Optional[List[PendingWrite]] = None,
) -> Optional[CheckpointTuple]:
    """Parse checkpoint data retrieved from Redis."""
    if not data:
        return None

    parsed_key = _parse_redis_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = serde.loads_typed((data[b"type"].decode(), data[b"checkpoint"]))
    metadata = serde.loads(data[b"metadata"].decode())
    parent_checkpoint_id = data.get(b"parent_checkpoint_id", b"").decode()
    parent_config = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )
```

### RedisSaver

Below is an implementation of RedisSaver (for synchronous use of graph, i.e. `.invoke()`, `.stream()`). RedisSaver implements four methods that are required for any checkpointer:

- `.put` - Store a checkpoint with its configuration and metadata.
- `.put_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- `.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`).
- `.list` - List checkpoints that match a given configuration and filter criteria.


```python
class RedisSaver(BaseCheckpointSaver):
    """Redis-based checkpoint saver implementation."""

    conn: Redis

    def __init__(self, conn: Redis):
        super().__init__()
        self.conn = conn

    @classmethod
    @contextmanager
    def from_conn_info(cls, *, host: str, port: int, db: int) -> Iterator["RedisSaver"]:
        conn = None
        try:
            conn = Redis(host=host, port=port, db=db)
            yield RedisSaver(conn)
        finally:
            if conn:
                conn.close()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Redis.

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
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }
        self.conn.hset(key, mapping=data)
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
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> RunnableConfig:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, data in enumerate(_dump_writes(self.serde, writes)):
            key = _make_redis_checkpoint_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            self.conn.hset(key, mapping=data)
        return config

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        This method retrieves a checkpoint tuple from Redis based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = self._get_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None

        checkpoint_data = self.conn.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): self.conn.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        # TODO: implement filtering
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from Redis based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")

        keys = _filter_keys(self.conn.keys(pattern), before, limit)
        for key in keys:
            data = self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                yield _parse_redis_checkpoint_data(self.serde, key.decode(), data)

    def _get_checkpoint_key(
        self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = conn.keys(_make_redis_checkpoint_key(thread_id, checkpoint_ns, "*"))
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()
```

### AsyncRedis

Below is a reference implementation of AsyncRedisSaver (for asynchronous use of graph, i.e. `.ainvoke()`, `.astream()`). AsyncRedisSaver implements four methods that are required for any async checkpointer:

- `.aput` - Store a checkpoint with its configuration and metadata.
- `.aput_writes` - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- `.aget_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`).
- `.alist` - List checkpoints that match a given configuration and filter criteria.


```python
class AsyncRedisSaver(BaseCheckpointSaver):
    """Async redis-based checkpoint saver implementation."""

    conn: AsyncRedis

    def __init__(self, conn: AsyncRedis):
        super().__init__()
        self.conn = conn

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls, *, host: str, port: int, db: int
    ) -> AsyncIterator["AsyncRedisSaver"]:
        conn = None
        try:
            conn = AsyncRedis(host=host, port=port, db=db)
            yield AsyncRedisSaver(conn)
        finally:
            if conn:
                await conn.aclose()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to Redis. The checkpoint is associated
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
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "checkpoint_id": checkpoint_id,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        await self.conn.hset(key, mapping=data)
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
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> RunnableConfig:
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

        for idx, data in enumerate(_dump_writes(self.serde, writes)):
            key = _make_redis_checkpoint_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            )
            await self.conn.hset(key, mapping=data)
        return config

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis asynchronously.

        This method retrieves a checkpoint tuple from Redis based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = await self._aget_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None
        checkpoint_data = await self.conn.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): await self.conn.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        # TODO: implement filtering
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """List checkpoints from Redis asynchronously.

        This method retrieves a list of checkpoint tuples from Redis based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = _filter_keys(await self.conn.keys(pattern), before, limit)
        for key in keys:
            data = await self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                yield _parse_redis_checkpoint_data(self.serde, key.decode(), data)

    async def _aget_checkpoint_key(
        self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Asynchronously determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = await conn.keys(
            _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        )
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()
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
with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
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
     'ts': '2024-08-09T01:56:48.328315+00:00',
     'id': '1ef55f2a-3614-69b4-8003-2181cff935cc',
     'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'),
       AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
       ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I'),
       AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})],
      'agent': 'agent'},
     'channel_versions': {'__start__': '00000000000000000000000000000002.',
      'messages': '00000000000000000000000000000005.16e98d6f7ece7598829eddf1b33a33c4',
      'start:agent': '00000000000000000000000000000003.',
      'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
      'branch:agent:should_continue:tools': '00000000000000000000000000000004.',
      'tools': '00000000000000000000000000000005.'},
     'versions_seen': {'__input__': {},
      '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'},
      'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc',
       'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'},
      'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}},
     'pending_sends': [],
     'current_tasks': {}}




```python
latest_checkpoint_tuple
```




    CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3614-69b4-8003-2181cff935cc'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.328315+00:00', 'id': '1ef55f2a-3614-69b4-8003-2181cff935cc', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.16e98d6f7ece7598829eddf1b33a33c4', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-306f-6252-8002-47c2374ec1f2'}}, pending_writes=[])




```python
checkpoint_tuples
```




    [CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3614-69b4-8003-2181cff935cc'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.328315+00:00', 'id': '1ef55f2a-3614-69b4-8003-2181cff935cc', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.16e98d6f7ece7598829eddf1b33a33c4', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-306f-6252-8002-47c2374ec1f2'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-306f-6252-8002-47c2374ec1f2'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:47.736251+00:00', 'id': '1ef55f2a-306f-6252-8002-47c2374ec1f2', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I')], 'tools': 'tools'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000004.b16eb718f179ac1dcde54c5652768cf5', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000004.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I')]}}, 'step': 2}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-305f-61cc-8001-efac33022ef7'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-305f-61cc-8001-efac33022ef7'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:47.729689+00:00', 'id': '1ef55f2a-305f-61cc-8001-efac33022ef7', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})], 'agent': 'agent', 'branch:agent:should_continue:tools': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000003.4dd312547dcca1cf91a19adb620a18d6', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})]}}, 'step': 1}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-2a52-6a7c-8000-27624d954d15'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-2a52-6a7c-8000-27624d954d15'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:47.095456+00:00', 'id': '1ef55f2a-2a52-6a7c-8000-27624d954d15', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf')], 'start:agent': '__start__'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000002.52e8b0c387f50c28345585c088150464', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': None, 'step': 0}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-2a50-6812-bfff-34e3be35d6f2'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-2a50-6812-bfff-34e3be35d6f2'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:47.094575+00:00', 'id': '1ef55f2a-2a50-6812-bfff-34e3be35d6f2', 'channel_values': {'messages': [], '__start__': {'messages': [['human', "what's the weather in sf"]]}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'versions_seen': {'__input__': {}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'input', 'writes': {'messages': [['human', "what's the weather in sf"]]}, 'step': -1}, parent_config=None, pending_writes=None)]



## Use async connection


```python
async with AsyncRedisSaver.from_conn_info(
    host="localhost", port=6379, db=0
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
     'ts': '2024-08-09T01:56:49.503241+00:00',
     'id': '1ef55f2a-4149-61ea-8003-dc5506862287',
     'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'),
       AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}),
       ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9'),
       AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})],
      'agent': 'agent'},
     'channel_versions': {'__start__': '00000000000000000000000000000002.',
      'messages': '00000000000000000000000000000005.2cb29d082da6435a7528b4c917fd0c28',
      'start:agent': '00000000000000000000000000000003.',
      'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
      'branch:agent:should_continue:tools': '00000000000000000000000000000004.',
      'tools': '00000000000000000000000000000005.'},
     'versions_seen': {'__input__': {},
      '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'},
      'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc',
       'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'},
      'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}},
     'pending_sends': [],
     'current_tasks': {}}




```python
latest_checkpoint_tuple
```




    CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-4149-61ea-8003-dc5506862287'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.503241+00:00', 'id': '1ef55f2a-4149-61ea-8003-dc5506862287', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.2cb29d082da6435a7528b4c917fd0c28', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9'}}, pending_writes=[])




```python
checkpoint_tuples
```




    [CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-4149-61ea-8003-dc5506862287'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.503241+00:00', 'id': '1ef55f2a-4149-61ea-8003-dc5506862287', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.2cb29d082da6435a7528b4c917fd0c28', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.056860+00:00', 'id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9')], 'tools': 'tools'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000004.07964a3a545f9ff95545db45a9753d11', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000004.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9')]}}, 'step': 2}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3cf9-6996-8001-88dab066840d'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3cf9-6996-8001-88dab066840d'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.051234+00:00', 'id': '1ef55f2a-3cf9-6996-8001-88dab066840d', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})], 'agent': 'agent', 'branch:agent:should_continue:tools': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000003.cc96d93b1afbd1b69d53851320670b97', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})]}}, 'step': 1}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a6-6788-8000-9efe1769f8c1'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a6-6788-8000-9efe1769f8c1'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.388067+00:00', 'id': '1ef55f2a-36a6-6788-8000-9efe1769f8c1', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a')], 'start:agent': '__start__'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000002.a6994b785a651d88df51020401745af8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': None, 'step': 0}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a3-6614-bfff-05dafa02b4d7'}}, pending_writes=None),
     CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a3-6614-bfff-05dafa02b4d7'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.386807+00:00', 'id': '1ef55f2a-36a3-6614-bfff-05dafa02b4d7', 'channel_values': {'messages': [], '__start__': {'messages': [['human', "what's the weather in nyc"]]}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'versions_seen': {'__input__': {}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'input', 'writes': {'messages': [['human', "what's the weather in nyc"]]}, 'step': -1}, parent_config=None, pending_writes=None)]


