# How to use callbacks in async environments

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)
- [Custom callback handlers](/docs/how_to/custom_callbacks)
:::

If you are planning to use the async APIs, it is recommended to use and extend [`AsyncCallbackHandler`](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.AsyncCallbackHandler.html) to avoid blocking the event.


:::warning
If you use a sync `CallbackHandler` while using an async method to run your LLM / Chain / Tool / Agent, it will still work. However, under the hood, it will be called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) which can cause issues if your `CallbackHandler` is not thread-safe.
:::

:::danger

If you're on `python<=3.10`, you need to remember to propagate `config` or `callbacks` when invoking other `runnable` from within a `RunnableLambda`, `RunnableGenerator` or `@tool`. If you do not do this,
the callbacks will not be propagated to the child runnables being invoked.
:::


```python
# | output: false
# | echo: false

%pip install -qU langchain langchain_anthropic

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
```


```python
import asyncio
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in a list with our custom handler
chat = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    max_tokens=25,
    streaming=True,
    callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
)

await chat.agenerate([[HumanMessage(content="Tell me a joke")]])
```

    zzzz....
    Hi! I just woke up. Your llm is starting
    Sync handler being called in a `thread_pool_executor`: token: Here
    Sync handler being called in a `thread_pool_executor`: token: 's
    Sync handler being called in a `thread_pool_executor`: token:  a
    Sync handler being called in a `thread_pool_executor`: token:  little
    Sync handler being called in a `thread_pool_executor`: token:  joke
    Sync handler being called in a `thread_pool_executor`: token:  for
    Sync handler being called in a `thread_pool_executor`: token:  you
    Sync handler being called in a `thread_pool_executor`: token: :
    Sync handler being called in a `thread_pool_executor`: token: 
    
    Why
    Sync handler being called in a `thread_pool_executor`: token:  can
    Sync handler being called in a `thread_pool_executor`: token: 't
    Sync handler being called in a `thread_pool_executor`: token:  a
    Sync handler being called in a `thread_pool_executor`: token:  bicycle
    Sync handler being called in a `thread_pool_executor`: token:  stan
    Sync handler being called in a `thread_pool_executor`: token: d up
    Sync handler being called in a `thread_pool_executor`: token:  by
    Sync handler being called in a `thread_pool_executor`: token:  itself
    Sync handler being called in a `thread_pool_executor`: token: ?
    Sync handler being called in a `thread_pool_executor`: token:  Because
    Sync handler being called in a `thread_pool_executor`: token:  it
    Sync handler being called in a `thread_pool_executor`: token: 's
    Sync handler being called in a `thread_pool_executor`: token:  two
    Sync handler being called in a `thread_pool_executor`: token: -
    Sync handler being called in a `thread_pool_executor`: token: tire
    zzzz....
    Hi! I just woke up. Your llm is ending
    




    LLMResult(generations=[[ChatGeneration(text="Here's a little joke for you:\n\nWhy can't a bicycle stand up by itself? Because it's two-tire", message=AIMessage(content="Here's a little joke for you:\n\nWhy can't a bicycle stand up by itself? Because it's two-tire", id='run-8afc89e8-02c0-4522-8480-d96977240bd4-0'))]], llm_output={}, run=[RunInfo(run_id=UUID('8afc89e8-02c0-4522-8480-d96977240bd4'))])



## Next steps

You've now learned how to create your own custom callback handlers.

Next, check out the other how-to guides in this section, such as [how to attach callbacks to a runnable](/docs/how_to/callbacks_attach).
