# How to propagate callbacks  constructor

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)
- [Custom callback handlers](/docs/how_to/custom_callbacks)

:::

Most LangChain modules allow you to pass `callbacks` directly into the constructor (i.e., initializer). In this case, the callbacks will only be called for that instance (and any nested runs).

:::warning
Constructor callbacks are scoped only to the object they are defined on. They are **not** inherited by children of the object. This can lead to confusing behavior,
and it's generally better to pass callbacks as a run time argument.
:::

Here's an example:


```python
# | output: false
# | echo: false

%pip install -qU langchain langchain_anthropic

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
```


```python
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate


class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")


callbacks = [LoggingHandler()]
llm = ChatAnthropic(model="claude-3-sonnet-20240229", callbacks=callbacks)
prompt = ChatPromptTemplate.from_template("What is 1 + {number}?")

chain = prompt | llm

chain.invoke({"number": "2"})
```

    Chat model started
    Chat model ended, response: generations=[[ChatGeneration(text='1 + 2 = 3', message=AIMessage(content='1 + 2 = 3', response_metadata={'id': 'msg_01CdKsRmeS9WRb8BWnHDEHm7', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 13}}, id='run-2d7fdf2a-7405-4e17-97c0-67e6b2a65305-0'))]] llm_output={'id': 'msg_01CdKsRmeS9WRb8BWnHDEHm7', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 13}} run=None
    




    AIMessage(content='1 + 2 = 3', response_metadata={'id': 'msg_01CdKsRmeS9WRb8BWnHDEHm7', 'model': 'claude-3-sonnet-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 13}}, id='run-2d7fdf2a-7405-4e17-97c0-67e6b2a65305-0')



You can see that we only see events from the chat model run - no chain events from the prompt or broader chain.

## Next steps

You've now learned how to pass callbacks into a constructor.

Next, check out the other how-to guides in this section, such as how to [pass callbacks at runtime](/docs/how_to/callbacks_runtime).
