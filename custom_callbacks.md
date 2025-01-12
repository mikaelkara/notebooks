# How to create custom callback handlers

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Callbacks](/docs/concepts/callbacks)

:::

LangChain has some built-in callback handlers, but you will often want to create your own handlers with custom logic.

To create a custom callback handler, we need to determine the [event(s)](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler) we want our callback handler to handle as well as what we want our callback handler to do when the event is triggered. Then all we need to do is attach the callback handler to the object, for example via [the constructor](/docs/how_to/callbacks_constructor) or [at runtime](/docs/how_to/callbacks_runtime).

In the example below, we'll implement streaming with a custom handler.

In our custom callback handler `MyCustomHandler`, we implement the `on_llm_new_token` handler to print the token we have just received. We then attach our custom handler to the model object as a constructor callback.


```python
# | output: false
# | echo: false

%pip install -qU langchain langchain_anthropic

import getpass
import os

os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()
```


```python
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


prompt = ChatPromptTemplate.from_messages(["Tell me a joke about {animal}"])

# To enable streaming, we pass in `streaming=True` to the ChatModel constructor
# Additionally, we pass in our custom handler as a list to the callbacks parameter
model = ChatAnthropic(
    model="claude-3-sonnet-20240229", streaming=True, callbacks=[MyCustomHandler()]
)

chain = prompt | model

response = chain.invoke({"animal": "bears"})
```

    My custom handler, token: Here
    My custom handler, token: 's
    My custom handler, token:  a
    My custom handler, token:  bear
    My custom handler, token:  joke
    My custom handler, token:  for
    My custom handler, token:  you
    My custom handler, token: :
    My custom handler, token: 
    
    Why
    My custom handler, token:  di
    My custom handler, token: d the
    My custom handler, token:  bear
    My custom handler, token:  dissol
    My custom handler, token: ve
    My custom handler, token:  in
    My custom handler, token:  water
    My custom handler, token: ?
    My custom handler, token: 
    Because
    My custom handler, token:  it
    My custom handler, token:  was
    My custom handler, token:  a
    My custom handler, token:  polar
    My custom handler, token:  bear
    My custom handler, token: !
    

You can see [this reference page](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html#langchain-core-callbacks-base-basecallbackhandler) for a list of events you can handle. Note that the `handle_chain_*` events run for most LCEL runnables.

## Next steps

You've now learned how to create your own custom callback handlers.

Next, check out the other how-to guides in this section, such as [how to attach callbacks to a runnable](/docs/how_to/callbacks_attach).
