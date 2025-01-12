---
sidebar_position: 0
---
# Build a Simple LLM Application with LCEL

In this quickstart we'll show you how to build a simple LLM application with LangChain. This application will translate text from English into another language. This is a relatively simple LLM application - it's just a single LLM call plus some prompting. Still, this is a great way to get started with LangChain - a lot of features can be built with just some prompting and an LLM call!

After reading this tutorial, you'll have a high level overview of:

- Using [language models](/docs/concepts/chat_models)

- Using [PromptTemplates](/docs/concepts/prompt_templates) and [OutputParsers](/docs/concepts/output_parsers)

- Using [LangChain Expression Language (LCEL)](/docs/concepts/lcel) to chain components together

- Debugging and tracing your application using [LangSmith](https://docs.smith.langchain.com/)

- Deploying your application with [LangServe](/docs/concepts/architecture/#langserve)

Let's dive in!

## Setup

### Jupyter Notebook

This guide (and most of the other guides in the documentation) uses [Jupyter notebooks](https://jupyter.org/) and assumes the reader is as well. Jupyter notebooks are perfect for learning how to work with LLM systems because oftentimes things can go wrong (unexpected output, API down, etc) and going through guides in an interactive environment is a great way to better understand them.

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import CodeBlock from "@theme/CodeBlock";

<Tabs>
  <TabItem value="pip" label="Pip" default>
    <CodeBlock language="bash">pip install langchain</CodeBlock>
  </TabItem>
  <TabItem value="conda" label="Conda">
    <CodeBlock language="bash">conda install langchain -c conda-forge</CodeBlock>
  </TabItem>
</Tabs>



For more details, see our [Installation guide](/docs/how_to/installation).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```python
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## Using Language Models

First up, let's learn how to use a language model by itself. LangChain supports many different language models that you can use interchangeably. For details on getting started with a specific model, refer to [supported integrations](/docs/integrations/chat/).

import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs openaiParams={`model="gpt-4"`} />



```python
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
```

Let's first use the model directly. `ChatModel`s are instances of LangChain "Runnables", which means they expose a standard interface for interacting with them. To just simply call the model, we can pass in a list of messages to the `.invoke` method.


```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

model.invoke(messages)
```




    AIMessage(content='ciao!', response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-fc5d7c88-9615-48ab-a3c7-425232b562c5-0')



If we've enabled LangSmith, we can see that this run is logged to LangSmith, and can see the [LangSmith trace](https://smith.langchain.com/public/88baa0b2-7c1a-4d09-ba30-a47985dde2ea/r)

## OutputParsers

Notice that the response from the model is an `AIMessage`. This contains a string response along with other metadata about the response. Oftentimes we may just want to work with the string response. We can parse out just this response by using a simple output parser.

We first import the simple output parser.


```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
```

One way to use it is to use it by itself. For example, we could save the result of the language model call and then pass it to the parser.


```python
result = model.invoke(messages)
```


```python
parser.invoke(result)
```




    'Ciao!'



More commonly, we can "chain" the model with this output parser. This means this output parser will get called every time in this chain. This chain takes on the input type of the language model (string or list of message) and returns the output type of the output parser (string).

We can easily create the chain using the `|` operator. The `|` operator is used in LangChain to combine two elements together.


```python
chain = model | parser
```


```python
chain.invoke(messages)
```




    'Ciao!'



If we now look at LangSmith, we can see that the chain has two steps: first the language model is called, then the result of that is passed to the output parser. We can see the [LangSmith trace]( https://smith.langchain.com/public/f1bdf656-2739-42f7-ac7f-0f1dd712322f/r)

## Prompt Templates

Right now we are passing a list of messages directly into the language model. Where does this list of messages come from? Usually, it is constructed from a combination of user input and application logic. This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language model. Common transformations include adding a system message or formatting a template with the user input.

PromptTemplates are a concept in LangChain designed to assist with this transformation. They take in raw user input and return data (a prompt) that is ready to pass into a language model. 

Let's create a PromptTemplate here. It will take in two user variables:

- `language`: The language to translate text into
- `text`: The text to translate


```python
from langchain_core.prompts import ChatPromptTemplate
```

First, let's create a string that we will format to be the system message:


```python
system_template = "Translate the following into {language}:"
```

Next, we can create the PromptTemplate. This will be a combination of the `system_template` as well as a simpler template for where to put the text to be translated


```python
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
```

The input to this prompt template is a dictionary. We can play around with this prompt template by itself to see what it does by itself


```python
result = prompt_template.invoke({"language": "italian", "text": "hi"})

result
```




    ChatPromptValue(messages=[SystemMessage(content='Translate the following into italian:'), HumanMessage(content='hi')])



We can see that it returns a `ChatPromptValue` that consists of two messages. If we want to access the messages directly we do:


```python
result.to_messages()
```




    [SystemMessage(content='Translate the following into italian:'),
     HumanMessage(content='hi')]



## Chaining together components with LCEL

We can now combine this with the model and the output parser from above using the pipe (`|`) operator:


```python
chain = prompt_template | model | parser
```


```python
chain.invoke({"language": "italian", "text": "hi"})
```




    'ciao'



This is a simple example of using [LangChain Expression Language (LCEL)](/docs/concepts/lcel) to chain together LangChain modules. There are several benefits to this approach, including optimized streaming and tracing support.

If we take a look at the LangSmith trace, we can see all three components show up in the [LangSmith trace](https://smith.langchain.com/public/bc49bec0-6b13-4726-967f-dbd3448b786d/r).

## Serving with LangServe

Now that we've built an application, we need to serve it. That's where LangServe comes in.
LangServe helps developers deploy LangChain chains as a REST API. You do not need to use LangServe to use LangChain, but in this guide we'll show how you can deploy your app with LangServe.

While the first part of this guide was intended to be run in a Jupyter Notebook or script, we will now move out of that. We will be creating a Python file and then interacting with it from the command line.

Install with:
```bash
pip install "langserve[all]"
```

### Server

To create a server for our application we'll make a `serve.py` file. This will contain our logic for serving our application. It consists of three things:
1. The definition of our chain that we just built above
2. Our FastAPI app
3. A definition of a route from which to serve the chain, which is done with `langserve.add_routes`


```python
#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 6. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

And that's it! If we execute this file:
```bash
python serve.py
```
we should see our chain being served at [http://localhost:8000](http://localhost:8000).

### Playground

Every LangServe service comes with a simple [built-in UI](https://github.com/langchain-ai/langserve/blob/main/README.md#playground) for configuring and invoking the application with streaming output and visibility into intermediate steps.
Head to [http://localhost:8000/chain/playground/](http://localhost:8000/chain/playground/) to try it out! Pass in the same inputs as before - `{"language": "italian", "text": "hi"}` - and it should respond same as before.

### Client

Now let's set up a client for programmatically interacting with our service. We can easily do this with the [langserve.RemoteRunnable](/docs/langserve/#client).
Using this, we can interact with the served chain as if it were running client-side.

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
```

To learn more about the many other features of LangServe [head here](/docs/langserve).

## Conclusion

That's it! In this tutorial you've learned how to create your first simple LLM application. You've learned how to work with language models, how to parse their outputs, how to create a prompt template, chaining them with LCEL, how to get great observability into chains you create with LangSmith, and how to deploy them with LangServe.

This just scratches the surface of what you will want to learn to become a proficient AI Engineer. Luckily - we've got a lot of other resources!

For further reading on the core concepts of LangChain, we've got detailed [Conceptual Guides](/docs/concepts).

If you have more specific questions on these concepts, check out the following sections of the how-to guides:

- [LangChain Expression Language (LCEL)](/docs/how_to/#langchain-expression-language-lcel)
- [Prompt templates](/docs/how_to/#prompt-templates)
- [Chat models](/docs/how_to/#chat-models)
- [Output parsers](/docs/how_to/#output-parsers)
- [LangServe](/docs/langserve/)

And the LangSmith docs:

- [LangSmith](https://docs.smith.langchain.com)


```python

```
