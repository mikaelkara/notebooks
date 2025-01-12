---
sidebar_position: 4
---
# Build an Extraction Chain

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [Chat Models](/docs/concepts/chat_models)
- [Tools](/docs/concepts/tools)
- [Tool calling](/docs/concepts/tool_calling)

:::

In this tutorial, we will build a chain to extract structured information from unstructured text. 

:::important
This tutorial will only work with models that support **tool calling**
:::

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

## The Schema

First, we need to describe what information we want to extract from the text.

We'll use Pydantic to define an example schema  to extract personal information.


```python
from typing import Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )
```

There are two best practices when defining schema:

1. Document the **attributes** and the **schema** itself: This information is sent to the LLM and is used to improve the quality of information extraction.
2. Do not force the LLM to make up information! Above we used `Optional` for the attributes allowing the LLM to output `None` if it doesn't know the answer.

:::important
For best performance, document the schema well and make sure the model isn't force to return results if there's no information to be extracted in the text.
:::

## The Extractor

Let's create an information extractor using the schema we defined above.


```python
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)
```

We need to use a model that supports function/tool calling.

Please review [the documentation](/docs/concepts/tool_calling) for list of some models that can be used with this API.


```python
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest", temperature=0)

runnable = prompt | llm.with_structured_output(schema=Person)
```

    /Users/harrisonchase/workplace/langchain/libs/core/langchain_core/_api/beta_decorator.py:87: LangChainBetaWarning: The method `ChatMistralAI.with_structured_output` is in beta. It is actively being worked on, so the API may change.
      warn_beta(
    

Let's test it out


```python
text = "Alan Smith is 6 feet tall and has blond hair."
runnable.invoke({"text": text})
```




    Person(name='Alan Smith', hair_color='blond', height_in_meters='1.83')



:::important 

Extraction is Generative 🤯

LLMs are generative models, so they can do some pretty cool things like correctly extract the height of the person in meters
even though it was provided in feet!
:::

We can see the LangSmith trace here: https://smith.langchain.com/public/44b69a63-3b3b-47b8-8a6d-61b46533f015/r

## Multiple Entities

In **most cases**, you should be extracting a list of entities rather than a single entity.

This can be easily achieved using pydantic by nesting models inside one another.


```python
from typing import List, Optional

from pydantic import BaseModel, Field


class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]
```

:::important
Extraction might not be perfect here. Please continue to see how to use **Reference Examples** to improve the quality of extraction, and see the **guidelines** section!
:::


```python
runnable = prompt | llm.with_structured_output(schema=Data)
text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
runnable.invoke({"text": text})
```




    Data(people=[Person(name='Jeff', hair_color=None, height_in_meters=None), Person(name='Anna', hair_color=None, height_in_meters=None)])



:::tip
When the schema accommodates the extraction of **multiple entities**, it also allows the model to extract **no entities** if no relevant information
is in the text by providing an empty list. 

This is usually a **good** thing! It allows specifying **required** attributes on an entity without necessarily forcing the model to detect this entity.
:::

We can see the LangSmith trace here: https://smith.langchain.com/public/7173764d-5e76-45fe-8496-84460bd9cdef/r

## Next steps

Now that you understand the basics of extraction with LangChain, you're ready to proceed to the rest of the how-to guides:

- [Add Examples](/docs/how_to/extraction_examples): Learn how to use **reference examples** to improve performance.
- [Handle Long Text](/docs/how_to/extraction_long_text): What should you do if the text does not fit into the context window of the LLM?
- [Use a Parsing Approach](/docs/how_to/extraction_parse): Use a prompt based approach to extract with models that do not support **tool/function calling**.


```python

```
