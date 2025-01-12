# How to create tools

When constructing an agent, you will need to provide it with a list of `Tool`s that it can use. Besides the actual function that is called, the Tool consists of several components:

| Attribute     | Type                            | Description                                                                                                                                                                    |
|---------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name          | str                             | Must be unique within a set of tools provided to an LLM or agent.                                                                                                              |
| description   | str                             | Describes what the tool does. Used as context by the LLM or agent.                                                                                                             |
| args_schema   | pydantic.BaseModel | Optional but recommended, and required if using callback handlers. It can be used to provide more information (e.g., few-shot examples) or validation for expected parameters. |
| return_direct | boolean                         | Only relevant for agents. When True, after invoking the given tool, the agent will stop and return the result direcly to the user.                                             |

LangChain supports the creation of tools from:

1. Functions;
2. LangChain [Runnables](/docs/concepts/runnables);
3. By sub-classing from [BaseTool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.BaseTool.html) -- This is the most flexible method, it provides the largest degree of control, at the expense of more effort and code.

Creating tools from functions may be sufficient for most use cases, and can be done via a simple [@tool decorator](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.tool.html#langchain_core.tools.tool). If more configuration is needed-- e.g., specification of both sync and async implementations-- one can also use the [StructuredTool.from_function](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.structured.StructuredTool.html#langchain_core.tools.structured.StructuredTool.from_function) class method.

In this guide we provide an overview of these methods.

:::tip

Models will perform better if the tools have well chosen names, descriptions and JSON schemas.
:::

## Creating tools from functions

### @tool decorator

This `@tool` decorator is the simplest way to define a custom tool. The decorator uses the function name as the tool name by default, but this can be overridden by passing a string as the first argument. Additionally, the decorator will use the function's docstring as the tool's description - so a docstring MUST be provided. 


```python
from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
```

    multiply
    Multiply two numbers.
    {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
    

Or create an **async** implementation, like this:


```python
from langchain_core.tools import tool


@tool
async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

Note that `@tool` supports parsing of annotations, nested schemas, and other features:


```python
from typing import Annotated, List


@tool
def multiply_by_max(
    a: Annotated[str, "scale factor"],
    b: Annotated[List[int], "list of ints over which to take maximum"],
) -> int:
    """Multiply a by the maximum of b."""
    return a * max(b)


multiply_by_max.args_schema.schema()
```




    {'description': 'Multiply a by the maximum of b.',
     'properties': {'a': {'description': 'scale factor',
       'title': 'A',
       'type': 'string'},
      'b': {'description': 'list of ints over which to take maximum',
       'items': {'type': 'integer'},
       'title': 'B',
       'type': 'array'}},
     'required': ['a', 'b'],
     'title': 'multiply_by_maxSchema',
     'type': 'object'}



You can also customize the tool name and JSON args by passing them into the tool decorator.


```python
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)
```

    multiplication-tool
    Multiply two numbers.
    {'a': {'description': 'first number', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'second number', 'title': 'B', 'type': 'integer'}}
    True
    

#### Docstring parsing

`@tool` can optionally parse [Google Style docstrings](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) and associate the docstring components (such as arg descriptions) to the relevant parts of the tool schema. To toggle this behavior, specify `parse_docstring`:


```python
@tool(parse_docstring=True)
def foo(bar: str, baz: int) -> str:
    """The foo.

    Args:
        bar: The bar.
        baz: The baz.
    """
    return bar


foo.args_schema.schema()
```




    {'description': 'The foo.',
     'properties': {'bar': {'description': 'The bar.',
       'title': 'Bar',
       'type': 'string'},
      'baz': {'description': 'The baz.', 'title': 'Baz', 'type': 'integer'}},
     'required': ['bar', 'baz'],
     'title': 'fooSchema',
     'type': 'object'}



:::caution
By default, `@tool(parse_docstring=True)` will raise `ValueError` if the docstring does not parse correctly. See [API Reference](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.tool.html) for detail and examples.
:::

### StructuredTool

The `StructuredTool.from_function` class method provides a bit more configurability than the `@tool` decorator, without requiring much additional code.


```python
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(await calculator.ainvoke({"a": 2, "b": 5}))
```

    6
    10
    

To configure it:


```python
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

print(calculator.invoke({"a": 2, "b": 3}))
print(calculator.name)
print(calculator.description)
print(calculator.args)
```

    6
    Calculator
    multiply numbers
    {'a': {'description': 'first number', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'second number', 'title': 'B', 'type': 'integer'}}
    

## Creating tools from Runnables

LangChain [Runnables](/docs/concepts/runnables) that accept string or `dict` input can be converted to tools using the [as_tool](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.as_tool) method, which allows for the specification of names, descriptions, and additional schema information for arguments.

Example usage:


```python
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [("human", "Hello. Please respond in the style of {answer_style}.")]
)

# Placeholder LLM
llm = GenericFakeChatModel(messages=iter(["hello matey"]))

chain = prompt | llm | StrOutputParser()

as_tool = chain.as_tool(
    name="Style responder", description="Description of when to use tool."
)
as_tool.args
```

    /var/folders/4j/2rz3865x6qg07tx43146py8h0000gn/T/ipykernel_95770/2548361071.py:14: LangChainBetaWarning: This API is in beta and may change in the future.
      as_tool = chain.as_tool(
    




    {'answer_style': {'title': 'Answer Style', 'type': 'string'}}



See [this guide](/docs/how_to/convert_runnable_to_tool) for more detail.

## Subclass BaseTool

You can define a custom tool by sub-classing from `BaseTool`. This provides maximal control over the tool definition, but requires writing more code.


```python
from typing import Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class CustomCalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        return self._run(a, b, run_manager=run_manager.get_sync())
```


```python
multiply = CustomCalculatorTool()
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)

print(multiply.invoke({"a": 2, "b": 3}))
print(await multiply.ainvoke({"a": 2, "b": 3}))
```

    Calculator
    useful for when you need to answer questions about math
    {'a': {'description': 'first number', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'second number', 'title': 'B', 'type': 'integer'}}
    True
    6
    6
    

## How to create async tools

LangChain Tools implement the [Runnable interface 🏃](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html).

All Runnables expose the `invoke` and `ainvoke` methods (as well as other methods like `batch`, `abatch`, `astream` etc).

So even if you only provide an `sync` implementation of a tool, you could still use the `ainvoke` interface, but there
are some important things to know:

* LangChain's by default provides an async implementation that assumes that the function is expensive to compute, so it'll delegate execution to another thread.
* If you're working in an async codebase, you should create async tools rather than sync tools, to avoid incuring a small overhead due to that thread.
* If you need both sync and async implementations, use `StructuredTool.from_function` or sub-class from `BaseTool`.
* If implementing both sync and async, and the sync code is fast to run, override the default LangChain async implementation and simply call the sync code.
* You CANNOT and SHOULD NOT use the sync `invoke` with an `async` tool.


```python
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(
    await calculator.ainvoke({"a": 2, "b": 5})
)  # Uses default LangChain async implementation incurs small overhead
```

    6
    10
    


```python
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(
    await calculator.ainvoke({"a": 2, "b": 5})
)  # Uses use provided amultiply without additional overhead
```

    6
    10
    

You should not and cannot use `.invoke` when providing only an async definition.


```python
@tool
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


try:
    multiply.invoke({"a": 2, "b": 3})
except NotImplementedError:
    print("Raised not implemented error. You should not be doing this.")
```

    Raised not implemented error. You should not be doing this.
    

## Handling Tool Errors 

If you're using tools with agents, you will likely need an error handling strategy, so the agent can recover from the error and continue execution.

A simple strategy is to throw a `ToolException` from inside the tool and specify an error handler using `handle_tool_error`. 

When the error handler is specified, the exception will be caught and the error handler will decide which output to return from the tool.

You can set `handle_tool_error` to `True`, a string value, or a function. If it's a function, the function should take a `ToolException` as a parameter and return a value.

Please note that only raising a `ToolException` won't be effective. You need to first set the `handle_tool_error` of the tool because its default value is `False`.


```python
from langchain_core.tools import ToolException


def get_weather(city: str) -> int:
    """Get weather for the given city."""
    raise ToolException(f"Error: There is no city by the name of {city}.")
```

Here's an example with the default `handle_tool_error=True` behavior.


```python
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True,
)

get_weather_tool.invoke({"city": "foobar"})
```




    'Error: There is no city by the name of foobar.'



We can set `handle_tool_error` to a string that will always be returned.


```python
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error="There is no such city, but it's probably above 0K there!",
)

get_weather_tool.invoke({"city": "foobar"})
```




    "There is no such city, but it's probably above 0K there!"



Handling the error using a function:


```python
def _handle_error(error: ToolException) -> str:
    return f"The following errors occurred during tool execution: `{error.args[0]}`"


get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=_handle_error,
)

get_weather_tool.invoke({"city": "foobar"})
```




    'The following errors occurred during tool execution: `Error: There is no city by the name of foobar.`'



## Returning artifacts of Tool execution

Sometimes there are artifacts of a tool's execution that we want to make accessible to downstream components in our chain or agent, but that we don't want to expose to the model itself. For example if a tool returns custom objects like Documents, we may want to pass some view or metadata about this output to the model without passing the raw output to the model. At the same time, we may want to be able to access this full output elsewhere, for example in downstream tools.

The Tool and [ToolMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolMessage.html) interfaces make it possible to distinguish between the parts of the tool output meant for the model (this is the ToolMessage.content) and those parts which are meant for use outside the model (ToolMessage.artifact).

:::info Requires ``langchain-core >= 0.2.19``

This functionality was added in ``langchain-core == 0.2.19``. Please make sure your package is up to date.

:::

If we want our tool to distinguish between message content and other artifacts, we need to specify `response_format="content_and_artifact"` when defining our tool and make sure that we return a tuple of (content, artifact):


```python
import random
from typing import List, Tuple

from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def generate_random_ints(min: int, max: int, size: int) -> Tuple[str, List[int]]:
    """Generate size random ints in the range [min, max]."""
    array = [random.randint(min, max) for _ in range(size)]
    content = f"Successfully generated array of {size} random ints in [{min}, {max}]."
    return content, array
```

If we invoke our tool directly with the tool arguments, we'll get back just the content part of the output:


```python
generate_random_ints.invoke({"min": 0, "max": 9, "size": 10})
```




    'Successfully generated array of 10 random ints in [0, 9].'



If we invoke our tool with a ToolCall (like the ones generated by tool-calling models), we'll get back a ToolMessage that contains both the content and artifact generated by the Tool:


```python
generate_random_ints.invoke(
    {
        "name": "generate_random_ints",
        "args": {"min": 0, "max": 9, "size": 10},
        "id": "123",  # required
        "type": "tool_call",  # required
    }
)
```




    ToolMessage(content='Successfully generated array of 10 random ints in [0, 9].', name='generate_random_ints', tool_call_id='123', artifact=[4, 8, 2, 4, 1, 0, 9, 5, 8, 1])



We can do the same when subclassing BaseTool:


```python
from langchain_core.tools import BaseTool


class GenerateRandomFloats(BaseTool):
    name: str = "generate_random_floats"
    description: str = "Generate size random floats in the range [min, max]."
    response_format: str = "content_and_artifact"

    ndigits: int = 2

    def _run(self, min: float, max: float, size: int) -> Tuple[str, List[float]]:
        range_ = max - min
        array = [
            round(min + (range_ * random.random()), ndigits=self.ndigits)
            for _ in range(size)
        ]
        content = f"Generated {size} floats in [{min}, {max}], rounded to {self.ndigits} decimals."
        return content, array

    # Optionally define an equivalent async method

    # async def _arun(self, min: float, max: float, size: int) -> Tuple[str, List[float]]:
    #     ...
```


```python
rand_gen = GenerateRandomFloats(ndigits=4)

rand_gen.invoke(
    {
        "name": "generate_random_floats",
        "args": {"min": 0.1, "max": 3.3333, "size": 3},
        "id": "123",
        "type": "tool_call",
    }
)
```




    ToolMessage(content='Generated 3 floats in [0.1, 3.3333], rounded to 4 decimals.', name='generate_random_floats', tool_call_id='123', artifact=[1.5566, 0.5134, 2.7914])


