# Extraction with OpenAI Tools

Performing extraction has never been easier! OpenAI's tool calling ability is the perfect thing to use as it allows for extracting multiple different elements from text that are different types. 

Models after 1106 use tools and support "parallel function calling" which makes this super easy.


```python
from typing import List, Optional

from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
```


```python
# Make sure to use a recent model that supports tools
model = ChatOpenAI(model="gpt-3.5-turbo-1106")
```


```python
# Pydantic is an easy way to define a schema
class Person(BaseModel):
    """Information about people to extract."""

    name: str
    age: Optional[int] = None
```


```python
chain = create_extraction_chain_pydantic(Person, model)
```


```python
chain.invoke({"input": "jane is 2 and bob is 3"})
```




    [Person(name='jane', age=2), Person(name='bob', age=3)]




```python
# Let's define another element
class Class(BaseModel):
    """Information about classes to extract."""

    teacher: str
    students: List[str]
```


```python
chain = create_extraction_chain_pydantic([Person, Class], model)
```


```python
chain.invoke({"input": "jane is 2 and bob is 3 and they are in Mrs Sampson's class"})
```




    [Person(name='jane', age=2),
     Person(name='bob', age=3),
     Class(teacher='Mrs Sampson', students=['jane', 'bob'])]



## Under the hood

Under the hood, this is a simple chain:

```python
from typing import Union, List, Type, Optional

from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_tool
from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseLanguageModel

_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned \
in the following passage together with their properties.

If a property is not present and is not required in the function parameters, do not include it in the output."""  # noqa: E501


def create_extraction_chain_pydantic(
    pydantic_schemas: Union[List[Type[BaseModel]], Type[BaseModel]],
    llm: BaseLanguageModel,
    system_message: str = _EXTRACTION_TEMPLATE,
) -> Runnable:
    if not isinstance(pydantic_schemas, list):
        pydantic_schemas = [pydantic_schemas]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{input}")
    ])
    tools = [convert_pydantic_to_openai_tool(p) for p in pydantic_schemas]
    model = llm.bind(tools=tools)
    chain = prompt | model | PydanticToolsParser(tools=pydantic_schemas)
    return chain
```


```python

```
