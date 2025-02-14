# AirbyteLoader

>[Airbyte](https://github.com/airbytehq/airbyte) is a data integration platform for ELT pipelines from APIs, databases & files to warehouses & lakes. It has the largest catalog of ELT connectors to data warehouses and databases.

This covers how to load any source from Airbyte into LangChain documents

## Installation

In order to use `AirbyteLoader` you need to install the `langchain-airbyte` integration package.


```python
%pip install -qU langchain-airbyte
```

Note: Currently, the `airbyte` library does not support Pydantic v2.
Please downgrade to Pydantic v1 to use this package.

Note: This package also currently requires Python 3.10+.

## Loading Documents

By default, the `AirbyteLoader` will load any structured data from a stream and output yaml-formatted documents.


```python
from langchain_airbyte import AirbyteLoader

loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 10},
)
docs = loader.load()
print(docs[0].page_content[:500])
```

    ```yaml
    academic_degree: PhD
    address:
      city: Lauderdale Lakes
      country_code: FI
      postal_code: '75466'
      province: New Jersey
      state: Hawaii
      street_name: Stoneyford
      street_number: '1112'
    age: 44
    blood_type: "O\u2212"
    created_at: '2004-04-02T13:05:27+00:00'
    email: bread2099+1@outlook.com
    gender: Fluid
    height: '1.62'
    id: 1
    language: Belarusian
    name: Moses
    nationality: Dutch
    occupation: Track Worker
    telephone: 1-467-194-2318
    title: M.Sc.Tech.
    updated_at: '2024-02-27T16:41:01+00:00'
    weight: 6
    

You can also specify a custom prompt template for formatting documents:


```python
from langchain_core.prompts import PromptTemplate

loader_templated = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 10},
    template=PromptTemplate.from_template(
        "My name is {name} and I am {height} meters tall."
    ),
)
docs_templated = loader_templated.load()
print(docs_templated[0].page_content)
```

    My name is Verdie and I am 1.73 meters tall.
    

## Lazy Loading Documents

One of the powerful features of `AirbyteLoader` is its ability to load large documents from upstream sources. When working with large datasets, the default `.load()` behavior can be slow and memory-intensive. To avoid this, you can use the `.lazy_load()` method to load documents in a more memory-efficient manner.


```python
import time

loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 3},
    template=PromptTemplate.from_template(
        "My name is {name} and I am {height} meters tall."
    ),
)

start_time = time.time()
my_iterator = loader.lazy_load()
print(
    f"Just calling lazy load is quick! This took {time.time() - start_time:.4f} seconds"
)
```

    Just calling lazy load is quick! This took 0.0001 seconds
    

And you can iterate over documents as they're yielded:


```python
for doc in my_iterator:
    print(doc.page_content)
```

    My name is Andera and I am 1.91 meters tall.
    My name is Jody and I am 1.85 meters tall.
    My name is Zonia and I am 1.53 meters tall.
    

You can also lazy load documents in an async manner with `.alazy_load()`:


```python
loader = AirbyteLoader(
    source="source-faker",
    stream="users",
    config={"count": 3},
    template=PromptTemplate.from_template(
        "My name is {name} and I am {height} meters tall."
    ),
)

my_async_iterator = loader.alazy_load()

async for doc in my_async_iterator:
    print(doc.page_content)
```

    My name is Carmelina and I am 1.74 meters tall.
    My name is Ali and I am 1.90 meters tall.
    My name is Rochell and I am 1.83 meters tall.
    

## Configuration

`AirbyteLoader` can be configured with the following options:

- `source` (str, required): The name of the Airbyte source to load from.
- `stream` (str, required): The name of the stream to load from (Airbyte sources can return multiple streams)
- `config` (dict, required): The configuration for the Airbyte source
- `template` (PromptTemplate, optional): A custom prompt template for formatting documents
- `include_metadata` (bool, optional, default True): Whether to include all fields as metadata in the output documents

The majority of the configuration will be in `config`, and you can find the specific configuration options in the "Config field reference" for each source in the [Airbyte documentation](https://docs.airbyte.com/integrations/).


