# Golden Query

>[Golden](https://golden.com) provides a set of natural language APIs for querying and enrichment using the Golden Knowledge Graph e.g. queries such as: `Products from OpenAI`, `Generative ai companies with series a funding`, and `rappers who invest` can be used to retrieve structured data about relevant entities.
>
>The `golden-query` langchain tool is a wrapper on top of the [Golden Query API](https://docs.golden.com/reference/query-api) which enables programmatic access to these results.
>See the [Golden Query API docs](https://docs.golden.com/reference/query-api) for more information.


This notebook goes over how to use the `golden-query` tool.

- Go to the [Golden API docs](https://docs.golden.com/) to get an overview about the Golden API.
- Get your API key from the [Golden API Settings](https://golden.com/settings/api) page.
- Save your API key into GOLDEN_API_KEY env variable


```python
%pip install -qU langchain-community
```


```python
import os

os.environ["GOLDEN_API_KEY"] = ""
```


```python
from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper
```


```python
golden_query = GoldenQueryAPIWrapper()
```


```python
import json

json.loads(golden_query.run("companies in nanotech"))
```




    {'results': [{'id': 4673886,
       'latestVersionId': 60276991,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Samsung', 'citations': []}]}]},
      {'id': 7008,
       'latestVersionId': 61087416,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Intel', 'citations': []}]}]},
      {'id': 24193,
       'latestVersionId': 60274482,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Texas Instruments', 'citations': []}]}]},
      {'id': 1142,
       'latestVersionId': 61406205,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Advanced Micro Devices', 'citations': []}]}]},
      {'id': 193948,
       'latestVersionId': 58326582,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Freescale Semiconductor', 'citations': []}]}]},
      {'id': 91316,
       'latestVersionId': 60387380,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Agilent Technologies', 'citations': []}]}]},
      {'id': 90014,
       'latestVersionId': 60388078,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Novartis', 'citations': []}]}]},
      {'id': 237458,
       'latestVersionId': 61406160,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'Analog Devices', 'citations': []}]}]},
      {'id': 3941943,
       'latestVersionId': 60382250,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'AbbVie Inc.', 'citations': []}]}]},
      {'id': 4178762,
       'latestVersionId': 60542667,
       'properties': [{'predicateId': 'name',
         'instances': [{'value': 'IBM', 'citations': []}]}]}],
     'next': 'https://golden.com/api/v2/public/queries/59044/results/?cursor=eyJwb3NpdGlvbiI6IFsxNzYxNiwgIklCTS04M1lQM1oiXX0%3D&pageSize=10',
     'previous': None}


