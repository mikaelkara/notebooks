<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/3p_integrations/llamaindex/dlai_agentic_rag/Building_Agentic_RAG_with_Llamaindex_L2_Tool_Calling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [Building Agentic RAG with Llamaindex Lesson 2 Tool Calling](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/3/tool-calling) to using Llama 3. It shows how to use Llama 3 to not only pick a function to execute, but also infer an argument to pass through the function.

You should take the course before or after going through this notebook to have a deeper understanding.

Note: Unlike Lesson 1 where we use Llama 3 70b on [Groq](https://groq.com/), this lesson uses Llama 3 on [Fireworks.ai](https://fireworks.ai/) to overcome the rate limit issue with Groq on some summary tool calling.


```python
!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install llama-index-llms-fireworks
```


```python
import nest_asyncio

nest_asyncio.apply()
```


```python
from llama_index.core.tools import FunctionTool

def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int:
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)


add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)
```


```python
import os 

os.environ['FIREWORKS_API_KEY'] = 'xxx' # get a free key at https://fireworks.ai/api-keys
```


```python
from llama_index.llms.fireworks import Fireworks

# Llama 3 8b on Fireworks.ai also works in some cases, but 70b works better overall
#llm = Fireworks(model="accounts/fireworks/models/llama-v3-8b-instruct", temperature=0)
llm = Fireworks(model="accounts/fireworks/models/llama-v3-70b-instruct", temperature=0)

# a quick sanity test
#llm.complete("Who wrote the  book godfather? ").text

response = llm.predict_and_call(
    [add_tool, mystery_tool],
    "Tell me the output of the mystery function on 2 and 9",
    verbose=True
)
print(str(response))
```


```python
!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
```


```python
from llama_index.core import SimpleDirectoryReader

# https://arxiv.org/pdf/2308.00352 metagpt.pdf
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
```


```python
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
```


```python
print(nodes[0].get_content(metadata_mode="all"))
```


```python
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```


```python
# Settings.llm and embed_model apply to which call below? VectorStoreIndex(), as_query_engine?

from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)
```


```python
from llama_index.core.vector_stores import MetadataFilters

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query(
    "What are some high-level results of MetaGPT?",
)
```


```python
print(str(response))
```


```python
for n in response.source_nodes:
    print(n.metadata)
```


```python
from typing import List
from llama_index.core.vector_stores import FilterCondition


def vector_query(
    query: str,
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.

    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.

    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]

    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response


vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)
```


```python
response = llm.predict_and_call(
    [vector_query_tool],
    "What are the high-level results of MetaGPT as described on page 2?",
    verbose=True
)
```


```python
for n in response.source_nodes:
    print(n.metadata)
```


```python
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool

summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get a summary of MetaGPT"
    ),
)
```


```python
response = llm.predict_and_call(
    [vector_query_tool, summary_tool],
    "What are the MetaGPT comparisons with ChatDev described on page 8?",
    verbose=True
)
```


```python
for n in response.source_nodes:
    print(n.metadata)
```


```python
response = llm.predict_and_call(
    [vector_query_tool, summary_tool],
    "What is a summary of the paper?",
    verbose=True
)
```


```python

```
