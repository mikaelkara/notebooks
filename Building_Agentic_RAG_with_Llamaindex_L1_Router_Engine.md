<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/quickstart/agents/DeepLearningai_Course_Notebooks/Building_Agentic_RAG_with_Llamaindex_L1_Router_Engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [Building Agentic RAG with Llamaindex Lesson 1 Router Engine](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/2/router-query-engine) to using Llama 3. 

You should take the course before or after going through this notebook to have a deeper understanding.


```python
!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install llama-index-llms-groq
```


```python
import os 
os.environ['GROQ_API_KEY'] = 'your_groq_api_key' # get a free key at https://console.groq.com/keys
```


```python
!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
```


```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
```


```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
```


```python
from llama_index.llms.groq import Groq

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = Groq(model="llama3-8b-8192") #, api_key=GROQ_API_TOKEN)
Settings.llm = llm
#llm.complete("Who wrote the book godfather").text

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```


```python
from llama_index.core import SummaryIndex, VectorStoreIndex

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)
```


```python
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
```


```python
from llama_index.core.tools import QueryEngineTool

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)
```


```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)
```


```python
import nest_asyncio

nest_asyncio.apply()
```


```python
response = query_engine.query("What is the summary of the document?")
print(str(response))
```


```python
print(len(response.source_nodes))
```


```python
response = query_engine.query(
    "How do agents share information with other agents? This is not a summarization question."
)
print(str(response))
```


```python
def get_router_query_engine(file_path: str):
    """Get router query engine."""

    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to MetaGPT"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the MetaGPT paper."
        ),
    )

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    return query_engine

query_engine = get_router_query_engine("metagpt.pdf")
```


```python
response = query_engine.query("Tell me about the ablation study results?")
print(str(response))
```


```python

```
