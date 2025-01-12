<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/3p_integrations/llamaindex/dlai_agentic_rag/Building_Agentic_RAG_with_Llamaindex_L3_Building_an_Agent_Reasoning_Loop.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [Building Agentic RAG with Llamaindex Lesson 3 Building an Agent Reasoning Loop](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/4/building-an-agent-reasoning-loop) to using Llama 3. It shows how to define a complete agent reasoning loop to reason over tools and multiple steps on a complex question the user asks about a single document while maintaining memory.

You should take the course before or after going through this notebook to have a deeper understanding.


```python
!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install llama-index-llms-groq
```


```python
import nest_asyncio

nest_asyncio.apply()
```

## Load the data


```python
!wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
```

## Setup the Query Tools


```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional

def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)

    def vector_query(
        query: str,
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over the MetaGPT paper.

        Useful if you have specific questions over the MetaGPT paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.

        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.

        """

        page_numbers = page_numbers or []
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
        name=f"vector_tool_{name}",
        fn=vector_query
    )

    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of MetaGPT. "
            "Do NOT use if you have specific questions over MetaGPT."
        ),
    )

    return vector_query_tool, summary_tool
```

## Setup Llama and Agent

Note: The LlamaIndex's FunctionCallingAgentWorker API doesn't work correctly with Groq Llama, so we use ReActAgent here.


```python
import os 

os.environ['GROQ_API_KEY'] = 'xxx' # get a free key at https://console.groq.com/keys
```


```python
from llama_index.llms.groq import Groq

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

llm = Groq(model="llama3-70b-8192", temperature=0)
Settings.llm = llm

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```


```python
vector_tool, summary_tool = get_doc_tools("metagpt.pdf", "metagpt")
```


```python
from llama_index.core.agent import ReActAgent

query_engine_tools = [vector_tool, summary_tool]

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
)
```


```python
response = agent.query(
    "Tell me about the agent roles in MetaGPT, and how they communicate with each other."
)
print(str(response))
```


```python
print(response.source_nodes[0].get_content(metadata_mode="all"))
```


```python
response = agent.query(
    "Tell me about the evaluation datasets used."
)
```


```python
print(response.source_nodes[0].get_content(metadata_mode="all"))
```


```python
# use agent.chat instead of agent.query to pass conversational history automatically to answer follow up questions
response = agent.chat("Tell me the results over one of the above datasets.")
```


```python
# use agent.chat instead of agent.query to pass conversational history automatically to answer follow up questions
response = agent.chat("Tell me more.")
```

## Lower-Level: Debuggability and Control

Note: The LlamaIndex's FunctionCallingAgentWorker API doesn't work correctly with Groq Llama, so we use ReActAgent here.


```python
agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
)
```


```python
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)
```


```python
step_output = agent.run_step(task.task_id)
```


```python
completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)
```


```python
upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]
```


```python
step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)
```


```python
step_output = agent.run_step(task.task_id)
print(step_output.is_last)
```


```python
response = agent.finalize_response(task.task_id)
```


```python
print(str(response))
```
