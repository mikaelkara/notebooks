<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/3p_integrations/llamaindex/dlai_agentic_rag/Building_Agentic_RAG_with_Llamaindex_L4_Building_a_Multi-Document_Agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook ports the DeepLearning.AI short course [Building Agentic RAG with Llamaindex Lesson 4 Building a Multi-Document Agent](https://learn.deeplearning.ai/courses/building-agentic-rag-with-llamaindex/lesson/5/building-a-multi-document-agent) to using Llama 3. It shows how to use an agent to handle multiple documents and increasing degrees of complexity.

You should take the course before or after going through this notebook to have a deeper understanding.

Note: Unlike Lessons 1 and 3 where we use Llama 3 70b on [Groq](https://groq.com/), this lesson uses Llama 3 on [Fireworks.ai](https://fireworks.ai/) to overcome the rate limit issue with Groq on some summary tool calling.


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


```python
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]
```


```python
for url, paper in zip(urls, papers):
  !wget "{url}" -O "{paper}"
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
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```


```python
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```


```python
from llama_index.core.agent import ReActAgent

query_engine_tools = [vector_tool, summary_tool]

agent = ReActAgent.from_tools(
    initial_tools,
    llm=llm,
    verbose=True,
)
```


```python
response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)
```


```python
response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))
```


```python
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D"
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "values.pdf",
    "finetune_fair_diffusion.pdf",
    "knowledge_card.pdf",
    "metra.pdf",
    "vr_mcl.pdf"
]
```


```python
for url, paper in zip(urls, papers):
  !wget "{url}" -O "{paper}"
```


```python
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```


```python
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```


```python
# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
```


```python
obj_retriever = obj_index.as_retriever(similarity_top_k=3)
```


```python
tools = obj_retriever.retrieve(
    "Tell me about the eval dataset used in MetaGPT and SWE-Bench"
)
```


```python
tools[2].metadata
```


```python
# The LlamaIndex's FunctionCallingAgentWorker API doesn't work correctly with Fireworks Llama, so we use ReActAgent here.
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tool_retriever=obj_retriever,
    llm=llm,
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
""",    
    verbose=True,
)
```


```python
response = agent.query(
    "Tell me about the evaluation dataset used "
    "in MetaGPT and compare it against SWE-Bench"
)
print(str(response))
```


```python
response = agent.query(
    "Compare and contrast the LoRA papers (LongLoRA, LoftQ). "
    "Analyze the approach in each paper first. "
)
```
