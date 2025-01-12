# LlamaIndex RAG Workflows using Gemini and Firestore

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/orchestration/llamaindex_workflows.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Forchestration%2Fllamaindex_workflows.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/orchestration/llamaindex_workflows.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/orchestration/llamaindex_workflows.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Noa Ben-Efraim](https://github.com/noabenefraim) |

## Overview
LlamaIndex workflows are a powerful way to orchestrate complex LLM (large language model) applications. They provide an event-driven framework for building AI systems that go beyond simple question-answering.   

Think of a workflow as a series of steps, where each step performs a specific action. These actions can be anything from querying an LLM, to retrieving data from a vector database, to interacting with external APIs. The workflow manages the flow of data between these steps, making it easy to build sophisticated AI applications.   

Here's a breakdown of the key concepts:

+ Events: These trigger actions within the workflow. For example, a user's query can be an initial event that kicks off the workflow.   
+ Steps: These are individual functions decorated with @step that process events and potentially emit new events. Steps are the building blocks of your workflow.   
+ Event-driven: This means that the workflow reacts to events as they happen, making it flexible and dynamic.

This notebook perform a complex Retrieval Augmented Generation (RAG) workflow using Gemini models and Firestore databases. There are two branches for this workflow:

_Branch 1_
+ Start Event triggered by providing a data directory to the workflow
+ Ingest data using the LlamaIndex `SimpleDirectoryReader`
+ Load data in the Firestore Database

_Branch 2_
+ Start Event triggered by providing a query to the workflow
+ The QueryMultiStep Event that breaks down a complex query into sequential sub-questions using Gemini. Then proceeds to answer the sub-questions.
+ The sub-questions results are passed to the RerankEvent where given the initial user query, Gemini reranks the returned answers to the sub-questions.
+ The reranked chunks are passed to the CreateCitationEvents where citations are added to the sub-questions used to generate the answer.
+ An answer is synthesized for the original query and returned to the user.

References:
+ https://docs.llamaindex.ai/en/stable/examples/workflow/rag/
+ https://docs.llamaindex.ai/en/stable/examples/workflow/multi_step_query_engine/
+ https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/


![RAGWorkflow](https://storage.googleapis.com/github-repo/generative-ai/gemini/orchestration/llamaindex_workflows/RAGWorkflow.png)


## Get started

### Install required packages



```
%pip install llama-index=="0.11.8" \
    llama-index-embeddings-vertex=="0.2.0" \
    llama-index-utils-workflow=="0.2.1" \
    llama-index-llms-vertex=="0.3.4" \
    llama-index-storage-docstore-firestore=="0.2.0"
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK
This notebook requires the following resources:
+ Initialized Google Cloud project
+ Vertex AI API enabled
+ Existing VPC/Subnet
+ Existing Firestore database

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

To get started using Firestore Database, refer to the following [documentation](https://cloud.google.com/firestore/docs/manage-databases).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).



```
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}
if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
FIRESTORE_DATABASE_ID = "[your-firestore-database-id]"

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Workflow

### Import libraries


```
from typing import Any, cast

from IPython.display import Markdown, display
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.storage.docstore.firestore import FirestoreDocumentStore
from llama_index.utils.workflow import draw_all_possible_flows
from vertexai.generative_models import HarmBlockThreshold, HarmCategory, SafetySetting
```

### Get data


```
!mkdir -p './data'
!wget 'https://www.gutenberg.org/cache/epub/64317/pg64317.txt' -O 'data/gatsby.txt'
```

### Set credentials


```
import google.auth
import google.auth.transport.requests

# credentials will now have an api token
credentials = google.auth.default(quota_project_id=PROJECT_ID)[0]
request = google.auth.transport.requests.Request()
credentials.refresh(request)
```

## Workflow

### Set up the LLM


```
safety_config = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
]
embedding_model = VertexTextEmbedding(
    model_name="text-embedding-004", credentials=credentials
)
llm = Vertex(
    model="gemini-pro",
    temperature=0.2,
    max_tokens=3000,
    safety_settings=safety_config,
    credentials=credentials,
)

Settings.embed_model = embedding_model
Settings.llm = llm
```

### Define Event classes

Here we will create custom events that can be emitted by steps and trigger other steps. 



```
class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]
    source_nodes: list[NodeWithScore]
    final_response_metadata: dict[str, Any]


class FirestoreIndexData(Event):
    """Result of indexing documents in Firestore"""

    status: str


class QueryMultiStepEvent(Event):
    """
    Event containing results of a multi-step query process.

    Attributes:
        nodes (List[NodeWithScore]): List of nodes with their associated scores.
        source_nodes (List[NodeWithScore]): List of source nodes with their scores.
        final_response_metadata (Dict[str, Any]): Metadata associated with the final response.
    """

    nodes: list[NodeWithScore]
    source_nodes: list[NodeWithScore]
    final_response_metadata: dict[str, Any]


class CreateCitationsEvent(Event):
    """Add citations to the nodes."""

    nodes: list[NodeWithScore]
    source_nodes: list[NodeWithScore]
    final_response_metadata: dict[str, Any]
```

### Update Prompt Templates

Defining custom prompts used for the citation portion of the workflow.


```
CITATION_QA_TEMPLATE = PromptTemplate(
    "Your task is to answer the question based on the information given in the sources listed below."
    "Use only the provided sources to answer."
    "Cite the source number(s) for any information you use in your answer (e.g., [1])."
    "Always include at least one source citation in your answer."
    "Only cite a source if you directly use information from it."
    "If the sources don't contain the information needed to answer the question, state that."
    "For example:"
    "Source 1: Apples are red, green, or yellow."
    "Source 2:  Bananas are yellow when ripe."
    "Source 3: Strawberries are red when ripe."
    "Query: Which fruits are red when ripe?"
    "Answer: Apples [1] and strawberries [3] can be red when ripe."
    "------"
    "Below are several numbered sources of information:"
    "------"
    "{context_str}"
    "------"
    "Query: {query_str}"
    "Answer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "You have an initial answer to a query."
    "Your job is to improve this answer using the information provided in the numbered sources below. Here's how:"
    " - Read the existing answer and the sources carefully."
    " - Identify any information in the sources that can improve the answer by adding details, making it more accurate, or providing better support."
    " - If the sources provide new information, incorporate it into the answer."
    " - If the sources contradict the existing answer, correct the answer."
    " - If the sources aren't helpful, keep the original answer."
    "Cite the source number(s) for any information you use in your answer (e.g., [1])."
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "------"
    "{context_msg}"
    "------"
    "Query: {query_str}"
    "Answer: "
)

DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 20
```

### Workflow Class

The RAGWorkflow() class contains all the steps of the workflow. We define the steps by decorating the method with @step.



```
class RAGWorkflow(Workflow):
    @step
    async def ingest_data(
        self, ctx: Context, ev: StartEvent
    ) -> FirestoreIndexData | None:
        """Entry point to ingest a document, triggered by a StartEvent with 'dirname'."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        documents = SimpleDirectoryReader(dirname).load_data()
        await ctx.set("documents", documents)
        return FirestoreIndexData(
            status="First step complete. Data loaded into Documents."
        )

    @step
    async def load_database(self, ctx: Context, ev: FirestoreIndexData) -> StopEvent:
        print(ev.status)

        # create (or load) docstore and add nodes
        docstore = FirestoreDocumentStore.from_database(
            project=PROJECT_ID,
            database=FIRESTORE_DATABASE_ID,
        )

        docstore.add_documents(await ctx.get("documents"))

        # create storage context
        storage_context = StorageContext.from_defaults(docstore=docstore)

        # setup index
        index = VectorStoreIndex.from_documents(
            documents=await ctx.get("documents"), storage_context=storage_context
        )

        print("Index created")
        return StopEvent(index)

    def combine_queries(
        self,
        query_bundle: QueryBundle,
        prev_reasoning: str,
        llm: LLM,
    ) -> QueryBundle:
        """Combine queries using StepDecomposeQueryTransform."""
        transform_metadata = {"prev_reasoning": prev_reasoning}
        return StepDecomposeQueryTransform(llm=llm)(
            query_bundle, metadata=transform_metadata
        )

    def default_stop_fn(self, stop_dict: dict) -> bool:
        """Stop function for multi-step query combiner."""
        query_bundle = cast(QueryBundle, stop_dict.get("query_bundle"))
        if query_bundle is None:
            raise ValueError("Response must be provided to stop function.")

        return "none" in query_bundle.query_str.lower()

    @step(pass_context=True)
    async def query_multistep(
        self, ctx: Context, ev: StartEvent
    ) -> QueryMultiStepEvent | None:
        """Entry point for RAG, triggered by a StartEvent with `query`. Execute multi-step query process."""

        query = ev.get("query")
        index = ev.get("index")

        prev_reasoning = ""
        cur_response = None
        should_stop = False
        cur_steps = 0

        # use response
        final_response_metadata: dict[str, Any] = {"sub_qa": []}

        text_chunks = []
        source_nodes = []

        stop_fn = self.default_stop_fn

        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        # get the index from the global context
        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        num_steps = ev.get("num_steps")
        query_engine = index.as_query_engine()

        while not should_stop:
            if num_steps is not None and cur_steps >= num_steps:
                should_stop = True
                break
            elif should_stop:
                break

            updated_query_bundle = self.combine_queries(
                QueryBundle(query_str=query),
                prev_reasoning,
                llm=Settings.llm,
            )

            print(
                f"Created query for the step - {cur_steps} is: {updated_query_bundle}"
            )

            stop_dict = {"query_bundle": updated_query_bundle}
            if stop_fn(stop_dict):
                should_stop = True
                break

            cur_response = query_engine.query(updated_query_bundle)

            # append to response builder
            cur_qa_text = (
                f"\nQuestion: {updated_query_bundle.query_str}\n"
                f"Answer: {cur_response!s}"
            )
            text_chunks.append(cur_qa_text)
            print("Source nodes used:\n")
            for source_node in cur_response.source_nodes:
                print(source_node)
                source_nodes.append(source_node)

            # update metadata
            final_response_metadata["sub_qa"].append(
                (updated_query_bundle.query_str, cur_response)
            )

            prev_reasoning += (
                f"- {updated_query_bundle.query_str}\n" f"- {cur_response!s}\n"
            )
            cur_steps += 1

        nodes = [
            NodeWithScore(node=TextNode(text=text_chunk)) for text_chunk in text_chunks
        ]
        return QueryMultiStepEvent(
            nodes=nodes,
            source_nodes=source_nodes,
            final_response_metadata=final_response_metadata,
        )

    @step
    async def rerank(self, ctx: Context, ev: QueryMultiStepEvent) -> RerankEvent:
        # Rerank the nodes
        ranker = LLMRerank(choice_batch_size=5, top_n=10, llm=Settings.llm)
        print("Entering reranking of nodes:\n")
        print("Original query: ", await ctx.get("query", default=None), flush=True)
        # print(await ctx.get("query", default=None), flush=True)
        try:
            new_nodes = ranker.postprocess_nodes(
                ev.nodes, query_str=await ctx.get("query", default=None)
            )
        except:
            # re ranker is not guaranteed to create parsable output
            new_nodes = ev.nodes

        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(
            nodes=new_nodes,
            source_nodes=ev.source_nodes,
            final_response_metadata=ev.final_response_metadata,
        )

    @step
    async def create_citation_nodes(self, ev: RerankEvent) -> CreateCitationsEvent:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of NodeWithScore objects and splits their content
        into smaller chunks, creating new NodeWithScore objects for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.

        Args:
            nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.

        Returns:
            List[NodeWithScore]: A new list of NodeWithScore objects, where each object
            represents a smaller chunk of the original nodes, labeled as a source.
        """
        nodes = ev.nodes

        new_nodes: list[NodeWithScore] = []

        text_splitter = SentenceSplitter(
            chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
        )

        for node in nodes:
            print(node)

            text_chunks = text_splitter.split_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE)
            )

            for text_chunk in text_chunks:
                text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )

                new_node.node.text = text
                new_nodes.append(new_node)
        return CreateCitationsEvent(
            nodes=new_nodes,
            source_nodes=ev.source_nodes,
            final_response_metadata=ev.final_response_metadata,
        )

    @step
    async def synthesize(self, ctx: Context, ev: CreateCitationsEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""

        print("Synthesizing final result...")

        response_synthesizer = get_response_synthesizer(
            llm=Vertex(model="gemini-1.5-pro", temperature=0.0, max_tokens=5000),
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )
        query = await ctx.get("query", default=None)
        response = await response_synthesizer.asynthesize(
            query, nodes=ev.nodes, additional_source_nodes=ev.source_nodes
        )
        return StopEvent(result=response)
```


```
# optional - generate DAG for workflow created above
draw_all_possible_flows(workflow=RAGWorkflow, filename="multi_step_workflow.html")  # type: ignore
```

### Run the workflow


```
w = RAGWorkflow(timeout=200)
```


```
# Ingest the documents
index = await w.run(dirname="./data")
```

    First step complete. Data loaded into Documents.
    Index created
    

#### Example 1
Query: "What is the significance of the green light?"


```
# Run a query
NUM_STEPS = 2  # @param {type:"int"} represents how many sub-questions generated based on the query
result = await w.run(
    query="What is the significance of the green light?",
    index=index,
    num_steps=NUM_STEPS,
)

display(Markdown(f"{result}"))
```

    Query the database with: What is the significance of the green light?
    Created query for the step - 0 is: What is the significance of the green light?
    Source nodes used:
    
    Node ID: 0eab96dd-33ef-4d5c-a97e-8ca897af48d6
    Text: Its vanished trees, the trees that had made way for Gatsby’s
    house, had once pandered in whispers to the last and greatest of all
    human dreams; for a transitory enchanted moment man must have held his
    breath in the presence of this continent, compelled into an aesthetic
    contemplation he neither understood nor desired, face to face for the
    l...
    Score:  0.540
    
    Node ID: 4b08ce92-cbf0-4469-88a5-8cb3514da22f
    Text: “I’ve got a man in England who buys me clothes. He sends over a
    selection of things at the beginning of each season, spring and fall.”
    He took out a pile of shirts and began throwing them, one by one,
    before us, shirts of sheer linen and thick silk and fine flannel,
    which lost their folds as they fell and covered the table in  many-
    coloure...
    Score:  0.525
    
    Created query for the step - 1 is: ## New Question:
    
    **What is the significance of the green light in the context of Gatsby's pursuit of Daisy?** 
    
    Source nodes used:
    
    Node ID: f323395e-7546-454a-9f8b-563e73fbb292
    Text: “Old sport, the dance is unimportant.”    He wanted nothing less
    of Daisy than that she should go to Tom and  say: “I never loved you.”
    After she had obliterated four years with  that sentence they could
    decide upon the more practical measures to be  taken. One of them was
    that, after she was free, they were to go back  to Louisville and be
    marr...
    Score:  0.662
    
    Node ID: a2ec7e02-2983-4da9-b08a-afa1b6cc4216
    Text: “Why didn’t he ask you to arrange a meeting?”    “He wants her
    to see his house,” she explained. “And your house is  right next
    door.”    “Oh!”    “I think he half expected her to wander into one of
    his parties, some  night,” went on Jordan, “but she never did. Then he
    began asking  people casually if they knew her, and I was the first
    one he fo...
    Score:  0.648
    
    Entering reranking of nodes:
    
    Original query:  What is the significance of the green light?
    Reranked nodes to 2
    Node ID: c2860521-c9c1-4cab-b7a9-ea1c784506be
    Text: Question: What is the significance of the green light? Answer:
    The green light is a symbol of Gatsby's dream of Daisy. It is the
    light at the end of her dock, which he can see from his house across
    the bay. The green light represents Gatsby's hope for a future with
    Daisy, and his belief that he can recapture the past. However, the
    green light is...
    Score: None
    
    Node ID: 7fe78bba-c870-486e-8f29-0168b09a792e
    Text: Question: ## New Question:  **What is the significance of the
    green light in the context of Gatsby's pursuit of Daisy?**   Answer:
    ## The Green Light: A Symbol of Gatsby's Dreams and Desires  The green
    light at the end of Daisy's dock plays a pivotal role in symbolizing
    Gatsby's aspirations and the unattainable nature of his dreams. It
    represent...
    Score: None
    
    Synthesizing final result...
    


## The Significance of the Green Light in The Great Gatsby

The green light at the end of Daisy's dock holds immense symbolic weight in F. Scott Fitzgerald's *The Great Gatsby*. It represents a multitude of Gatsby's aspirations and desires, while simultaneously highlighting the unattainable nature of his dreams.

**Unrequited Love:** The green light's physical proximity to Gatsby, yet separation by the bay, mirrors the emotional distance between him and Daisy. He yearns for her, but she remains out of reach, symbolizing his unrequited love.

**The Past:** The green light evokes memories of Gatsby's past with Daisy, a time when their love seemed possible. He desperately wants to recapture that lost time and recreate their romance, clinging to the hope of a second chance.

**Hope and Illusion:** The green light embodies Gatsby's unwavering hope for a future with Daisy. He believes that if he can achieve enough wealth and success, he can win her back. However, this hope is ultimately an illusion, as Daisy has moved on and their circumstances have changed.

**The American Dream:** The green light can be interpreted as a symbol of the American Dream, representing Gatsby's relentless pursuit of wealth and social status. He believes that achieving these goals will bring him happiness and allow him to win Daisy's love. However, the novel ultimately suggests that the American Dream is often unattainable and can lead to disillusionment.

**Additional Points:**

* The green light's color reinforces its symbolic meaning. Green often represents hope, growth, and new beginnings, but in this context, it takes on a more melancholic and unattainable quality.
* The light's flickering nature reflects the instability of Gatsby's dreams and the uncertainty of his future.
* Gatsby's constant focus on the green light highlights his single-minded obsession with Daisy and his inability to move on from the past.

**Overall, the green light serves as a powerful symbol that encapsulates Gatsby's longing, his yearning for a lost love, and the ultimately unattainable nature of his dreams.**

**Sources:**

* [1] The Great Gatsby by F. Scott Fitzgerald
* [2] SparkNotes: The Great Gatsby - Symbols, Imagery, Allegory


Check the ranked LLM generated sub-question answers used:


```
for idx in range(0, NUM_STEPS):
    print(result.source_nodes[idx])
```

    Node ID: c2860521-c9c1-4cab-b7a9-ea1c784506be
    Text: Source 1: Question: What is the significance of the green light?
    Answer: The green light is a symbol of Gatsby's dream of Daisy. It is
    the light at the end of her dock, which he can see from his house
    across the bay. The green light represents Gatsby's hope for a future
    with Daisy, and his belief that he can recapture the past. However,
    the gree...
    Score: None
    
    Node ID: 7fe78bba-c870-486e-8f29-0168b09a792e
    Text: Source 2: Question: ## New Question:  **What is the significance
    of the green light in the context of Gatsby's pursuit of Daisy?**
    Answer: ## The Green Light: A Symbol of Gatsby's Dreams and Desires
    The green light at the end of Daisy's dock plays a pivotal role in
    symbolizing Gatsby's aspirations and the unattainable nature of his
    dreams. It...
    Score: None
    
    

Check the citations from the original source used:


```
for idx in range(NUM_STEPS, len(result.source_nodes)):
    print(result.source_nodes[idx])
```

    Node ID: 0eab96dd-33ef-4d5c-a97e-8ca897af48d6
    Text: Its vanished trees, the trees that had made way for Gatsby’s
    house, had once pandered in whispers to the last and greatest of all
    human dreams; for a transitory enchanted moment man must have held his
    breath in the presence of this continent, compelled into an aesthetic
    contemplation he neither understood nor desired, face to face for the
    l...
    Score:  0.540
    
    Node ID: 4b08ce92-cbf0-4469-88a5-8cb3514da22f
    Text: “I’ve got a man in England who buys me clothes. He sends over a
    selection of things at the beginning of each season, spring and fall.”
    He took out a pile of shirts and began throwing them, one by one,
    before us, shirts of sheer linen and thick silk and fine flannel,
    which lost their folds as they fell and covered the table in  many-
    coloure...
    Score:  0.525
    
    Node ID: f323395e-7546-454a-9f8b-563e73fbb292
    Text: “Old sport, the dance is unimportant.”    He wanted nothing less
    of Daisy than that she should go to Tom and  say: “I never loved you.”
    After she had obliterated four years with  that sentence they could
    decide upon the more practical measures to be  taken. One of them was
    that, after she was free, they were to go back  to Louisville and be
    marr...
    Score:  0.662
    
    Node ID: a2ec7e02-2983-4da9-b08a-afa1b6cc4216
    Text: “Why didn’t he ask you to arrange a meeting?”    “He wants her
    to see his house,” she explained. “And your house is  right next
    door.”    “Oh!”    “I think he half expected her to wander into one of
    his parties, some  night,” went on Jordan, “but she never did. Then he
    began asking  people casually if they knew her, and I was the first
    one he fo...
    Score:  0.648
    
    

## Cleaning up

To clean up all Google Cloud resources used in this project, you can delete the Google Cloud project you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial.
