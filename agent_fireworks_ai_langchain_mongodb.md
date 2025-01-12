[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/agents/agent_fireworks_ai_langchain_mongodb.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/agent-fireworksai-mongodb-langchain/)

## Install Libraries


```python
!pip install langchain langchain_openai langchain-fireworks langchain-mongodb arxiv pymupdf datasets pymongo
```

## Set Evironment Variables


```python
import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREWORKS_API_KEY"] = ""
os.environ["MONGO_URI"] = ""

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
```

## Data Ingestion into MongoDB Vector Database



```python
import pandas as pd
from datasets import load_dataset

data = load_dataset("MongoDB/subset_arxiv_papers_with_emebeddings")
dataset_df = pd.DataFrame(data["train"])
```

    /Users/richmondalake/miniconda3/envs/langchain_workarea/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    Downloading readme: 100%|██████████| 701/701 [00:00<00:00, 2.04MB/s]
    Repo card metadata block was not found. Setting CardData to empty.
    Downloading data: 100%|██████████| 102M/102M [00:15<00:00, 6.41MB/s] 
    Generating train split: 50000 examples [00:01, 38699.64 examples/s]
    


```python
print(len(dataset_df))
dataset_df.head()
```

    50000
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>submitter</th>
      <th>authors</th>
      <th>title</th>
      <th>comments</th>
      <th>journal-ref</th>
      <th>doi</th>
      <th>report-no</th>
      <th>categories</th>
      <th>license</th>
      <th>abstract</th>
      <th>versions</th>
      <th>update_date</th>
      <th>authors_parsed</th>
      <th>embedding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>704.0001</td>
      <td>Pavel Nadolsky</td>
      <td>C. Bal\'azs, E. L. Berger, P. M. Nadolsky, C.-...</td>
      <td>Calculation of prompt diphoton production cros...</td>
      <td>37 pages, 15 figures; published version</td>
      <td>Phys.Rev.D76:013009,2007</td>
      <td>10.1103/PhysRevD.76.013009</td>
      <td>ANL-HEP-PR-07-12</td>
      <td>hep-ph</td>
      <td>None</td>
      <td>A fully differential calculation in perturba...</td>
      <td>[{'version': 'v1', 'created': 'Mon, 2 Apr 2007...</td>
      <td>2008-11-26</td>
      <td>[[Balázs, C., ], [Berger, E. L., ], [Nadolsky,...</td>
      <td>[0.0594153292, -0.0440569334, -0.0487333685, -...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>704.0002</td>
      <td>Louis Theran</td>
      <td>Ileana Streinu and Louis Theran</td>
      <td>Sparsity-certifying Graph Decompositions</td>
      <td>To appear in Graphs and Combinatorics</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>math.CO cs.CG</td>
      <td>http://arxiv.org/licenses/nonexclusive-distrib...</td>
      <td>We describe a new algorithm, the $(k,\ell)$-...</td>
      <td>[{'version': 'v1', 'created': 'Sat, 31 Mar 200...</td>
      <td>2008-12-13</td>
      <td>[[Streinu, Ileana, ], [Theran, Louis, ]]</td>
      <td>[0.0247399714, -0.065658465, 0.0201423876, -0....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>704.0003</td>
      <td>Hongjun Pan</td>
      <td>Hongjun Pan</td>
      <td>The evolution of the Earth-Moon system based o...</td>
      <td>23 pages, 3 figures</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>physics.gen-ph</td>
      <td>None</td>
      <td>The evolution of Earth-Moon system is descri...</td>
      <td>[{'version': 'v1', 'created': 'Sun, 1 Apr 2007...</td>
      <td>2008-01-13</td>
      <td>[[Pan, Hongjun, ]]</td>
      <td>[0.0491479263, 0.0728017688, 0.0604138002, 0.0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>704.0004</td>
      <td>David Callan</td>
      <td>David Callan</td>
      <td>A determinant of Stirling cycle numbers counts...</td>
      <td>11 pages</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>math.CO</td>
      <td>None</td>
      <td>We show that a determinant of Stirling cycle...</td>
      <td>[{'version': 'v1', 'created': 'Sat, 31 Mar 200...</td>
      <td>2007-05-23</td>
      <td>[[Callan, David, ]]</td>
      <td>[0.0389556214, -0.0410280302, 0.0410280302, -0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>704.0005</td>
      <td>Alberto Torchinsky</td>
      <td>Wael Abu-Shammala and Alberto Torchinsky</td>
      <td>From dyadic $\Lambda_{\alpha}$ to $\Lambda_{\a...</td>
      <td>None</td>
      <td>Illinois J. Math. 52 (2008) no.2, 681-689</td>
      <td>None</td>
      <td>None</td>
      <td>math.CA math.FA</td>
      <td>None</td>
      <td>In this paper we show how to compute the $\L...</td>
      <td>[{'version': 'v1', 'created': 'Mon, 2 Apr 2007...</td>
      <td>2013-10-15</td>
      <td>[[Abu-Shammala, Wael, ], [Torchinsky, Alberto, ]]</td>
      <td>[0.118412666, -0.0127423415, 0.1185125113, 0.0...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pymongo import MongoClient

# Initialize MongoDB python client
client = MongoClient(MONGO_URI, appname="devrel.content.ai_agent_firechain.python")

DB_NAME = "agent_demo"
COLLECTION_NAME = "knowledge"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]
```


```python
# Delete any existing records in the collection
collection.delete_many({})

# Data Ingestion
records = dataset_df.to_dict("records")
collection.insert_many(records)

print("Data ingestion into MongoDB completed")
```

    Data ingestion into MongoDB completed
    

## Create Vector Search Index Defintion

```
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 256,
      "similarity": "cosine"
    }
  ]
}
```

## Create LangChain Retriever (MongoDB)


```python
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)

# Vector Store Creation
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGO_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="abstract",
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```

### Optional: Creating a retrevier with compression capabilities using LLMLingua



```python
!pip install langchain_community llmlingua
```


```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import LLMLinguaCompressor
```


```python
compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
```

    /Users/richmondalake/miniconda3/envs/langchain_workarea/lib/python3.12/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    

## Configure LLM Using Fireworks AI


```python
from langchain_fireworks import ChatFireworks

llm = ChatFireworks(model="accounts/fireworks/models/firefunction-v1", max_tokens=256)
```

## Agent Tools Creation


```python
from langchain.agents import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import ArxivLoader


# Custom Tool Definiton
@tool
def get_metadata_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a maximum of ten documents from arXiv matching the given query word.

    Args:
      word (str): The search query to find relevant documents on arXiv.

    Returns:
      list: Metadata about the documents matching the query.
    """
    docs = ArxivLoader(query=word, load_max_docs=10).load()
    # Extract just the metadata from each document
    metadata_list = [doc.metadata for doc in docs]
    return metadata_list


@tool
def get_information_from_arxiv(word: str) -> list:
    """
    Fetches and returns metadata for a single research paper from arXiv matching the given query word, which is the ID of the paper, for example: 704.0001.

    Args:
      word (str): The search query to find the relevant paper on arXiv using the ID.

    Returns:
      list: Data about the paper matching the query.
    """
    doc = ArxivLoader(query=word, load_max_docs=1).load()
    return doc


# If you created a retriever with compression capaitilies in the optional cell in an earlier cell, you can replace 'retriever' with 'compression_retriever'
# Otherwise you can also create a compression procedure as a tool for the agent as shown in the `compress_prompt_using_llmlingua` tool definition function
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="knowledge_base",
    description="This serves as the base knowledge source of the agent and contains some records of research papers from Arxiv. This tool is used as the first step for exploration and reseach efforts.",
)
```


```python
from langchain_community.document_compressors import LLMLinguaCompressor

compressor = LLMLinguaCompressor(model_name="openai-community/gpt2", device_map="cpu")


@tool
def compress_prompt_using_llmlingua(prompt: str, compression_rate: float = 0.5) -> str:
    """
    Compresses a long data or prompt using the LLMLinguaCompressor.

    Args:
        data (str): The data or prompt to be compressed.
        compression_rate (float): The rate at which to compress the data (default is 0.5).

    Returns:
        str: The compressed data or prompt.
    """
    compressed_data = compressor.compress_prompt(
        prompt,
        rate=compression_rate,
        force_tokens=["!", ".", "?", "\n"],
        drop_consecutive=True,
    )
    return compressed_data
```


```python
tools = [
    retriever_tool,
    get_metadata_information_from_arxiv,
    get_information_from_arxiv,
    compress_prompt_using_llmlingua,
]
```

## Agent Prompt Creation


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

agent_purpose = """
You are a helpful research assistant equipped with various tools to assist with your tasks efficiently. 
You have access to conversational history stored in your inpout as chat_history.
You are cost-effective and utilize the compress_prompt_using_llmlingua tool whenever you determine that a prompt or conversational history is too long. 
Below are instructions on when and how to use each tool in your operations.

1. get_metadata_information_from_arxiv

Purpose: To fetch and return metadata for up to ten documents from arXiv that match a given query word.
When to Use: Use this tool when you need to gather metadata about multiple research papers related to a specific topic.
Example: If you are asked to provide an overview of recent papers on "machine learning," use this tool to fetch metadata for relevant documents.

2. get_information_from_arxiv

Purpose: To fetch and return metadata for a single research paper from arXiv using the paper's ID.
When to Use: Use this tool when you need detailed information about a specific research paper identified by its arXiv ID.
Example: If you are asked to retrieve detailed information about the paper with the ID "704.0001," use this tool.

3. retriever_tool

Purpose: To serve as your base knowledge, containing records of research papers from arXiv.
When to Use: Use this tool as the first step for exploration and research efforts when dealing with topics covered by the documents in the knowledge base.
Example: When beginning research on a new topic that is well-documented in the arXiv repository, use this tool to access the relevant papers.

4. compress_prompt_using_llmlingua

Purpose: To compress long prompts or conversational histories using the LLMLinguaCompressor.
When to Use: Use this tool whenever you determine that a prompt or conversational history is too long to be efficiently processed.
Example: If you receive a very lengthy query or conversation context that exceeds the typical token limits, compress it using this tool before proceeding with further processing.

"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_purpose),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
```

## Agent Memory Using MongoDB


```python
from langchain.memory import ConversationBufferMemory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory


def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGO_URI, session_id, database_name=DB_NAME, collection_name="history"
    )


memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=get_session_history("latest_agent_session")
)
```

## Agent Creation


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
)
```

## Agent Exectution


```python
agent_executor.invoke(
    {
        "input": "Get me a list of research papers on the topic Prompt Compression in LLM Applications."
    }
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `get_metadata_information_from_arxiv` with `{'word': 'Prompt Compression in LLM Applications'}`
    
    
    [0m[33;1m[1;3m[{'Published': '2024-05-27', 'Title': 'SelfCP: Compressing Long Prompt to 1/12 Using the Frozen Large Language Model Itself', 'Authors': 'Jun Gao', 'Summary': 'Long prompt leads to huge hardware costs when using Large Language Models\n(LLMs). Unfortunately, many tasks, such as summarization, inevitably introduce\nlong task-inputs, and the wide application of in-context learning easily makes\nthe prompt length explode. Inspired by the language understanding ability of\nLLMs, this paper proposes SelfCP, which uses the LLM \\textbf{itself} to\n\\textbf{C}ompress long \\textbf{P}rompt into compact virtual tokens. SelfCP\napplies a general frozen LLM twice, first as an encoder to compress the prompt\nand then as a decoder to generate responses. Specifically, given a long prompt,\nwe place special tokens within the lengthy segment for compression and signal\nthe LLM to generate $k$ virtual tokens. Afterward, the virtual tokens\nconcatenate with the uncompressed prompt and are fed into the same LLM to\ngenerate the response. In general, SelfCP facilitates the unconditional and\nconditional compression of prompts, fitting both standard tasks and those with\nspecific objectives. Since the encoder and decoder are frozen, SelfCP only\ncontains 17M trainable parameters and allows for convenient adaptation across\nvarious backbones. We implement SelfCP with two LLM backbones and evaluate it\nin both in- and out-domain tasks. Results show that the compressed virtual\ntokens can substitute $12 \\times$ larger original prompts effectively'}, {'Published': '2024-04-18', 'Title': 'Adapting LLMs for Efficient Context Processing through Soft Prompt Compression', 'Authors': 'Cangqing Wang, Yutian Yang, Ruisi Li, Dan Sun, Ruicong Cai, Yuzhu Zhang, Chengqian Fu, Lillian Floyd', 'Summary': "The rapid advancement of Large Language Models (LLMs) has inaugurated a\ntransformative epoch in natural language processing, fostering unprecedented\nproficiency in text generation, comprehension, and contextual scrutiny.\nNevertheless, effectively handling extensive contexts, crucial for myriad\napplications, poses a formidable obstacle owing to the intrinsic constraints of\nthe models' context window sizes and the computational burdens entailed by\ntheir operations. This investigation presents an innovative framework that\nstrategically tailors LLMs for streamlined context processing by harnessing the\nsynergies among natural language summarization, soft prompt compression, and\naugmented utility preservation mechanisms. Our methodology, dubbed\nSoftPromptComp, amalgamates natural language prompts extracted from\nsummarization methodologies with dynamically generated soft prompts to forge a\nconcise yet semantically robust depiction of protracted contexts. This\ndepiction undergoes further refinement via a weighting mechanism optimizing\ninformation retention and utility for subsequent tasks. We substantiate that\nour framework markedly diminishes computational overhead and enhances LLMs'\nefficacy across various benchmarks, while upholding or even augmenting the\ncaliber of the produced content. By amalgamating soft prompt compression with\nsophisticated summarization, SoftPromptComp confronts the dual challenges of\nmanaging lengthy contexts and ensuring model scalability. Our findings point\ntowards a propitious trajectory for augmenting LLMs' applicability and\nefficiency, rendering them more versatile and pragmatic for real-world\napplications. This research enriches the ongoing discourse on optimizing\nlanguage models, providing insights into the potency of soft prompts and\nsummarization techniques as pivotal instruments for the forthcoming generation\nof NLP solutions."}, {'Published': '2023-12-06', 'Title': 'LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models', 'Authors': 'Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu', 'Summary': 'Large language models (LLMs) have been applied in various applications due to\ntheir astonishing capabilities. With advancements in technologies such as\nchain-of-thought (CoT) prompting and in-context learning (ICL), the prompts fed\nto LLMs are becoming increasingly lengthy, even exceeding tens of thousands of\ntokens. To accelerate model inference and reduce cost, this paper presents\nLLMLingua, a coarse-to-fine prompt compression method that involves a budget\ncontroller to maintain semantic integrity under high compression ratios, a\ntoken-level iterative compression algorithm to better model the interdependence\nbetween compressed contents, and an instruction tuning based method for\ndistribution alignment between language models. We conduct experiments and\nanalysis over four datasets from different scenarios, i.e., GSM8K, BBH,\nShareGPT, and Arxiv-March23; showing that the proposed approach yields\nstate-of-the-art performance and allows for up to 20x compression with little\nperformance loss. Our code is available at https://aka.ms/LLMLingua.'}, {'Published': '2024-04-02', 'Title': 'Learning to Compress Prompt in Natural Language Formats', 'Authors': 'Yu-Neng Chuang, Tianwei Xing, Chia-Yuan Chang, Zirui Liu, Xun Chen, Xia Hu', 'Summary': 'Large language models (LLMs) are great at processing multiple natural\nlanguage processing tasks, but their abilities are constrained by inferior\nperformance with long context, slow inference speed, and the high cost of\ncomputing the results. Deploying LLMs with precise and informative context\nhelps users process large-scale datasets more effectively and cost-efficiently.\nExisting works rely on compressing long prompt contexts into soft prompts.\nHowever, soft prompt compression encounters limitations in transferability\nacross different LLMs, especially API-based LLMs. To this end, this work aims\nto compress lengthy prompts in the form of natural language with LLM\ntransferability. This poses two challenges: (i) Natural Language (NL) prompts\nare incompatible with back-propagation, and (ii) NL prompts lack flexibility in\nimposing length constraints. In this work, we propose a Natural Language Prompt\nEncapsulation (Nano-Capsulator) framework compressing original prompts into NL\nformatted Capsule Prompt while maintaining the prompt utility and\ntransferability. Specifically, to tackle the first challenge, the\nNano-Capsulator is optimized by a reward function that interacts with the\nproposed semantics preserving loss. To address the second question, the\nNano-Capsulator is optimized by a reward function featuring length constraints.\nExperimental results demonstrate that the Capsule Prompt can reduce 81.4% of\nthe original length, decrease inference latency up to 4.5x, and save 80.1% of\nbudget overheads while providing transferability across diverse LLMs and\ndifferent datasets.'}, {'Published': '2024-03-30', 'Title': 'PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression', 'Authors': 'Muhammad Asif Ali, Zhengping Li, Shu Yang, Keyuan Cheng, Yang Cao, Tianhao Huang, Lijie Hu, Lu Yu, Di Wang', 'Summary': "Large language models (LLMs) have shown exceptional abilities for multiple\ndifferent natural language processing tasks. While prompting is a crucial tool\nfor LLM inference, we observe that there is a significant cost associated with\nexceedingly lengthy prompts. Existing attempts to compress lengthy prompts lead\nto sub-standard results in terms of readability and interpretability of the\ncompressed prompt, with a detrimental impact on prompt utility. To address\nthis, we propose PROMPT-SAW: Prompt compresSion via Relation AWare graphs, an\neffective strategy for prompt compression over task-agnostic and task-aware\nprompts. PROMPT-SAW uses the prompt's textual information to build a graph,\nlater extracts key information elements in the graph to come up with the\ncompressed prompt. We also propose GSM8K-AUG, i.e., an extended version of the\nexisting GSM8k benchmark for task-agnostic prompts in order to provide a\ncomprehensive evaluation platform. Experimental evaluation using benchmark\ndatasets shows that prompts compressed by PROMPT-SAW are not only better in\nterms of readability, but they also outperform the best-performing baseline\nmodels by up to 14.3 and 13.7 respectively for task-aware and task-agnostic\nsettings while compressing the original prompt text by 33.0 and 56.7."}, {'Published': '2024-02-25', 'Title': 'Say More with Less: Understanding Prompt Learning Behaviors through Gist Compression', 'Authors': 'Xinze Li, Zhenghao Liu, Chenyan Xiong, Shi Yu, Yukun Yan, Shuo Wang, Ge Yu', 'Summary': 'Large language models (LLMs) require lengthy prompts as the input context to\nproduce output aligned with user intentions, a process that incurs extra costs\nduring inference. In this paper, we propose the Gist COnditioned deCOding\n(Gist-COCO) model, introducing a novel method for compressing prompts which\nalso can assist the prompt interpretation and engineering. Gist-COCO employs an\nencoder-decoder based language model and then incorporates an additional\nencoder as a plugin module to compress prompts with inputs using gist tokens.\nIt finetunes the compression plugin module and uses the representations of gist\ntokens to emulate the raw prompts in the vanilla language model. By verbalizing\nthe representations of gist tokens into gist prompts, the compression ability\nof Gist-COCO can be generalized to different LLMs with high compression rates.\nOur experiments demonstrate that Gist-COCO outperforms previous prompt\ncompression models in both passage and instruction compression tasks. Further\nanalysis on gist verbalization results suggests that our gist prompts serve\ndifferent functions in aiding language models. They may directly provide\npotential answers, generate the chain-of-thought, or simply repeat the inputs.\nAll data and codes are available at https://github.com/OpenMatch/Gist-COCO .'}, {'Published': '2023-10-10', 'Title': 'Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt', 'Authors': 'Zhaozhuo Xu, Zirui Liu, Beidi Chen, Yuxin Tang, Jue Wang, Kaixiong Zhou, Xia Hu, Anshumali Shrivastava', 'Summary': "While the numerous parameters in Large Language Models (LLMs) contribute to\ntheir superior performance, this massive scale makes them inefficient and\nmemory-hungry. Thus, they are hard to deploy on commodity hardware, such as one\nsingle GPU. Given the memory and power constraints of such devices, model\ncompression methods are widely employed to reduce both the model size and\ninference latency, which essentially trades off model quality in return for\nimproved efficiency. Thus, optimizing this accuracy-efficiency trade-off is\ncrucial for the LLM deployment on commodity hardware. In this paper, we\nintroduce a new perspective to optimize this trade-off by prompting compressed\nmodels. Specifically, we first observe that for certain questions, the\ngeneration quality of a compressed LLM can be significantly improved by adding\ncarefully designed hard prompts, though this isn't the case for all questions.\nBased on this observation, we propose a soft prompt learning method where we\nexpose the compressed model to the prompt learning process, aiming to enhance\nthe performance of prompts. Our experimental analysis suggests our soft prompt\nstrategy greatly improves the performance of the 8x compressed LLaMA-7B model\n(with a joint 4-bit quantization and 50% weight pruning compression), allowing\nthem to match their uncompressed counterparts on popular benchmarks. Also, we\ndemonstrate that these learned prompts can be transferred across various\ndatasets, tasks, and compression levels. Hence with this transferability, we\ncan stitch the soft prompt to a newly compressed model to improve the test-time\naccuracy in an ``in-situ'' way."}, {'Published': '2024-04-01', 'Title': 'Efficient Prompting Methods for Large Language Models: A Survey', 'Authors': 'Kaiyan Chang, Songcheng Xu, Chenglong Wang, Yingfeng Luo, Tong Xiao, Jingbo Zhu', 'Summary': 'Prompting has become a mainstream paradigm for adapting large language models\n(LLMs) to specific natural language processing tasks. While this approach opens\nthe door to in-context learning of LLMs, it brings the additional computational\nburden of model inference and human effort of manual-designed prompts,\nparticularly when using lengthy and complex prompts to guide and control the\nbehavior of LLMs. As a result, the LLM field has seen a remarkable surge in\nefficient prompting methods. In this paper, we present a comprehensive overview\nof these methods. At a high level, efficient prompting methods can broadly be\ncategorized into two approaches: prompting with efficient computation and\nprompting with efficient design. The former involves various ways of\ncompressing prompts, and the latter employs techniques for automatic prompt\noptimization. We present the basic concepts of prompting, review the advances\nfor efficient prompting, and highlight future research directions.'}, {'Published': '2023-10-10', 'Title': 'Model Tuning or Prompt Tuning? A Study of Large Language Models for Clinical Concept and Relation Extraction', 'Authors': 'Cheng Peng, Xi Yang, Kaleb E Smith, Zehao Yu, Aokun Chen, Jiang Bian, Yonghui Wu', 'Summary': 'Objective To develop soft prompt-based learning algorithms for large language\nmodels (LLMs), examine the shape of prompts, prompt-tuning using\nfrozen/unfrozen LLMs, transfer learning, and few-shot learning abilities.\nMethods We developed a soft prompt-based LLM model and compared 4 training\nstrategies including (1) fine-tuning without prompts; (2) hard-prompt with\nunfrozen LLMs; (3) soft-prompt with unfrozen LLMs; and (4) soft-prompt with\nfrozen LLMs. We evaluated 7 pretrained LLMs using the 4 training strategies for\nclinical concept and relation extraction on two benchmark datasets. We\nevaluated the transfer learning ability of the prompt-based learning algorithms\nin a cross-institution setting. We also assessed the few-shot learning ability.\nResults and Conclusion When LLMs are unfrozen, GatorTron-3.9B with soft\nprompting achieves the best strict F1-scores of 0.9118 and 0.8604 for concept\nextraction, outperforming the traditional fine-tuning and hard prompt-based\nmodels by 0.6~3.1% and 1.2~2.9%, respectively; GatorTron-345M with soft\nprompting achieves the best F1-scores of 0.8332 and 0.7488 for end-to-end\nrelation extraction, outperforming the other two models by 0.2~2% and\n0.6~11.7%, respectively. When LLMs are frozen, small (i.e., 345 million\nparameters) LLMs have a big gap to be competitive with unfrozen models; scaling\nLLMs up to billions of parameters makes frozen LLMs competitive with unfrozen\nLLMs. For cross-institute evaluation, soft prompting with a frozen\nGatorTron-8.9B model achieved the best performance. This study demonstrates\nthat (1) machines can learn soft prompts better than humans, (2) frozen LLMs\nhave better few-shot learning ability and transfer learning ability to\nfacilitate muti-institution applications, and (3) frozen LLMs require large\nmodels.'}, {'Published': '2024-02-16', 'Title': 'Do Compressed LLMs Forget Knowledge? An Experimental Study with Practical Implications', 'Authors': 'Duc N. M Hoang, Minsik Cho, Thomas Merth, Mohammad Rastegari, Zhangyang Wang', 'Summary': 'Compressing Large Language Models (LLMs) often leads to reduced performance,\nespecially for knowledge-intensive tasks. In this work, we dive into how\ncompression damages LLMs\' inherent knowledge and the possible remedies. We\nstart by proposing two conjectures on the nature of the damage: one is certain\nknowledge being forgotten (or erased) after LLM compression, hence\nnecessitating the compressed model to (re)learn from data with additional\nparameters; the other presumes that knowledge is internally displaced and hence\none requires merely "inference re-direction" with input-side augmentation such\nas prompting, to recover the knowledge-related performance. Extensive\nexperiments are then designed to (in)validate the two conjectures. We observe\nthe promise of prompting in comparison to model tuning; we further unlock\nprompting\'s potential by introducing a variant called Inference-time Dynamic\nPrompting (IDP), that can effectively increase prompt diversity without\nincurring any inference overhead. Our experiments consistently suggest that\ncompared to the classical re-training alternatives such as LoRA, prompting with\nIDP leads to better or comparable post-compression performance recovery, while\nsaving the extra parameter size by 21x and reducing inference latency by 60%.\nOur experiments hence strongly endorse the conjecture of "knowledge displaced"\nover "knowledge forgotten", and shed light on a new efficient mechanism to\nrestore compressed LLM performance. We additionally visualize and analyze the\ndifferent attention and activation patterns between prompted and re-trained\nmodels, demonstrating they achieve performance recovery in two different\nregimes.'}][0m[32;1m[1;3mHere are some research papers on the topic Prompt Compression in LLM Applications:
    
    1. "SelfCP: Compressing Long Prompt to 1/12 Using the Frozen Large Language Model Itself" by Jun Gao
    2. "Adapting LLMs for Efficient Context Processing through Soft Prompt Compression" by Cangqing Wang, Yutian Yang, Ruisi Li, Dan Sun, Ruicong Cai, Yuzhu Zhang, Chengqian Fu, Lillian Floyd
    3. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models" by Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu
    4. "Learning to Compress Prompt in Natural Language Formats" by Yu-Neng Chuang, Tianwei Xing, Chia-Yuan Chang, Zirui Liu, Xun Chen, Xia Hu
    5. "PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression"[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'Get me a list of research papers on the topic Prompt Compression in LLM Applications.',
     'chat_history': '',
     'output': 'Here are some research papers on the topic Prompt Compression in LLM Applications:\n\n1. "SelfCP: Compressing Long Prompt to 1/12 Using the Frozen Large Language Model Itself" by Jun Gao\n2. "Adapting LLMs for Efficient Context Processing through Soft Prompt Compression" by Cangqing Wang, Yutian Yang, Ruisi Li, Dan Sun, Ruicong Cai, Yuzhu Zhang, Chengqian Fu, Lillian Floyd\n3. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models" by Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu\n4. "Learning to Compress Prompt in Natural Language Formats" by Yu-Neng Chuang, Tianwei Xing, Chia-Yuan Chang, Zirui Liu, Xun Chen, Xia Hu\n5. "PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression"'}




```python
agent_executor.invoke({"input": "What paper did we speak about from our chat history?"})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `get_metadata_information_from_arxiv` with `{'word': 'chat history'}`
    responded: I need to access the chat history to answer this question. 
    
    [0m[33;1m[1;3m[{'Published': '2023-10-20', 'Title': 'Towards Detecting Contextual Real-Time Toxicity for In-Game Chat', 'Authors': 'Zachary Yang, Nicolas Grenan-Godbout, Reihaneh Rabbany', 'Summary': "Real-time toxicity detection in online environments poses a significant\nchallenge, due to the increasing prevalence of social media and gaming\nplatforms. We introduce ToxBuster, a simple and scalable model that reliably\ndetects toxic content in real-time for a line of chat by including chat history\nand metadata. ToxBuster consistently outperforms conventional toxicity models\nacross popular multiplayer games, including Rainbow Six Siege, For Honor, and\nDOTA 2. We conduct an ablation study to assess the importance of each model\ncomponent and explore ToxBuster's transferability across the datasets.\nFurthermore, we showcase ToxBuster's efficacy in post-game moderation,\nsuccessfully flagging 82.1% of chat-reported players at a precision level of\n90.0%. Additionally, we show how an additional 6% of unreported toxic players\ncan be proactively moderated."}, {'Published': '2021-07-13', 'Title': "A First Look at Developers' Live Chat on Gitter", 'Authors': 'Lin Shi, Xiao Chen, Ye Yang, Hanzhi Jiang, Ziyou Jiang, Nan Niu, Qing Wang', 'Summary': "Modern communication platforms such as Gitter and Slack play an increasingly\ncritical role in supporting software teamwork, especially in open source\ndevelopment.Conversations on such platforms often contain intensive, valuable\ninformation that may be used for better understanding OSS developer\ncommunication and collaboration. However, little work has been done in this\nregard. To bridge the gap, this paper reports a first comprehensive empirical\nstudy on developers' live chat, investigating when they interact, what\ncommunity structures look like, which topics are discussed, and how they\ninteract. We manually analyze 749 dialogs in the first phase, followed by an\nautomated analysis of over 173K dialogs in the second phase. We find that\ndevelopers tend to converse more often on weekdays, especially on Wednesdays\nand Thursdays (UTC), that there are three common community structures observed,\nthat developers tend to discuss topics such as API usages and errors, and that\nsix dialog interaction patterns are identified in the live chat communities.\nBased on the findings, we provide recommendations for individual developers and\nOSS communities, highlight desired features for platform vendors, and shed\nlight on future research directions. We believe that the findings and insights\nwill enable a better understanding of developers' live chat, pave the way for\nother researchers, as well as a better utilization and mining of knowledge\nembedded in the massive chat history."}, {'Published': '2022-02-28', 'Title': 'MSCTD: A Multimodal Sentiment Chat Translation Dataset', 'Authors': 'Yunlong Liang, Fandong Meng, Jinan Xu, Yufeng Chen, Jie Zhou', 'Summary': 'Multimodal machine translation and textual chat translation have received\nconsiderable attention in recent years. Although the conversation in its\nnatural form is usually multimodal, there still lacks work on multimodal\nmachine translation in conversations. In this work, we introduce a new task\nnamed Multimodal Chat Translation (MCT), aiming to generate more accurate\ntranslations with the help of the associated dialogue history and visual\ncontext. To this end, we firstly construct a Multimodal Sentiment Chat\nTranslation Dataset (MSCTD) containing 142,871 English-Chinese utterance pairs\nin 14,762 bilingual dialogues and 30,370 English-German utterance pairs in\n3,079 bilingual dialogues. Each utterance pair, corresponding to the visual\ncontext that reflects the current conversational scene, is annotated with a\nsentiment label. Then, we benchmark the task by establishing multiple baseline\nsystems that incorporate multimodal and sentiment features for MCT. Preliminary\nexperiments on four language directions (English-Chinese and English-German)\nverify the potential of contextual and multimodal information fusion and the\npositive impact of sentiment on the MCT task. Additionally, as a by-product of\nthe MSCTD, it also provides two new benchmarks on multimodal dialogue sentiment\nanalysis. Our work can facilitate research on both multimodal chat translation\nand multimodal dialogue sentiment analysis.'}, {'Published': '2021-09-15', 'Title': 'ISPY: Automatic Issue-Solution Pair Extraction from Community Live Chats', 'Authors': 'Lin Shi, Ziyou Jiang, Ye Yang, Xiao Chen, Yumin Zhang, Fangwen Mu, Hanzhi Jiang, Qing Wang', 'Summary': 'Collaborative live chats are gaining popularity as a development\ncommunication tool. In community live chatting, developers are likely to post\nissues they encountered (e.g., setup issues and compile issues), and other\ndevelopers respond with possible solutions. Therefore, community live chats\ncontain rich sets of information for reported issues and their corresponding\nsolutions, which can be quite useful for knowledge sharing and future reuse if\nextracted and restored in time. However, it remains challenging to accurately\nmine such knowledge due to the noisy nature of interleaved dialogs in live chat\ndata. In this paper, we first formulate the problem of issue-solution pair\nextraction from developer live chat data, and propose an automated approach,\nnamed ISPY, based on natural language processing and deep learning techniques\nwith customized enhancements, to address the problem. Specifically, ISPY\nautomates three tasks: 1) Disentangle live chat logs, employing a feedforward\nneural network to disentangle a conversation history into separate dialogs\nautomatically; 2) Detect dialogs discussing issues, using a novel convolutional\nneural network (CNN), which consists of a BERT-based utterance embedding layer,\na context-aware dialog embedding layer, and an output layer; 3) Extract\nappropriate utterances and combine them as corresponding solutions, based on\nthe same CNN structure but with different feeding inputs. To evaluate ISPY, we\ncompare it with six baselines, utilizing a dataset with 750 dialogs including\n171 issue-solution pairs and evaluate ISPY from eight open source communities.\nThe results show that, for issue-detection, our approach achieves the F1 of\n76%, and outperforms all baselines by 30%. Our approach achieves the F1 of 63%\nfor solution-extraction and outperforms the baselines by 20%.'}, {'Published': '2023-05-23', 'Title': 'ChatGPT-EDSS: Empathetic Dialogue Speech Synthesis Trained from ChatGPT-derived Context Word Embeddings', 'Authors': 'Yuki Saito, Shinnosuke Takamichi, Eiji Iimori, Kentaro Tachibana, Hiroshi Saruwatari', 'Summary': "We propose ChatGPT-EDSS, an empathetic dialogue speech synthesis (EDSS)\nmethod using ChatGPT for extracting dialogue context. ChatGPT is a chatbot that\ncan deeply understand the content and purpose of an input prompt and\nappropriately respond to the user's request. We focus on ChatGPT's reading\ncomprehension and introduce it to EDSS, a task of synthesizing speech that can\nempathize with the interlocutor's emotion. Our method first gives chat history\nto ChatGPT and asks it to generate three words representing the intention,\nemotion, and speaking style for each line in the chat. Then, it trains an EDSS\nmodel using the embeddings of ChatGPT-derived context words as the conditioning\nfeatures. The experimental results demonstrate that our method performs\ncomparably to ones using emotion labels or neural network-derived context\nembeddings learned from chat histories. The collected ChatGPT-derived context\ninformation is available at\nhttps://sarulab-speech.github.io/demo_ChatGPT_EDSS/."}, {'Published': '2019-06-04', 'Title': 'Joint Effects of Context and User History for Predicting Online Conversation Re-entries', 'Authors': 'Xingshan Zeng, Jing Li, Lu Wang, Kam-Fai Wong', 'Summary': "As the online world continues its exponential growth, interpersonal\ncommunication has come to play an increasingly central role in opinion\nformation and change. In order to help users better engage with each other\nonline, we study a challenging problem of re-entry prediction foreseeing\nwhether a user will come back to a conversation they once participated in. We\nhypothesize that both the context of the ongoing conversations and the users'\nprevious chatting history will affect their continued interests in future\nengagement. Specifically, we propose a neural framework with three main layers,\neach modeling context, user history, and interactions between them, to explore\nhow the conversation context and user chatting history jointly result in their\nre-entry behavior. We experiment with two large-scale datasets collected from\nTwitter and Reddit. Results show that our proposed framework with bi-attention\nachieves an F1 score of 61.1 on Twitter conversations, outperforming the\nstate-of-the-art methods from previous work."}, {'Published': '2022-01-27', 'Title': 'Group Chat Ecology in Enterprise Instant Messaging: How Employees Collaborate Through Multi-User Chat Channels on Slack', 'Authors': 'Dakuo Wang, Haoyu Wang, Mo Yu, Zahra Ashktorab, Ming Tan', 'Summary': "Despite the long history of studying instant messaging usage, we know very\nlittle about how today's people participate in group chat channels and interact\nwith others inside a real-world organization. In this short paper, we aim to\nupdate the existing knowledge on how group chat is used in the context of\ntoday's organizations. The knowledge is particularly important for the new norm\nof remote works under the COVID-19 pandemic. We have the privilege of\ncollecting two valuable datasets: a total of 4,300 group chat channels in Slack\nfrom an R&D department in a multinational IT company; and a total of 117\ngroups' performance data. Through qualitative coding of 100 randomly sampled\ngroup channels from the 4,300 channels dataset, we identified and reported 9\ncategories such as Project channels, IT-Support channels, and Event channels.\nWe further defined a feature metric with 21 meta features (and their derived\nfeatures) without looking at the message content to depict the group\ncommunication style for these group chat channels, with which we successfully\ntrained a machine learning model that can automatically classify a given group\nchannel into one of the 9 categories. In addition to the descriptive data\nanalysis, we illustrated how these communication metrics can be used to analyze\nteam performance. We cross-referenced 117 project teams and their team-based\nSlack channels and identified 57 teams that appeared in both datasets, then we\nbuilt a regression model to reveal the relationship between these group\ncommunication styles and the project team performance. This work contributes an\nupdated empirical understanding of human-human communication practices within\nthe enterprise setting, and suggests design opportunities for the future of\nhuman-AI communication experience."}, {'Published': '2023-05-21', 'Title': 'ToxBuster: In-game Chat Toxicity Buster with BERT', 'Authors': 'Zachary Yang, Yasmine Maricar, MohammadReza Davari, Nicolas Grenon-Godbout, Reihaneh Rabbany', 'Summary': 'Detecting toxicity in online spaces is challenging and an ever more pressing\nproblem given the increase in social media and gaming consumption. We introduce\nToxBuster, a simple and scalable model trained on a relatively large dataset of\n194k lines of game chat from Rainbow Six Siege and For Honor, carefully\nannotated for different kinds of toxicity. Compared to the existing\nstate-of-the-art, ToxBuster achieves 82.95% (+7) in precision and 83.56% (+57)\nin recall. This improvement is obtained by leveraging past chat history and\nmetadata. We also study the implication towards real-time and post-game\nmoderation as well as the model transferability from one game to another.'}, {'Published': '2023-07-30', 'Title': 'ChatGPT is Good but Bing Chat is Better for Vietnamese Students', 'Authors': 'Xuan-Quy Dao, Ngoc-Bich Le', 'Summary': 'This study examines the efficacy of two SOTA large language models (LLMs),\nnamely ChatGPT and Microsoft Bing Chat (BingChat), in catering to the needs of\nVietnamese students. Although ChatGPT exhibits proficiency in multiple\ndisciplines, Bing Chat emerges as the more advantageous option. We conduct a\ncomparative analysis of their academic achievements in various disciplines,\nencompassing mathematics, literature, English language, physics, chemistry,\nbiology, history, geography, and civic education. The results of our study\nsuggest that BingChat demonstrates superior performance compared to ChatGPT\nacross a wide range of subjects, with the exception of literature, where\nChatGPT exhibits better performance. Additionally, BingChat utilizes the more\nadvanced GPT-4 technology in contrast to ChatGPT, which is built upon GPT-3.5.\nThis allows BingChat to improve to comprehension, reasoning and generation of\ncreative and informative text. Moreover, the fact that BingChat is accessible\nin Vietnam and its integration of hyperlinks and citations within responses\nserve to reinforce its superiority. In our analysis, it is evident that while\nChatGPT exhibits praiseworthy qualities, BingChat presents a more apdated\nsolutions for Vietnamese students.'}, {'Published': '2020-04-23', 'Title': 'Distilling Knowledge for Fast Retrieval-based Chat-bots', 'Authors': 'Amir Vakili Tahami, Kamyar Ghajar, Azadeh Shakery', 'Summary': 'Response retrieval is a subset of neural ranking in which a model selects a\nsuitable response from a set of candidates given a conversation history.\nRetrieval-based chat-bots are typically employed in information seeking\nconversational systems such as customer support agents. In order to make\npairwise comparisons between a conversation history and a candidate response,\ntwo approaches are common: cross-encoders performing full self-attention over\nthe pair and bi-encoders encoding the pair separately. The former gives better\nprediction quality but is too slow for practical use. In this paper, we propose\na new cross-encoder architecture and transfer knowledge from this model to a\nbi-encoder model using distillation. This effectively boosts bi-encoder\nperformance at no cost during inference time. We perform a detailed analysis of\nthis approach on three response retrieval datasets.'}][0m[32;1m[1;3mThe paper we spoke about from our chat history is "ToxBuster: In-game Chat Toxicity Buster with BERT" by Zachary Yang, Yasmine Maricar, MohammadReza Davari, Nicolas Grenon-Godbout, and Reihaneh Rabbany.[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'What paper did we speak about from our chat history?',
     'chat_history': 'Human: Get me a list of research papers on the topic Prompt Compression in LLM Applications.\nAI: Here are some research papers on the topic Prompt Compression in LLM Applications:\n\n1. "SelfCP: Compressing Long Prompt to 1/12 Using the Frozen Large Language Model Itself" by Jun Gao\n2. "Adapting LLMs for Efficient Context Processing through Soft Prompt Compression" by Cangqing Wang, Yutian Yang, Ruisi Li, Dan Sun, Ruicong Cai, Yuzhu Zhang, Chengqian Fu, Lillian Floyd\n3. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models" by Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu\n4. "Learning to Compress Prompt in Natural Language Formats" by Yu-Neng Chuang, Tianwei Xing, Chia-Yuan Chang, Zirui Liu, Xun Chen, Xia Hu\n5. "PROMPT-SAW: Leveraging Relation-Aware Graphs for Textual Prompt Compression"',
     'output': 'The paper we spoke about from our chat history is "ToxBuster: In-game Chat Toxicity Buster with BERT" by Zachary Yang, Yasmine Maricar, MohammadReza Davari, Nicolas Grenon-Godbout, and Reihaneh Rabbany.'}


