## Fireworks.AI + LangChain + RAG
 
[Fireworks AI](https://python.langchain.com/docs/integrations/llms/fireworks) wants to provide the best experience when working with LangChain, and here is an example of Fireworks + LangChain doing RAG

See [our models page](https://fireworks.ai/models) for the full list of models. We use `accounts/fireworks/models/mixtral-8x7b-instruct` for RAG In this tutorial.

For the RAG target, we will use the Gemma technical report https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf 


```python
%pip install --quiet pypdf langchain-chroma tiktoken openai 
%pip uninstall -y langchain-fireworks
%pip install --editable /mnt/disks/data/langchain/libs/partners/fireworks
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    Found existing installation: langchain-fireworks 0.0.1
    Uninstalling langchain-fireworks-0.0.1:
      Successfully uninstalled langchain-fireworks-0.0.1
    Note: you may need to restart the kernel to use updated packages.
    Obtaining file:///mnt/disks/data/langchain/libs/partners/fireworks
      Installing build dependencies ... [?25ldone
    [?25h  Checking if build backend supports build_editable ... [?25ldone
    [?25h  Getting requirements to build editable ... [?25ldone
    [?25h  Preparing editable metadata (pyproject.toml) ... [?25ldone
    [?25hRequirement already satisfied: aiohttp<4.0.0,>=3.9.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-fireworks==0.0.1) (3.9.3)
    Requirement already satisfied: fireworks-ai<0.13.0,>=0.12.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-fireworks==0.0.1) (0.12.0)
    Requirement already satisfied: langchain-core<0.2,>=0.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-fireworks==0.0.1) (0.1.23)
    Requirement already satisfied: requests<3,>=2 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-fireworks==0.0.1) (2.31.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks==0.0.1) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks==0.0.1) (23.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks==0.0.1) (1.4.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks==0.0.1) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks==0.0.1) (1.9.2)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from aiohttp<4.0.0,>=3.9.1->langchain-fireworks==0.0.1) (4.0.3)
    Requirement already satisfied: httpx in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (0.26.0)
    Requirement already satisfied: httpx-sse in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (0.4.0)
    Requirement already satisfied: pydantic in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (2.4.2)
    Requirement already satisfied: Pillow in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (10.2.0)
    Requirement already satisfied: PyYAML>=5.3 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (6.0.1)
    Requirement already satisfied: anyio<5,>=3 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (3.7.1)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (1.33)
    Requirement already satisfied: langsmith<0.2.0,>=0.1.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (0.1.5)
    Requirement already satisfied: packaging<24.0,>=23.2 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (23.2)
    Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (8.2.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from requests<3,>=2->langchain-fireworks==0.0.1) (3.3.0)
    Requirement already satisfied: idna<4,>=2.5 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from requests<3,>=2->langchain-fireworks==0.0.1) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from requests<3,>=2->langchain-fireworks==0.0.1) (2.0.6)
    Requirement already satisfied: certifi>=2017.4.17 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from requests<3,>=2->langchain-fireworks==0.0.1) (2023.7.22)
    Requirement already satisfied: sniffio>=1.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from anyio<5,>=3->langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (1.3.0)
    Requirement already satisfied: exceptiongroup in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from anyio<5,>=3->langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (1.1.3)
    Requirement already satisfied: jsonpointer>=1.9 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from jsonpatch<2.0,>=1.33->langchain-core<0.2,>=0.1->langchain-fireworks==0.0.1) (2.4)
    Requirement already satisfied: annotated-types>=0.4.0 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from pydantic->fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (0.5.0)
    Requirement already satisfied: pydantic-core==2.10.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from pydantic->fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (2.10.1)
    Requirement already satisfied: typing-extensions>=4.6.1 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from pydantic->fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (4.8.0)
    Requirement already satisfied: httpcore==1.* in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from httpx->fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in /mnt/disks/data/langchain/.venv/lib/python3.9/site-packages (from httpcore==1.*->httpx->fireworks-ai<0.13.0,>=0.12.0->langchain-fireworks==0.0.1) (0.14.0)
    Building wheels for collected packages: langchain-fireworks
      Building editable for langchain-fireworks (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for langchain-fireworks: filename=langchain_fireworks-0.0.1-py3-none-any.whl size=2228 sha256=564071b120b09ec31f2dc737733448a33bbb26e40b49fcde0c129ad26045259d
      Stored in directory: /tmp/pip-ephem-wheel-cache-oz368vdk/wheels/e0/ad/31/d7e76dd73d61905ff7f369f5b0d21a4b5e7af4d3cb7487aece
    Successfully built langchain-fireworks
    Installing collected packages: langchain-fireworks
    Successfully installed langchain-fireworks-0.0.1
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    


```python
import fireworks

print(fireworks)
import fireworks.client
```

    <module 'fireworks' from '/mnt/disks/data/langchain/.venv/lib/python3.9/site-packages/fireworks/__init__.py'>
    


```python
# Load
import requests
from langchain_community.document_loaders import PyPDFLoader

# Download the PDF from a URL and save it to a temporary location
url = "https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf"
response = requests.get(url, stream=True)
file_name = "temp_file.pdf"
with open(file_name, "wb") as pdf:
    pdf.write(response.content)

loader = PyPDFLoader(file_name)
data = loader.load()

# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
from langchain_chroma import Chroma
from langchain_fireworks.embeddings import FireworksEmbeddings

vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=FireworksEmbeddings(),
)

retriever = vectorstore.as_retriever()
```


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
from langchain_together import Together

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.0,
    max_tokens=2000,
    top_k=1,
)

# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)
```


```python
chain.invoke("What are the Architectural details of Mixtral?")
```




    '\nAnswer: The architectural details of Mixtral are as follows:\n- Dimension (dim): 4096\n- Number of layers (n\\_layers): 32\n- Dimension of each head (head\\_dim): 128\n- Hidden dimension (hidden\\_dim): 14336\n- Number of heads (n\\_heads): 32\n- Number of kv heads (n\\_kv\\_heads): 8\n- Context length (context\\_len): 32768\n- Vocabulary size (vocab\\_size): 32000\n- Number of experts (num\\_experts): 8\n- Number of top k experts (top\\_k\\_experts): 2\n\nMixtral is based on a transformer architecture and uses the same modifications as described in [18], with the notable exceptions that Mixtral supports a fully dense context length of 32k tokens, and the feedforward block picks from a set of 8 distinct groups of parameters. At every layer, for every token, a router network chooses two of these groups (the “experts”) to process the token and combine their output additively. This technique increases the number of parameters of a model while controlling cost and latency, as the model only uses a fraction of the total set of parameters per token. Mixtral is pretrained with multilingual data using a context size of 32k tokens. It either matches or exceeds the performance of Llama 2 70B and GPT-3.5, over several benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks.'



Trace: 

https://smith.langchain.com/public/935fd642-06a6-4b42-98e3-6074f93115cd/r
