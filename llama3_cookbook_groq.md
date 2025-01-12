# Llama 3 Cookbook with LlamaIndex and Groq

<a href="https://colab.research.google.com/github/meta-llama/llama-recipes/blob/main/recipes/llama_api_providers/llama3_cookbook_groq.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Meta developed and released the Meta [Llama 3](https://ai.meta.com/blog/meta-llama-3/) family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks.

In this notebook, we demonstrate how to use Llama 3 with LlamaIndex for a comprehensive set of use cases. 
1. Basic completion / chat 
2. Basic RAG (Vector Search, Summarization)
3. Advanced RAG (Routing)
4. Text-to-SQL 
5. Structured Data Extraction
6. Chat Engine + Memory
7. Agents


We use Llama3-8B and Llama3-70B through [Groq](https://groq.com) - you can sign up there to get a free trial API key.

## Installation and Setup


```python
!pip install llama-index
!pip install llama-index-llms-groq
!pip install llama-index-embeddings-huggingface
!pip install llama-parse
```

    Requirement already satisfied: llama-index in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (0.10.16)
    Requirement already satisfied: llama-index-embeddings-openai<0.2.0,>=0.1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.6)
    Requirement already satisfied: llama-index-cli<0.2.0,>=0.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.7)
    Requirement already satisfied: llama-index-multi-modal-llms-openai<0.2.0,>=0.1.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.4)
    Requirement already satisfied: llama-index-agent-openai<0.2.0,>=0.1.4 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.5)
    Requirement already satisfied: llama-index-question-gen-openai<0.2.0,>=0.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.3)
    Requirement already satisfied: llama-index-readers-file<0.2.0,>=0.1.4 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.8)
    Requirement already satisfied: llama-index-legacy<0.10.0,>=0.9.48 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.9.48)
    Requirement already satisfied: llama-index-readers-llama-parse<0.2.0,>=0.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.3)
    Requirement already satisfied: llama-index-llms-openai<0.2.0,>=0.1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.7)
    Requirement already satisfied: llama-index-indices-managed-llama-cloud<0.2.0,>=0.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.3)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.16 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.10.16.post1)
    Requirement already satisfied: llama-index-program-openai<0.2.0,>=0.1.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index) (0.1.4)
    Requirement already satisfied: llama-index-vector-stores-chroma<0.2.0,>=0.1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.1.5)
    Requirement already satisfied: SQLAlchemy[asyncio]>=1.4.49 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (2.0.25)
    Requirement already satisfied: deprecated>=1.2.9.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (1.2.14)
    Requirement already satisfied: typing-inspect>=0.8.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (0.9.0)
    Requirement already satisfied: pandas in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (1.5.1)
    Requirement already satisfied: dataclasses-json in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (0.6.4)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (3.8.1)
    Requirement already satisfied: requests>=2.31.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (2.31.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (2024.2.0)
    Requirement already satisfied: numpy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (1.23.4)
    Requirement already satisfied: pillow>=9.0.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (10.2.0)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (3.2.1)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (3.9.3)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (4.66.1)
    Requirement already satisfied: typing-extensions>=4.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (4.9.0)
    Requirement already satisfied: tiktoken>=0.3.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (0.5.2)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (1.0.8)
    Requirement already satisfied: httpx in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (0.26.0)
    Requirement already satisfied: tenacity<9.0.0,>=8.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (8.2.3)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (0.1.13)
    Requirement already satisfied: PyYAML>=6.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (6.0.1)
    Requirement already satisfied: openai>=1.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.16->llama-index) (1.13.3)
    Requirement already satisfied: pypdf<5.0.0,>=4.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index) (4.1.0)
    Requirement already satisfied: pymupdf<2.0.0,>=1.23.21 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index) (1.23.26)
    Requirement already satisfied: beautifulsoup4<5.0.0,>=4.12.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index) (4.12.3)
    Requirement already satisfied: bs4<0.0.3,>=0.0.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index) (0.0.2)
    Requirement already satisfied: llama-parse<0.4.0,>=0.3.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-readers-llama-parse<0.2.0,>=0.1.2->llama-index) (0.3.7)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.4.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.16->llama-index) (4.0.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.16->llama-index) (22.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.9.4)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.16->llama-index) (6.0.5)
    Requirement already satisfied: soupsieve>1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from beautifulsoup4<5.0.0,>=4.12.3->llama-index-readers-file<0.2.0,>=0.1.4->llama-index) (2.3.2.post1)
    Requirement already satisfied: wrapt<2,>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from deprecated>=1.2.9.3->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.16.0)
    Requirement already satisfied: onnxruntime<2.0.0,>=1.17.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.17.1)
    Requirement already satisfied: tokenizers<0.16.0,>=0.15.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.15.1)
    Requirement already satisfied: chromadb<0.5.0,>=0.4.22 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.4.24)
    Requirement already satisfied: pydantic>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core<0.11.0,>=0.10.16->llama-index) (2.5.1)
    

    Requirement already satisfied: sniffio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.3.0)
    Requirement already satisfied: idna in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (3.4)
    Requirement already satisfied: anyio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (3.7.1)
    Requirement already satisfied: certifi in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.0.2)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (0.14.0)
    Requirement already satisfied: regex>=2021.8.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.16->llama-index) (2023.12.25)
    Requirement already satisfied: click in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.16->llama-index) (8.1.7)
    Requirement already satisfied: joblib in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.3.2)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.9.0)
    Requirement already satisfied: PyMuPDFb==1.23.22 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pymupdf<2.0.0,>=1.23.21->llama-index-readers-file<0.2.0,>=0.1.4->llama-index) (1.23.22)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.16->llama-index) (2.2.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.16->llama-index) (3.3.2)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.16->llama-index) (3.0.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.16->llama-index) (3.20.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.16->llama-index) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.16->llama-index) (2022.5)
    Requirement already satisfied: exceptiongroup in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from anyio->httpx->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.2.0)
    Requirement already satisfied: opentelemetry-instrumentation-fastapi>=0.41b0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.44b0)
    Requirement already satisfied: overrides>=7.3.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (7.7.0)
    Requirement already satisfied: orjson>=3.9.12 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.9.15)
    Requirement already satisfied: fastapi>=0.95.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.104.1)
    Requirement already satisfied: chroma-hnswlib==0.7.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.7.3)
    Requirement already satisfied: uvicorn[standard]>=0.18.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.24.0.post1)
    Requirement already satisfied: kubernetes>=28.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (29.0.0)
    Requirement already satisfied: opentelemetry-sdk>=1.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.23.0)
    Requirement already satisfied: posthog>=2.4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.5.0)
    Requirement already satisfied: build>=1.0.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.1.1)
    Requirement already satisfied: importlib-resources in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (6.1.2)
    Requirement already satisfied: typer>=0.9.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.9.0)
    Requirement already satisfied: opentelemetry-exporter-otlp-proto-grpc>=1.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.23.0)
    Requirement already satisfied: opentelemetry-api>=1.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.23.0)
    Requirement already satisfied: grpcio>=1.58.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.62.1)
    Requirement already satisfied: pypika>=0.48.9 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.48.9)
    Requirement already satisfied: mmh3>=4.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (4.1.0)
    Requirement already satisfied: bcrypt>=4.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (4.1.2)
    Requirement already satisfied: pulsar-client>=3.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.4.0)
    Requirement already satisfied: packaging>=17.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from marshmallow<4.0.0,>=3.18.0->dataclasses-json->llama-index-core<0.11.0,>=0.10.16->llama-index) (23.2)
    

    Requirement already satisfied: flatbuffers in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from onnxruntime<2.0.0,>=1.17.0->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (23.5.26)
    Requirement already satisfied: protobuf in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from onnxruntime<2.0.0,>=1.17.0->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (4.25.2)
    Requirement already satisfied: coloredlogs in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from onnxruntime<2.0.0,>=1.17.0->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (15.0.1)
    Requirement already satisfied: sympy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from onnxruntime<2.0.0,>=1.17.0->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.12)
    Requirement already satisfied: annotated-types>=0.4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core<0.11.0,>=0.10.16->llama-index) (0.6.0)
    Requirement already satisfied: pydantic-core==2.14.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core<0.11.0,>=0.10.16->llama-index) (2.14.3)
    Requirement already satisfied: six>=1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas->llama-index-core<0.11.0,>=0.10.16->llama-index) (1.16.0)
    Requirement already satisfied: huggingface_hub<1.0,>=0.16.4 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from tokenizers<0.16.0,>=0.15.1->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.20.3)
    Requirement already satisfied: tomli>=1.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from build>=1.0.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (2.0.1)
    Requirement already satisfied: pyproject_hooks in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from build>=1.0.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.0.0)
    Requirement already satisfied: starlette<0.28.0,>=0.27.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from fastapi>=0.95.2->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.27.0)
    Requirement already satisfied: filelock in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface_hub<1.0,>=0.16.4->tokenizers<0.16.0,>=0.15.1->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.13.1)
    Requirement already satisfied: google-auth>=1.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (2.28.1)
    Requirement already satisfied: websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.4.1)
    Requirement already satisfied: requests-oauthlib in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.3.1)
    Requirement already satisfied: oauthlib>=3.2.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.2.2)
    Requirement already satisfied: importlib-metadata<7.0,>=6.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-api>=1.2.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (6.11.0)
    Requirement already satisfied: opentelemetry-exporter-otlp-proto-common==1.23.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.23.0)
    Requirement already satisfied: opentelemetry-proto==1.23.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.23.0)
    Requirement already satisfied: googleapis-common-protos~=1.52 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.62.0)
    Requirement already satisfied: opentelemetry-semantic-conventions==0.44b0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.44b0)
    Requirement already satisfied: opentelemetry-util-http==0.44b0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.44b0)
    Requirement already satisfied: opentelemetry-instrumentation-asgi==0.44b0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.44b0)
    Requirement already satisfied: opentelemetry-instrumentation==0.44b0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.44b0)
    Requirement already satisfied: setuptools>=16.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-instrumentation==0.44b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (58.1.0)
    Requirement already satisfied: asgiref~=3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from opentelemetry-instrumentation-asgi==0.44b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.7.2)
    Requirement already satisfied: backoff>=1.10.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from posthog>=2.4.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (2.2.1)
    Requirement already satisfied: monotonic>=1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from posthog>=2.4.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.6)
    Requirement already satisfied: watchfiles>=0.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from uvicorn[standard]>=0.18.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.21.0)
    Requirement already satisfied: uvloop!=0.15.0,!=0.15.1,>=0.14.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from uvicorn[standard]>=0.18.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.19.0)
    Requirement already satisfied: httptools>=0.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from uvicorn[standard]>=0.18.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.6.1)
    Requirement already satisfied: python-dotenv>=0.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from uvicorn[standard]>=0.18.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.0.1)
    Requirement already satisfied: websockets>=10.4 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from uvicorn[standard]>=0.18.3->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (12.0)
    Requirement already satisfied: humanfriendly>=9.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from coloredlogs->onnxruntime<2.0.0,>=1.17.0->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (10.0)
    Requirement already satisfied: mpmath>=0.19 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from sympy->onnxruntime<2.0.0,>=1.17.0->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (1.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (4.9)
    

    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.3.0)
    Requirement already satisfied: zipp>=0.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from importlib-metadata<7.0,>=6.0->opentelemetry-api>=1.2.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (3.17.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth>=1.0.1->kubernetes>=28.1.0->chromadb<0.5.0,>=0.4.22->llama-index-vector-stores-chroma<0.2.0,>=0.1.1->llama-index-cli<0.2.0,>=0.1.2->llama-index) (0.5.1)
    [33mWARNING: You are using pip version 22.0.4; however, version 24.0 is available.
    You should consider upgrading via the '/Users/daniel/.pyenv/versions/3.10.3/bin/python3.10 -m pip install --upgrade pip' command.[0m[33m
    [0mRequirement already satisfied: llama-index-llms-groq in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (0.1.3)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-llms-groq) (0.10.16.post1)
    Requirement already satisfied: llama-index-llms-openai-like<0.2.0,>=0.1.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-llms-groq) (0.1.3)
    Requirement already satisfied: openai>=1.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.13.3)
    Requirement already satisfied: requests>=2.31.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.31.0)
    Requirement already satisfied: SQLAlchemy[asyncio]>=1.4.49 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.0.25)
    Requirement already satisfied: dataclasses-json in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.6.4)
    Requirement already satisfied: pillow>=9.0.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (10.2.0)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.9.3)
    Requirement already satisfied: httpx in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.26.0)
    Requirement already satisfied: numpy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.23.4)
    Requirement already satisfied: pandas in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.5.1)
    Requirement already satisfied: deprecated>=1.2.9.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.2.14)
    Requirement already satisfied: tenacity<9.0.0,>=8.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (8.2.3)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (4.66.1)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.8.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.2.0)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.1.13)
    Requirement already satisfied: networkx>=3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.2.1)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.0.8)
    Requirement already satisfied: tiktoken>=0.3.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.5.2)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.6.0)
    Requirement already satisfied: PyYAML>=6.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (6.0.1)
    Requirement already satisfied: typing-inspect>=0.8.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.9.0)
    Requirement already satisfied: typing-extensions>=4.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (4.9.0)
    Requirement already satisfied: transformers<5.0.0,>=4.37.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (4.37.2)
    Requirement already satisfied: llama-index-llms-openai<0.2.0,>=0.1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.1.7)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.9.4)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (6.0.5)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.3.1)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.4.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (4.0.3)
    Requirement already satisfied: attrs>=17.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (22.1.0)
    Requirement already satisfied: wrapt<2,>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from deprecated>=1.2.9.3->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.16.0)
    Requirement already satisfied: pydantic>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.5.1)
    

    Requirement already satisfied: anyio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.7.1)
    Requirement already satisfied: idna in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.4)
    Requirement already satisfied: certifi in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.0.2)
    Requirement already satisfied: sniffio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.3.0)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.14.0)
    Requirement already satisfied: joblib in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.3.2)
    Requirement already satisfied: click in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (8.1.7)
    Requirement already satisfied: regex>=2021.8.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2023.12.25)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.9.0)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.2.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.3.2)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.0.1)
    Requirement already satisfied: safetensors>=0.4.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.4.2)
    Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.20.3)
    Requirement already satisfied: packaging>=20.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (23.2)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.15.1)
    Requirement already satisfied: filelock in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (3.13.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.20.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2022.5)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.8.2)
    Requirement already satisfied: exceptiongroup in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from anyio->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.2.0)
    Requirement already satisfied: annotated-types>=0.4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.6.0)
    Requirement already satisfied: pydantic-core==2.14.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.14.3)
    Requirement already satisfied: six>=1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.16.0)
    [33mWARNING: You are using pip version 22.0.4; however, version 24.0 is available.
    You should consider upgrading via the '/Users/daniel/.pyenv/versions/3.10.3/bin/python3.10 -m pip install --upgrade pip' command.[0m[33m
    [0mRequirement already satisfied: llama-index-embeddings-huggingface in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (0.2.0)
    Requirement already satisfied: sentence-transformers<3.0.0,>=2.6.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-embeddings-huggingface) (2.7.0)
    Requirement already satisfied: huggingface-hub[inference]>=0.19.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-embeddings-huggingface) (0.20.3)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-embeddings-huggingface) (0.10.16.post1)
    Requirement already satisfied: packaging>=20.9 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (23.2)
    Requirement already satisfied: requests in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.31.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (4.9.0)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2024.2.0)
    Requirement already satisfied: filelock in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (3.13.1)
    Requirement already satisfied: pyyaml>=5.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (6.0.1)
    Requirement already satisfied: tqdm>=4.42.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (4.66.1)
    Requirement already satisfied: aiohttp in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (3.9.3)
    Requirement already satisfied: pydantic<3.0,>1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.5.1)
    

    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.6.0)
    Requirement already satisfied: tiktoken>=0.3.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.5.2)
    Requirement already satisfied: SQLAlchemy[asyncio]>=1.4.49 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.0.25)
    Requirement already satisfied: typing-inspect>=0.8.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.9.0)
    Requirement already satisfied: deprecated>=1.2.9.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.2.14)
    Requirement already satisfied: pandas in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.5.1)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.1.13)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.8.1)
    Requirement already satisfied: tenacity<9.0.0,>=8.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (8.2.3)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.0.8)
    Requirement already satisfied: numpy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.23.4)
    Requirement already satisfied: pillow>=9.0.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (10.2.0)
    Requirement already satisfied: dataclasses-json in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.6.4)
    Requirement already satisfied: networkx>=3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.2.1)
    Requirement already satisfied: httpx in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.26.0)
    Requirement already satisfied: openai>=1.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.13.3)
    Requirement already satisfied: scipy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (1.12.0)
    Requirement already satisfied: torch>=1.11.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (2.2.0)
    Requirement already satisfied: transformers<5.0.0,>=4.34.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (4.37.2)
    Requirement already satisfied: scikit-learn in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (1.4.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (6.0.5)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (1.3.1)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (1.4.1)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (1.9.4)
    Requirement already satisfied: attrs>=17.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (22.1.0)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (4.0.3)
    Requirement already satisfied: wrapt<2,>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from deprecated>=1.2.9.3->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.16.0)
    Requirement already satisfied: certifi in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2024.2.2)
    Requirement already satisfied: idna in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.4)
    Requirement already satisfied: httpcore==1.* in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.0.2)
    Requirement already satisfied: anyio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.7.1)
    Requirement already satisfied: sniffio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.3.0)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.14.0)
    Requirement already satisfied: click in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (8.1.7)
    Requirement already satisfied: joblib in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.3.2)
    Requirement already satisfied: regex>=2021.8.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2023.12.25)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.9.0)
    Requirement already satisfied: pydantic-core==2.14.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic<3.0,>1.1->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.14.3)
    Requirement already satisfied: annotated-types>=0.4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic<3.0,>1.1->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (0.6.0)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.2.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (3.3.2)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.0.1)
    Requirement already satisfied: sympy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (1.12)
    Requirement already satisfied: jinja2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (3.1.2)
    

    Requirement already satisfied: safetensors>=0.4.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.34.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (0.4.2)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from transformers<5.0.0,>=4.34.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (0.15.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.20.2)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2022.5)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from scikit-learn->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (3.2.0)
    Requirement already satisfied: exceptiongroup in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from anyio->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.2.0)
    Requirement already satisfied: six>=1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from jinja2->torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (2.1.1)
    Requirement already satisfied: mpmath>=0.19 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from sympy->torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface) (1.3.0)
    [33mWARNING: You are using pip version 22.0.4; however, version 24.0 is available.
    You should consider upgrading via the '/Users/daniel/.pyenv/versions/3.10.3/bin/python3.10 -m pip install --upgrade pip' command.[0m[33m
    [0mRequirement already satisfied: llama-parse in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (0.3.7)
    Requirement already satisfied: llama-index-core>=0.10.7 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-parse) (0.10.16.post1)
    Requirement already satisfied: SQLAlchemy[asyncio]>=1.4.49 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (2.0.25)
    Requirement already satisfied: httpx in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (0.26.0)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (1.6.0)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (3.9.3)
    Requirement already satisfied: typing-inspect>=0.8.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (0.9.0)
    Requirement already satisfied: tenacity<9.0.0,>=8.2.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (8.2.3)
    Requirement already satisfied: networkx>=3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (3.2.1)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (1.0.8)
    Requirement already satisfied: pandas in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (1.5.1)
    Requirement already satisfied: openai>=1.1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (1.13.3)
    Requirement already satisfied: pillow>=9.0.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (10.2.0)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (0.1.13)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (3.8.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (2024.2.0)
    Requirement already satisfied: PyYAML>=6.0.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (6.0.1)
    Requirement already satisfied: deprecated>=1.2.9.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (1.2.14)
    Requirement already satisfied: tiktoken>=0.3.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (0.5.2)
    Requirement already satisfied: typing-extensions>=4.5.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (4.9.0)
    Requirement already satisfied: numpy in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (1.23.4)
    Requirement already satisfied: requests>=2.31.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (2.31.0)
    Requirement already satisfied: dataclasses-json in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (0.6.4)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llama-index-core>=0.10.7->llama-parse) (4.66.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core>=0.10.7->llama-parse) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core>=0.10.7->llama-parse) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core>=0.10.7->llama-parse) (4.0.3)
    Requirement already satisfied: aiosignal>=1.1.2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core>=0.10.7->llama-parse) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core>=0.10.7->llama-parse) (22.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core>=0.10.7->llama-parse) (1.4.1)
    Requirement already satisfied: wrapt<2,>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from deprecated>=1.2.9.3->llama-index-core>=0.10.7->llama-parse) (1.16.0)
    Requirement already satisfied: pydantic>=1.10 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core>=0.10.7->llama-parse) (2.5.1)
    Requirement already satisfied: idna in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core>=0.10.7->llama-parse) (3.4)
    Requirement already satisfied: anyio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core>=0.10.7->llama-parse) (3.7.1)
    Requirement already satisfied: httpcore==1.* in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core>=0.10.7->llama-parse) (1.0.2)
    Requirement already satisfied: sniffio in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core>=0.10.7->llama-parse) (1.3.0)
    Requirement already satisfied: certifi in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpx->llama-index-core>=0.10.7->llama-parse) (2024.2.2)
    Requirement already satisfied: h11<0.15,>=0.13 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from httpcore==1.*->httpx->llama-index-core>=0.10.7->llama-parse) (0.14.0)
    Requirement already satisfied: regex>=2021.8.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core>=0.10.7->llama-parse) (2023.12.25)
    Requirement already satisfied: joblib in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core>=0.10.7->llama-parse) (1.3.2)
    Requirement already satisfied: click in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core>=0.10.7->llama-parse) (8.1.7)
    Requirement already satisfied: distro<2,>=1.7.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from openai>=1.1.0->llama-index-core>=0.10.7->llama-parse) (1.9.0)
    

    Requirement already satisfied: charset-normalizer<4,>=2 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests>=2.31.0->llama-index-core>=0.10.7->llama-parse) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from requests>=2.31.0->llama-index-core>=0.10.7->llama-parse) (2.2.0)
    Requirement already satisfied: greenlet!=0.4.17 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core>=0.10.7->llama-parse) (3.0.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from typing-inspect>=0.8.0->llama-index-core>=0.10.7->llama-parse) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from dataclasses-json->llama-index-core>=0.10.7->llama-parse) (3.20.2)
    Requirement already satisfied: pytz>=2020.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core>=0.10.7->llama-parse) (2022.5)
    Requirement already satisfied: python-dateutil>=2.8.1 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pandas->llama-index-core>=0.10.7->llama-parse) (2.8.2)
    Requirement already satisfied: exceptiongroup in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from anyio->httpx->llama-index-core>=0.10.7->llama-parse) (1.2.0)
    Requirement already satisfied: packaging>=17.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from marshmallow<4.0.0,>=3.18.0->dataclasses-json->llama-index-core>=0.10.7->llama-parse) (23.2)
    Requirement already satisfied: annotated-types>=0.4.0 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core>=0.10.7->llama-parse) (0.6.0)
    Requirement already satisfied: pydantic-core==2.14.3 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from pydantic>=1.10->llamaindex-py-client<0.2.0,>=0.1.13->llama-index-core>=0.10.7->llama-parse) (2.14.3)
    Requirement already satisfied: six>=1.5 in /Users/daniel/.pyenv/versions/3.10.3/lib/python3.10/site-packages (from python-dateutil>=2.8.1->pandas->llama-index-core>=0.10.7->llama-parse) (1.16.0)
    [33mWARNING: You are using pip version 22.0.4; however, version 24.0 is available.
    You should consider upgrading via the '/Users/daniel/.pyenv/versions/3.10.3/bin/python3.10 -m pip install --upgrade pip' command.[0m[33m
    [0m


```python
import nest_asyncio

nest_asyncio.apply()
```

### Setup LLM using Groq

To use [Groq](https://groq.com), you need to make sure that `GROQ_API_KEY` is specified as an environment variable.


```python
import os

os.environ["GROQ_API_KEY"] = "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"
```


```python
from llama_index.llms.groq import Groq

llm = Groq(model="llama3-8b-8192")
llm_70b = Groq(model="llama3-70b-8192")
```

### Setup Embedding Model


```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

### Define Global Settings Configuration

In LlamaIndex, you can define global settings so you don't have to pass the LLM / embedding model objects everywhere.


```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
```

### Download Data

Here you'll download data that's used in section 2 and onwards.

We'll download some articles on Kendrick, Drake, and their beef (as of May 2024).


```python
!mkdir data
!wget "https://www.dropbox.com/scl/fi/t1soxfjdp0v44an6sdymd/drake_kendrick_beef.pdf?rlkey=u9546ymb7fj8lk2v64r6p5r5k&st=wjzzrgil&dl=1" -O data/drake_kendrick_beef.pdf
!wget "https://www.dropbox.com/scl/fi/nts3n64s6kymner2jppd6/drake.pdf?rlkey=hksirpqwzlzqoejn55zemk6ld&st=mohyfyh4&dl=1" -O data/drake.pdf
!wget "https://www.dropbox.com/scl/fi/8ax2vnoebhmy44bes2n1d/kendrick.pdf?rlkey=fhxvn94t5amdqcv9vshifd3hj&st=dxdtytn6&dl=1" -O data/kendrick.pdf
```

    mkdir: data: File exists
    --2024-05-20 09:27:56--  https://www.dropbox.com/scl/fi/t1soxfjdp0v44an6sdymd/drake_kendrick_beef.pdf?rlkey=u9546ymb7fj8lk2v64r6p5r5k&st=wjzzrgil&dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 2620:100:6019:18::a27d:412, 162.125.4.18
    Connecting to www.dropbox.com (www.dropbox.com)|2620:100:6019:18::a27d:412|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com/cd/0/inline/CTQhAFm1iI5gNTeE_NytPzfcLl6Ilp9PSwNsVHJg7h_C2mUfnd6DL__txef3V5PoEV68APiuzt1UaHr4GVFHs-iYtSYqNJ9YT-chZyGn5GTRT837J92mPPDHpPnxibg3FCE/file?dl=1# [following]
    --2024-05-20 09:27:57--  https://uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com/cd/0/inline/CTQhAFm1iI5gNTeE_NytPzfcLl6Ilp9PSwNsVHJg7h_C2mUfnd6DL__txef3V5PoEV68APiuzt1UaHr4GVFHs-iYtSYqNJ9YT-chZyGn5GTRT837J92mPPDHpPnxibg3FCE/file?dl=1
    Resolving uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com (uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com)... 2620:100:6019:15::a27d:40f, 162.125.4.15
    Connecting to uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com (uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com)|2620:100:6019:15::a27d:40f|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /cd/0/inline2/CTTKkMZQK-Fk13zt0Wc04FPhWEZ2Mfy-DhMgx4k3kmgqTZFkhDUieUVZNJ5S9fESwn1XTt68Cm6-T9FuNDFxv0SE7JN8WtpJJaZHbV4EfVkffGctU9aiy7m_xfo8OViwDmMo3PeRerVdwDilsblJLH0Z9_eeVicSjRCQh03eeybgZZr_zzF6ydj5V9evnXEhVp0CmBs-DfNL3s-AbIZ4nYwFLmrufsyw17rSqLDDmbIUQxV349HByliOgJqdZ-C-gH0-MaBSnIa3g88T8RvxAzyrdNpEdJoEvCVqOYdl2JtKleQYxuR4XO4EHxJWTwNj735jMjHf1rQVkRcSx71MYrL-YSkvVYQBhoCUwxJoNIvaeg/file?dl=1 [following]
    --2024-05-20 09:27:58--  https://uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com/cd/0/inline2/CTTKkMZQK-Fk13zt0Wc04FPhWEZ2Mfy-DhMgx4k3kmgqTZFkhDUieUVZNJ5S9fESwn1XTt68Cm6-T9FuNDFxv0SE7JN8WtpJJaZHbV4EfVkffGctU9aiy7m_xfo8OViwDmMo3PeRerVdwDilsblJLH0Z9_eeVicSjRCQh03eeybgZZr_zzF6ydj5V9evnXEhVp0CmBs-DfNL3s-AbIZ4nYwFLmrufsyw17rSqLDDmbIUQxV349HByliOgJqdZ-C-gH0-MaBSnIa3g88T8RvxAzyrdNpEdJoEvCVqOYdl2JtKleQYxuR4XO4EHxJWTwNj735jMjHf1rQVkRcSx71MYrL-YSkvVYQBhoCUwxJoNIvaeg/file?dl=1
    Reusing existing connection to [uc4425830a1d2d4c42bbf6c89b7f.dl.dropboxusercontent.com]:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 49318627 (47M) [application/binary]
    Saving to: ‘data/drake_kendrick_beef.pdf’
    
    data/drake_kendrick 100%[===================>]  47.03M  32.9MB/s    in 1.4s    
    
    2024-05-20 09:28:00 (32.9 MB/s) - ‘data/drake_kendrick_beef.pdf’ saved [49318627/49318627]
    
    --2024-05-20 09:28:00--  https://www.dropbox.com/scl/fi/nts3n64s6kymner2jppd6/drake.pdf?rlkey=hksirpqwzlzqoejn55zemk6ld&st=mohyfyh4&dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 2620:100:6019:18::a27d:412, 162.125.4.18
    Connecting to www.dropbox.com (www.dropbox.com)|2620:100:6019:18::a27d:412|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com/cd/0/inline/CTTKsxu4SC50fGZs5aEVnvyeCyoCcebsEJLbgiKc-zs4xz7qUrHw3KfJmFvC3LCbaD1qeP5FE5Z_irFNBzYG-4Nbr3sR0f4AY7GrHUOtSMzmtVCS1G2okbjCLLOoj8Urdkw/file?dl=1# [following]
    --2024-05-20 09:28:01--  https://uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com/cd/0/inline/CTTKsxu4SC50fGZs5aEVnvyeCyoCcebsEJLbgiKc-zs4xz7qUrHw3KfJmFvC3LCbaD1qeP5FE5Z_irFNBzYG-4Nbr3sR0f4AY7GrHUOtSMzmtVCS1G2okbjCLLOoj8Urdkw/file?dl=1
    Resolving uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com (uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com)... 2620:100:6019:15::a27d:40f, 162.125.4.15
    Connecting to uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com (uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com)|2620:100:6019:15::a27d:40f|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /cd/0/inline2/CTQv1f9QtlDimE_MTAN-OEDn6BGT9UTJ8QjgwkGGhcWJN5O_F7cNTeAlo6ThMraOXNh9P9ENA-IS08GWOU9Pu1cQPyxsjiT8o0_KZRwsjrPam9a_bZ0uydRciFz3i6PRI8EwAAAHD7V-XibNLg9uv5b_-jKxg6SXmIMuN7ZUItSKxKyhfg0YF0UeOp7BgEnjabJIfXTFSD0y4_Kvnl3_isvMbBUZ6os7vOsnjjgN2eLGNHVnfEdbSlBSw1cGsXA1ZRwR3NwF05BIZT-Lsgspw8TPN4updOfgCXsSERWFHDmiKLozDCU3UPWh1QAEVTct9mW3vRHIGQ7i8xr1nO7h8lR_VSMJ-C9Ep40O2rjeEGbKEQ/file?dl=1 [following]
    --2024-05-20 09:28:01--  https://uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com/cd/0/inline2/CTQv1f9QtlDimE_MTAN-OEDn6BGT9UTJ8QjgwkGGhcWJN5O_F7cNTeAlo6ThMraOXNh9P9ENA-IS08GWOU9Pu1cQPyxsjiT8o0_KZRwsjrPam9a_bZ0uydRciFz3i6PRI8EwAAAHD7V-XibNLg9uv5b_-jKxg6SXmIMuN7ZUItSKxKyhfg0YF0UeOp7BgEnjabJIfXTFSD0y4_Kvnl3_isvMbBUZ6os7vOsnjjgN2eLGNHVnfEdbSlBSw1cGsXA1ZRwR3NwF05BIZT-Lsgspw8TPN4updOfgCXsSERWFHDmiKLozDCU3UPWh1QAEVTct9mW3vRHIGQ7i8xr1nO7h8lR_VSMJ-C9Ep40O2rjeEGbKEQ/file?dl=1
    Reusing existing connection to [uc306cc6b72bb0c6b4807adfbf69.dl.dropboxusercontent.com]:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 4590973 (4.4M) [application/binary]
    Saving to: ‘data/drake.pdf’
    
    data/drake.pdf      100%[===================>]   4.38M  12.0MB/s    in 0.4s    
    
    2024-05-20 09:28:02 (12.0 MB/s) - ‘data/drake.pdf’ saved [4590973/4590973]
    
    --2024-05-20 09:28:02--  https://www.dropbox.com/scl/fi/8ax2vnoebhmy44bes2n1d/kendrick.pdf?rlkey=fhxvn94t5amdqcv9vshifd3hj&st=dxdtytn6&dl=1
    Resolving www.dropbox.com (www.dropbox.com)... 2620:100:6019:18::a27d:412, 162.125.4.18
    Connecting to www.dropbox.com (www.dropbox.com)|2620:100:6019:18::a27d:412|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com/cd/0/inline/CTS6obqeEm8Mzu1a_hWd2GmLrYndc7ctcFK1-6-yM2PPXFyvOsoe9OFDf2ZbCA-mE-19OCycTm4OD8D47idzH09Lf-M501waiDDcEDejhhFjgJr5wABuD4FV4kKtLgecZhI/file?dl=1# [following]
    --2024-05-20 09:28:03--  https://uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com/cd/0/inline/CTS6obqeEm8Mzu1a_hWd2GmLrYndc7ctcFK1-6-yM2PPXFyvOsoe9OFDf2ZbCA-mE-19OCycTm4OD8D47idzH09Lf-M501waiDDcEDejhhFjgJr5wABuD4FV4kKtLgecZhI/file?dl=1
    Resolving uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com (uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com)... 2620:100:6019:15::a27d:40f, 162.125.4.15
    Connecting to uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com (uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com)|2620:100:6019:15::a27d:40f|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: /cd/0/inline2/CTQNM0rmZNvzX5Lwg1iXBmqIz4EJ2ZhyZOITdANOekmgSe03MihquuCWfGxT8LH24oZNn9uwX1HUqaRF2BHUzBsQEiTEvONnVsh7d6pcpd0O0TV-_vyKIQn26qk4cCTpHEy-GcRIKa1opOd-degk9giPIli7-IJsS0WL6EIchoA74Homi43Qmo-Tarf8lF70O9b7eN8AjsjQZ6PFJl8EcRy0s_ox30TH93GvN3NQh_2lVmD3n8f1xPSrLRcyIFyzWJN0GZzTeYrAX-bAPF8IbW_2laURmBVYT1fg4vHdwH0wMFfJR7WDfY5XRWYyRVia6m6VwTVuWW-fddR4jW9HSXvBX8YjnjrwAwNum_jnbOpJTg/file?dl=1 [following]
    --2024-05-20 09:28:03--  https://uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com/cd/0/inline2/CTQNM0rmZNvzX5Lwg1iXBmqIz4EJ2ZhyZOITdANOekmgSe03MihquuCWfGxT8LH24oZNn9uwX1HUqaRF2BHUzBsQEiTEvONnVsh7d6pcpd0O0TV-_vyKIQn26qk4cCTpHEy-GcRIKa1opOd-degk9giPIli7-IJsS0WL6EIchoA74Homi43Qmo-Tarf8lF70O9b7eN8AjsjQZ6PFJl8EcRy0s_ox30TH93GvN3NQh_2lVmD3n8f1xPSrLRcyIFyzWJN0GZzTeYrAX-bAPF8IbW_2laURmBVYT1fg4vHdwH0wMFfJR7WDfY5XRWYyRVia6m6VwTVuWW-fddR4jW9HSXvBX8YjnjrwAwNum_jnbOpJTg/file?dl=1
    Reusing existing connection to [uc3ad47fc720b85fdd36566e9669.dl.dropboxusercontent.com]:443.
    HTTP request sent, awaiting response... 200 OK
    Length: 5595364 (5.3M) [application/binary]
    Saving to: ‘data/kendrick.pdf’
    
    data/kendrick.pdf   100%[===================>]   5.34M  11.4MB/s    in 0.5s    
    
    2024-05-20 09:28:04 (11.4 MB/s) - ‘data/kendrick.pdf’ saved [5595364/5595364]
    
    

### Load Data

We load data using LlamaParse by default, but you can also choose to opt for our free pypdf reader (in SimpleDirectoryReader by default) if you don't have an account! 

1. LlamaParse: Signup for an account here: cloud.llamaindex.ai. You get 1k free pages a day, and paid plan is 7k free pages + 0.3c per additional page. LlamaParse is a good option if you want to parse complex documents, like PDFs with charts, tables, and more. 

2. Default PDF Parser (In `SimpleDirectoryReader`). If you don't want to signup for an account / use a PDF service, just use the default PyPDF reader bundled in our file loader. It's a good choice for getting started!


```python
# Uncomment this code if you want to use LlamaParse
# from llama_parse import LlamaParse

# docs_kendrick = LlamaParse(result_type="text").load_data("./data/kendrick.pdf")
# docs_drake = LlamaParse(result_type="text").load_data("./data/drake.pdf")
# docs_both = LlamaParse(result_type="text").load_data(
#     "./data/drake_kendrick_beef.pdf"
# )

# Uncomment this code if you want to use SimpleDirectoryReader / default PDF Parser
from llama_index.core import SimpleDirectoryReader

docs_kendrick = SimpleDirectoryReader(input_files=["data/kendrick.pdf"]).load_data()
docs_drake = SimpleDirectoryReader(input_files=["data/drake.pdf"]).load_data()
docs_both = SimpleDirectoryReader(input_files=["data/drake_kendrick_beef.pdf"]).load_data()
```

## 1. Basic Completion and Chat

### Call complete with a prompt


```python
response = llm.complete("do you like drake or kendrick better?")

print(response)
```

    I'm just an AI, I don't have personal preferences or opinions, nor do I have the capacity to enjoy or dislike music. I can provide information and insights about different artists and their work, but I don't have personal feelings or emotions.
    
    However, I can tell you that both Drake and Kendrick Lamar are highly acclaimed and influential artists in the music industry. They have both received widespread critical acclaim and have won numerous awards for their work.
    
    Drake is known for his introspective and emotive lyrics, as well as his ability to blend different genres such as hip-hop, R&B, and pop. He has released several successful albums, including "Take Care" and "Views".
    
    Kendrick Lamar is known for his socially conscious and thought-provoking lyrics, as well as his unique blend of jazz, funk, and hip-hop. He has released several critically acclaimed albums, including "Good Kid, M.A.A.D City" and "To Pimp a Butterfly".
    
    Ultimately, whether you prefer Drake or Kendrick Lamar depends on your personal taste in music and the type of music you enjoy.
    


```python
stream_response = llm.stream_complete(
    "you're a drake fan. tell me why you like drake more than kendrick"
)

for t in stream_response:
    print(t.delta, end="")
```

    Man, I'm a die-hard Drake fan, and I gotta say, I love the 6 God for a lot of reasons. Now, I know some people might say Kendrick is the king of hip-hop, and I respect that, but for me, Drake brings something unique to the table that sets him apart.
    
    First of all, Drake's lyrics are so relatable. He's not just rapping about gangsta life or street cred; he's talking about real-life struggles, relationships, and emotions. His songs are like a diary entry, you know? He's sharing his thoughts, feelings, and experiences in a way that resonates with people from all walks of life. I mean, who hasn't been through a breakup or felt like they're stuck in a rut? Drake's music speaks to that.
    
    And let's not forget his storytelling ability. The man can paint a picture with his words. He's got this effortless flow, and his rhymes are like a puzzle – intricate, clever, and always surprising. He's got this ability to weave together complex narratives that keep you engaged from start to finish.
    
    Now, I know some people might say Kendrick's lyrics are more socially conscious, and that's true. But for me, Drake's music is more personal, more intimate. He's not just preaching to the choir; he's sharing his own struggles, fears, and doubts. That vulnerability is what makes his music so powerful.
    
    And let's not forget his production. Drake's got an ear for beats, man. He's always pushing the boundaries of what hip-hop can sound like. From "Marvin's Room" to "God's Plan," he's consistently delivered some of the most innovative, catchy, and emotive production in the game.
    
    Now, I'm not saying Kendrick isn't a genius – he is. But for me, Drake's music is more relatable, more personal, and more innovative. He's the perfect blend of street cred and pop sensibility. And let's be real, his flow is unmatched. The man can spit bars like nobody's business.
    
    So, yeah, I'm a Drake fan through and through. I love his music, his message, and his artistry. He's the real MVP, and I'm not ashamed to say it.

### Call chat with a list of messages


```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are Kendrick."),
    ChatMessage(role="user", content="Write a verse."),
]
response = llm.chat(messages)
```


```python
print(response)
```

    assistant: "I'm the king of the game, no debate
    My rhymes are fire, can't nobody relate
    I'm on a mission, to spread the message wide
    My flow's on a hundred, ain't nobody gonna divide"
    

## 2. Basic RAG (Vector Search, Summarization)

### Basic RAG (Vector Search)


```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(docs_both)
query_engine = index.as_query_engine(similarity_top_k=3)
```


```python
response = query_engine.query("Tell me about family matters")
```


```python
print(str(response))
```

    Drake's diss track "Family Matters" is essentially three songs in one, on three different beats. The track is a seven-and-a-half-minute diss track with an accompanying video.
    

### Basic RAG (Summarization)


```python
from llama_index.core import SummaryIndex

summary_index = SummaryIndex.from_documents(docs_both)
summary_engine = summary_index.as_query_engine()
```


```python
response = summary_engine.query(
    "Given your assessment of this article, who won the beef?"
)
```


```python
print(str(response))
```

    It's difficult to declare a clear winner in this beef, as both parties have delivered strong diss tracks and have been engaging in a back-and-forth exchange.
    

## 3. Advanced RAG (Routing)

### Build a Router that can choose whether to do vector search or summarization


```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

vector_tool = QueryEngineTool(
    index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_search",
        description="Useful for searching for specific facts.",
    ),
)

summary_tool = QueryEngineTool(
    index.as_query_engine(response_mode="tree_summarize"),
    metadata=ToolMetadata(
        name="summary",
        description="Useful for summarizing an entire document.",
    ),
)
```


```python
from llama_index.core.query_engine import RouterQueryEngine

query_engine = RouterQueryEngine.from_defaults(
    [vector_tool, summary_tool], select_multi=False, llm=llm_70b
)

response = query_engine.query(
    "Tell me about the song meet the grahams - why is it significant"
)
```


```python
print(response)
```

    "Meet the Grahams" is significant because it marks a turning point in the beef between Kendrick Lamar and Drake. The song is notable for its lighthearted and humorous tone, with Kendrick cracking jokes and making playful jabs at Drake. The track also showcases Kendrick's ability to poke fun at himself and not take himself too seriously.
    

## 4. Text-to-SQL 

Here, we download and use a sample SQLite database with 11 tables, with various info about music, playlists, and customers. We will limit to a select few tables for this test.


```python
!wget "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip" -O "./data/chinook.zip"
!unzip "./data/chinook.zip"
```

    --2024-05-20 09:31:46--  https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
    Resolving www.sqlitetutorial.net (www.sqlitetutorial.net)... 2606:4700:3037::6815:1e8d, 2606:4700:3037::ac43:acfa, 172.67.172.250, ...
    Connecting to www.sqlitetutorial.net (www.sqlitetutorial.net)|2606:4700:3037::6815:1e8d|:443... connected.
    HTTP request sent, awaiting response... 

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    

    200 OK
    Length: 305596 (298K) [application/zip]
    Saving to: ‘./data/chinook.zip’
    
    ./data/chinook.zip  100%[===================>] 298.43K  --.-KB/s    in 0.07s   
    
    2024-05-20 09:31:46 (4.30 MB/s) - ‘./data/chinook.zip’ saved [305596/305596]
    
    Archive:  ./data/chinook.zip
    replace chinook.db? [y]es, [n]o, [A]ll, [N]one, [r]ename: 

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    

    ^C



```python
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite:///chinook.db")
```


```python
from llama_index.core import SQLDatabase

sql_database = SQLDatabase(engine)
```


```python
from llama_index.core.indices.struct_store import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    llm=llm_70b,
)
```


```python
response = query_engine.query("What are some albums?")

print(response)
```

    Here are some albums: For Those About To Rock We Salute You, Balls to the Wall, Restless and Wild, Let There Be Rock, Big Ones, Jagged Little Pill, Facelift, Warner 25 Anos, Plays Metallica By Four Cellos, and Audioslave.
    


```python
response = query_engine.query("What are some artists? Limit it to 5.")

print(response)
```

    Here are 5 artists: AC/DC, Accept, Aerosmith, Alanis Morissette, and Alice In Chains.
    

This last query should be a more complex join


```python
response = query_engine.query(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)

print(response)
```

    Here are three tracks from the legendary Australian rock band AC/DC: "For Those About To Rock (We Salute You)", "Put The Finger On You", and "Let's Get It Up".
    


```python
print(response.metadata["sql_query"])
```

    SELECT tracks.Name FROM tracks INNER JOIN albums ON tracks.AlbumId = albums.AlbumId INNER JOIN artists ON albums.ArtistId = artists.ArtistId WHERE artists.Name = 'AC/DC' LIMIT 3;
    

## 5. Structured Data Extraction

An important use case for function calling is extracting structured objects. LlamaIndex provides an intuitive interface for this through `structured_predict` - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.

**NOTE**: Since there's no native function calling support with Llama3, the structured extraction is performed by prompting the LLM + output parsing.


```python
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str


llm = Groq(model="llama3-8b-8192", pydantic_program_mode="llm")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)
```


```python
restaurant_obj = llm.structured_predict(
    Restaurant, prompt_tmpl, city_name="Miami"
)
print(restaurant_obj)
```

    name='Café Havana' city='Miami' cuisine='Cuban'
    

## 6. Adding Chat History to RAG (Chat Engine)

In this section we create a stateful chatbot from a RAG pipeline, with our chat engine abstraction.

Unlike a stateless query engine, the chat engine maintains conversation history (through a memory module like buffer memory). It performs retrieval given a condensed question, and feeds the condensed question + context + chat history into the final LLM prompt.

Related resource: https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/


```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

chat_engine = CondensePlusContextChatEngine.from_defaults(
    index.as_retriever(),
    memory=memory,
    llm=llm,
    context_prompt=(
        "You are a chatbot, able to have normal interactions, as well as talk"
        " about the Kendrick and Drake beef."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=True,
)
```


```python
response = chat_engine.chat(
    "Tell me about the songs Drake released in the beef."
)
print(str(response))
```

    Condensed question: Tell me about the songs Drake released in the beef.
    Context: page_label: 31
    file_path: data/drake_kendrick_beef.pdf
    
    Culture
    Shaboo zey’s Cowboy Carter Features Were Only the Be ginning
    By Heven Haile
    Sign up for Manual, our new flagship newsletter
    Useful advice on style, health, and more, four days a week.
    5/10/24, 10:08 PM The Kendrick Lamar/Drake Beef, Explained | GQ
    https://www.gq.com/story/the-kendrick-lamar-drake-beef-explained 31/34
    
    page_label: 18
    file_path: data/drake_kendrick_beef.pdf
    
    Kurrco
    @Kurrco·Follow
    KENDRICK LAMAR
    6 16 IN LA
    (DRAKE DISS)
    OUT NOW 
    This video has been deleted.
    6 08 AM · May 3, 2024
    59.3K Reply Copy link
    Read 1.3K replies
    After all this talk about “the clock,” who among us expected Kendrick to follow up his
    own titanic diss track with another missile just three days later? Friday morning he
    released “6:16 in LA,” with its title of course being a nod to Drake's series of time-stamp-
    Sign up for Manual, our new flagship newsletter
    Useful advice on style, health, and more, four days a week.
    5/10/24, 10:08 PM The Kendrick Lamar/Drake Beef, Explained | GQ
    https://www.gq.com/story/the-kendrick-lamar-drake-beef-explained 18/34
    The infamous Drake-Kendrick beef! According to the context, Drake didn't release any songs directly addressing the beef. However, Kendrick Lamar did release a few tracks that were perceived as diss tracks aimed at Drake.
    
    One of the notable tracks is "King Kunta" from Kendrick's album "To Pimp a Butterfly" (2015). Although not directly aimed at Drake, some interpreted the lyrics as a subtle jab at the Canadian rapper.
    
    Later, in 2024, Kendrick released "6:16 in LA", which was seen as a response to Drake's "The Clock" (2024). However, Drake didn't release any direct responses to Kendrick's diss tracks.
    
    Would you like to know more about the beef or the songs involved?
    


```python
response = chat_engine.chat("What about Kendrick?")
print(str(response))
```

    Condensed question: What do you want to know about Kendrick Lamar's involvement in the Drake beef?
    Context: page_label: 17
    file_path: data/drake_kendrick_beef.pdf
    
    Melly is, of course, the Florida rapper whose rising career came to a screeching halt
    thanks to a still ongoing murder trial accusing Melly of the premeditated murders of two
    YNW associates—ostensibly, two close friends. (Second best line: using Haley Joel
    Osment's IMDb for a two-for-one A.I. and ghostwriters reference.)
    With lines referencing Puff Daddy notoriously slapping Drake and calling out Drake's
    right-hand enforcer Chubbs by name, Kendrick's threatening to “take it there,” but for
    now it remains a fun war of words and one that doesn't seem likely to end anytime soon,
    much less in an anticlimax like the Drake-Pusha T beef. Drake can only have been
    desperate for Kendrick to respond because he has a fully loaded clip waiting to shoot,
    and Kendrick for his part here, promises “headshots all year, you better walk around like
    Daft Punk.” Summer's heating up.
    May 3: K endrick g oes back-to-back with “6:16 in L A”
    Sign up for Manual, our new flagship newsletter
    Useful advice on style, health, and more, four days a week.
    5/10/24, 10:08 PM The Kendrick Lamar/Drake Beef, Explained | GQ
    https://www.gq.com/story/the-kendrick-lamar-drake-beef-explained 17/34
    
    page_label: 1
    file_path: data/drake_kendrick_beef.pdf
    
    Culture
    The K endrick L amar /Drake Bee f, ExplainedChrist opher P olk/Getty Ima ges
    Sign up for Manual, our new flagship newsletter
    Useful advice on style, health, and more, four days a week.Email address
    SIGN ME UP
    NO THANKS
    5/10/24, 10:08 PM The Kendrick Lamar/Drake Beef, Explained | GQ
    https://www.gq.com/story/the-kendrick-lamar-drake-beef-explained 1/34
    Kendrick Lamar! According to the context, Kendrick Lamar did release some tracks that were perceived as diss tracks aimed at Drake. One notable example is "The Heart Part 4" (2017), which contains lyrics that some interpreted as a response to Drake.
    
    Additionally, Kendrick released "Humble" (2017) which some saw as a diss track aimed at Drake. The lyrics in "Humble" contain lines that some interpreted as a reference to Drake's lyrics in his song "Glow" (2016).
    
    Kendrick also released "King Kunta" (2015) which, although not directly aimed at Drake, some interpreted as a subtle jab at the Canadian rapper.
    
    Would you like to know more about the beef or the songs involved?
    

## 7. Agents

Here we build agents with Llama 3. We perform RAG over simple functions as well as the documents above.

### Agents And Tools


```python
from llama_index.llms.groq import Groq

llm = Groq(model="llama3-8b-8192")
llm_70b = Groq(model="llama3-70b-8192")
```


```python
import json
from typing import Sequence, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.agent.openai import OpenAIAgent

import nest_asyncio

nest_asyncio.apply()
```

### Define Tools


```python
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)
llm_70b.is_function_calling_model = True
```

### ReAct Agent


```python
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm_70b,
    verbose=True,
)
```

### Querying


```python
response = agent.chat("What is (121 + 2) * 5?")
print(str(response))
```

    Added user message to memory: What is (121 + 2) * 5?
    === Calling Function ===
    Calling function: add with args: {"a":121,"b":2}
    Got output: 123
    ========================
    
    === Calling Function ===
    Calling function: multiply with args: {"a":123,"b":5}
    Got output: 615
    ========================
    
    The answer is 615.
    

### ReAct Agent With RAG QueryEngine Tools


```python
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
```

### Create ReAct Agent using RAG QueryEngine Tools

This may take 4 minutes to run:


```python
drake_index = VectorStoreIndex.from_documents(docs_drake)
drake_query_engine = drake_index.as_query_engine(similarity_top_k=3)

kendrick_index = VectorStoreIndex.from_documents(docs_kendrick)
kendrick_query_engine = kendrick_index.as_query_engine(similarity_top_k=3)
```


```python
drake_tool = QueryEngineTool(
    drake_index.as_query_engine(),
    metadata=ToolMetadata(
        name="drake_search",
        description="Useful for searching over Drake's life.",
    ),
)

kendrick_tool = QueryEngineTool(
    kendrick_index.as_query_engine(),
    metadata=ToolMetadata(
        name="kendrick_search",
        description="Useful for searching over Kendrick's life.",
    ),
)

query_engine_tools = [drake_tool, kendrick_tool]
```


```python
agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm_70b,
    verbose=True,
)
```

### Querying


```python
response = agent.chat("Tell me about how Kendrick and Drake grew up")
print(str(response))
```

    [1;3;38;5;200mThought: I need to use a tool to help me answer the question.
    Action: kendrick_search
    Action Input: {'input': "Kendrick Lamar's childhood"}
    [0m[1;3;34mObservation: Kendrick Lamar was born on June 17, 1987, in Compton, California. His parents, Kenneth "Kenny" Duckworth and Paul Oliver, relocated to Compton in 1984 due to his father's affiliation with the Gangster Disciples. Lamar was named after singer-songwriter Eddie Kendricks of the Temptations. He was an only child until the age of seven and was described as a loner by his mother. Eventually, his parents had his two younger brothers and younger sister, businesswoman Kayla Sawyer (née Duckworth).
    [0m[1;3;38;5;200mThought: I need to use a tool to help me answer the question.
    Action: drake_search
    Action Input: {'input': "Drake's childhood"}
    [0m[1;3;34mObservation: Drake was raised in two neighborhoods. He lived on Weston Road in Toronto's working-class west end until grade six and attended Weston Memorial Junior Public School until grade four.
    [0m[1;3;38;5;200mThought: I need to use a tool to help me answer the question.
    Action: drake_search
    Action Input: {'input': "Drake's family and early life"}
    [0m[1;3;34mObservation: Drake was raised in two neighborhoods. He lived on Weston Road in Toronto's working-class west end until grade six and attended Weston Memorial Junior Public School until grade four. He was a promising right winger in minor hockey with the Weston Red Wings.
    [0m[1;3;38;5;200mThought: I need to use a tool to help me answer the question.
    Action: drake_search
    Action Input: {'input': "Drake's parents and family background"}
    [0m[1;3;34mObservation: Drake's parents are Sandi Graham and Dennis Graham. His mother, Sandi Graham, is a Jewish Canadian and his father, Dennis Graham, is an African American from Memphis, Tennessee.
    [0m[1;3;38;5;200mThought: I have enough information to answer the question.
    Answer: Kendrick Lamar grew up in Compton, California, with his parents and siblings, while Drake grew up in Toronto, Canada, with his Jewish-Canadian mother and African-American father, moving between two neighborhoods and playing minor hockey.
    [0mKendrick Lamar grew up in Compton, California, with his parents and siblings, while Drake grew up in Toronto, Canada, with his Jewish-Canadian mother and African-American father, moving between two neighborhoods and playing minor hockey.
    


```python

```
