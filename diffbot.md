# Diffbot

>[Diffbot](https://docs.diffbot.com/docs/getting-started-with-diffbot) is a suite of ML-based products that make it easy to structure web data.

>Diffbot's [Extract API](https://docs.diffbot.com/reference/extract-introduction) is a service that structures and normalizes data from web pages.

>Unlike traditional web scraping tools, `Diffbot Extract` doesn't require any rules to read the content on a page. It uses a computer vision model to classify a page into one of 20 possible types, and then transforms raw HTML markup into JSON. The resulting structured JSON follows a consistent [type-based ontology](https://docs.diffbot.com/docs/ontology), which makes it easy to extract data from multiple different web sources with the same schema.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/integrations/document_loaders/diffbot.ipynb)


## Overview
This guide covers how to extract data from a list of URLs using the [Diffbot Extract API](https://www.diffbot.com/products/extract/) into structured JSON that we can use downstream.

## Setting up

Start by installing the required packages.


```python
%pip install --upgrade --quiet langchain-community
```

Diffbot's Extract API requires an API token. Follow these instructions to [get a free API token](/docs/integrations/providers/diffbot#installation-and-setup) and then set an environment variable.


```python
%env DIFFBOT_API_TOKEN REPLACE_WITH_YOUR_TOKEN
```

## Using the Document Loader

Import the DiffbotLoader module and instantiate it with a list of URLs and your Diffbot token.


```python
import os

from langchain_community.document_loaders import DiffbotLoader

urls = [
    "https://python.langchain.com/",
]

loader = DiffbotLoader(urls=urls, api_token=os.environ.get("DIFFBOT_API_TOKEN"))
```

With the `.load()` method, you can see the documents loaded


```python
loader.load()
```




    [Document(page_content="LangChain is a framework for developing applications powered by large language models (LLMs).\nLangChain simplifies every stage of the LLM application lifecycle:\nDevelopment: Build your applications using LangChain's open-source building blocks and components. Hit the ground running using third-party integrations and Templates.\nProductionization: Use LangSmith to inspect, monitor and evaluate your chains, so that you can continuously optimize and deploy with confidence.\nDeployment: Turn any chain into an API with LangServe.\nlangchain-core: Base abstractions and LangChain Expression Language.\nlangchain-community: Third party integrations.\nPartner packages (e.g. langchain-openai, langchain-anthropic, etc.): Some integrations have been further split into their own lightweight packages that only depend on langchain-core.\nlangchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.\nlanggraph: Build robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph.\nlangserve: Deploy LangChain chains as REST APIs.\nThe broader ecosystem includes:\nLangSmith: A developer platform that lets you debug, test, evaluate, and monitor LLM applications and seamlessly integrates with LangChain.\nGet started\nWe recommend following our Quickstart guide to familiarize yourself with the framework by building your first LangChain application.\nSee here for instructions on how to install LangChain, set up your environment, and start building.\nnote\nThese docs focus on the Python LangChain library. Head here for docs on the JavaScript LangChain library.\nUse cases\nIf you're looking to build something specific or are more of a hands-on learner, check out our use-cases. They're walkthroughs and techniques for common end-to-end tasks, such as:\nQuestion answering with RAG\nExtracting structured output\nChatbots\nand more!\nExpression Language\nLangChain Expression Language (LCEL) is the foundation of many of LangChain's components, and is a declarative way to compose chains. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains.\nGet started: LCEL and its benefits\nRunnable interface: The standard interface for LCEL objects\nPrimitives: More on the primitives LCEL includes\nand more!\nEcosystem\n🦜🛠️ LangSmith\nTrace and evaluate your language model applications and intelligent agents to help you move from prototype to production.\n🦜🕸️ LangGraph\nBuild stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain primitives.\n🦜🏓 LangServe\nDeploy LangChain runnables and chains as REST APIs.\nSecurity\nRead up on our Security best practices to make sure you're developing safely with LangChain.\nAdditional resources\nComponents\nLangChain provides standard, extendable interfaces and integrations for many different components, including:\nIntegrations\nLangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it. Check out our growing list of integrations.\nGuides\nBest practices for developing with LangChain.\nAPI reference\nHead to the reference section for full documentation of all classes and methods in the LangChain and LangChain Experimental Python packages.\nContributing\nCheck out the developer's guide for guidelines on contributing and help getting your dev environment set up.\nHelp us out by providing feedback on this documentation page:", metadata={'source': 'https://python.langchain.com/'})]



## Transform Extracted Text to a Graph Document

Structured page content can be further processed with `DiffbotGraphTransformer` to extract entities and relationships into a graph.


```python
%pip install --upgrade --quiet langchain-experimental
```


```python
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer

diffbot_nlp = DiffbotGraphTransformer(
    diffbot_api_key=os.environ.get("DIFFBOT_API_TOKEN")
)
graph_documents = diffbot_nlp.convert_to_graph_documents(loader.load())
```

To continue loading the data into a Knowledge Graph, follow the [`DiffbotGraphTransformer` guide](/docs/integrations/graphs/diffbot/#loading-the-data-into-a-knowledge-graph).
