# Combine agents and vector stores

This notebook covers how to combine agents and vector stores. The use case for this is that you've ingested your data into a vector store and want to interact with it in an agentic manner.

The recommended method for doing so is to create a `RetrievalQA` and then use that as a tool in the overall agent. Let's take a look at doing this below. You can do this with multiple different vector DBs, and use the agent as a way to route between them. There are two different ways of doing this - you can either let the agent use the vector stores as normal tools, or you can set `return_direct=True` to really just use the agent as a router.

## Create the vector store


```python
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

llm = OpenAI(temperature=0)
```


```python
from pathlib import Path

relevant_parts = []
for p in Path(".").absolute().parts:
    relevant_parts.append(p)
    if relevant_parts[-3:] == ["langchain", "docs", "modules"]:
        break
doc_path = str(Path(*relevant_parts) / "state_of_the_union.txt")
```


```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader(doc_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
```

    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
    


```python
state_of_union = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
)
```


```python
from langchain_community.document_loaders import WebBaseLoader
```


```python
loader = WebBaseLoader("https://beta.ruff.rs/docs/faq/")
```


```python
docs = loader.load()
ruff_texts = text_splitter.split_documents(docs)
ruff_db = Chroma.from_documents(ruff_texts, embeddings, collection_name="ruff")
ruff = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=ruff_db.as_retriever()
)
```

    Running Chroma using direct local API.
    Using DuckDB in-memory for database. Data will be transient.
    


```python

```

## Create the Agent


```python
# Import things that are needed generically
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_openai import OpenAI
```


```python
tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
    ),
]
```


```python
# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```


```python
agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to find out what Biden said about Ketanji Brown Jackson in the State of the Union address.
    Action: State of Union QA System
    Action Input: What did Biden say about Ketanji Brown Jackson in the State of the Union address?[0m
    Observation: [36;1m[1;3m Biden said that Jackson is one of the nation's top legal minds and that she will continue Justice Breyer's legacy of excellence.[0m
    Thought:[32;1m[1;3m I now know the final answer
    Final Answer: Biden said that Jackson is one of the nation's top legal minds and that she will continue Justice Breyer's legacy of excellence.[0m
    
    [1m> Finished chain.[0m
    




    "Biden said that Jackson is one of the nation's top legal minds and that she will continue Justice Breyer's legacy of excellence."




```python
agent.run("Why use ruff over flake8?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to find out the advantages of using ruff over flake8
    Action: Ruff QA System
    Action Input: What are the advantages of using ruff over flake8?[0m
    Observation: [33;1m[1;3m Ruff can be used as a drop-in replacement for Flake8 when used (1) without or with a small number of plugins, (2) alongside Black, and (3) on Python 3 code. It also re-implements some of the most popular Flake8 plugins and related code quality tools natively, including isort, yesqa, eradicate, and most of the rules implemented in pyupgrade. Ruff also supports automatically fixing its own lint violations, which Flake8 does not.[0m
    Thought:[32;1m[1;3m I now know the final answer
    Final Answer: Ruff can be used as a drop-in replacement for Flake8 when used (1) without or with a small number of plugins, (2) alongside Black, and (3) on Python 3 code. It also re-implements some of the most popular Flake8 plugins and related code quality tools natively, including isort, yesqa, eradicate, and most of the rules implemented in pyupgrade. Ruff also supports automatically fixing its own lint violations, which Flake8 does not.[0m
    
    [1m> Finished chain.[0m
    




    'Ruff can be used as a drop-in replacement for Flake8 when used (1) without or with a small number of plugins, (2) alongside Black, and (3) on Python 3 code. It also re-implements some of the most popular Flake8 plugins and related code quality tools natively, including isort, yesqa, eradicate, and most of the rules implemented in pyupgrade. Ruff also supports automatically fixing its own lint violations, which Flake8 does not.'



## Use the Agent solely as a router

You can also set `return_direct=True` if you intend to use the agent as a router and just want to directly return the result of the RetrievalQAChain.

Notice that in the above examples the agent did some extra work after querying the RetrievalQAChain. You can avoid that and just return the result directly.


```python
tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question.",
        return_direct=True,
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question.",
        return_direct=True,
    ),
]
```


```python
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```


```python
agent.run(
    "What did biden say about ketanji brown jackson in the state of the union address?"
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to find out what Biden said about Ketanji Brown Jackson in the State of the Union address.
    Action: State of Union QA System
    Action Input: What did Biden say about Ketanji Brown Jackson in the State of the Union address?[0m
    Observation: [36;1m[1;3m Biden said that Jackson is one of the nation's top legal minds and that she will continue Justice Breyer's legacy of excellence.[0m
    [32;1m[1;3m[0m
    
    [1m> Finished chain.[0m
    




    " Biden said that Jackson is one of the nation's top legal minds and that she will continue Justice Breyer's legacy of excellence."




```python
agent.run("Why use ruff over flake8?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to find out the advantages of using ruff over flake8
    Action: Ruff QA System
    Action Input: What are the advantages of using ruff over flake8?[0m
    Observation: [33;1m[1;3m Ruff can be used as a drop-in replacement for Flake8 when used (1) without or with a small number of plugins, (2) alongside Black, and (3) on Python 3 code. It also re-implements some of the most popular Flake8 plugins and related code quality tools natively, including isort, yesqa, eradicate, and most of the rules implemented in pyupgrade. Ruff also supports automatically fixing its own lint violations, which Flake8 does not.[0m
    [32;1m[1;3m[0m
    
    [1m> Finished chain.[0m
    




    ' Ruff can be used as a drop-in replacement for Flake8 when used (1) without or with a small number of plugins, (2) alongside Black, and (3) on Python 3 code. It also re-implements some of the most popular Flake8 plugins and related code quality tools natively, including isort, yesqa, eradicate, and most of the rules implemented in pyupgrade. Ruff also supports automatically fixing its own lint violations, which Flake8 does not.'



## Multi-Hop vector store reasoning

Because vector stores are easily usable as tools in agents, it is easy to use answer multi-hop questions that depend on vector stores using the existing agent framework.


```python
tools = [
    Tool(
        name="State of Union QA System",
        func=state_of_union.run,
        description="useful for when you need to answer questions about the most recent state of the union address. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
    Tool(
        name="Ruff QA System",
        func=ruff.run,
        description="useful for when you need to answer questions about ruff (a python linter). Input should be a fully formed question, not referencing any obscure pronouns from the conversation before.",
    ),
]
```


```python
# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```


```python
agent.run(
    "What tool does ruff use to run over Jupyter Notebooks? Did the president mention that tool in the state of the union?"
)
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to find out what tool ruff uses to run over Jupyter Notebooks, and if the president mentioned it in the state of the union.
    Action: Ruff QA System
    Action Input: What tool does ruff use to run over Jupyter Notebooks?[0m
    Observation: [33;1m[1;3m Ruff is integrated into nbQA, a tool for running linters and code formatters over Jupyter Notebooks. After installing ruff and nbqa, you can run Ruff over a notebook like so: > nbqa ruff Untitled.html[0m
    Thought:[32;1m[1;3m I now need to find out if the president mentioned this tool in the state of the union.
    Action: State of Union QA System
    Action Input: Did the president mention nbQA in the state of the union?[0m
    Observation: [36;1m[1;3m No, the president did not mention nbQA in the state of the union.[0m
    Thought:[32;1m[1;3m I now know the final answer.
    Final Answer: No, the president did not mention nbQA in the state of the union.[0m
    
    [1m> Finished chain.[0m
    




    'No, the president did not mention nbQA in the state of the union.'




```python

```
