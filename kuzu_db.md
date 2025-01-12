# Kuzu

>[Kùzu](https://kuzudb.com) is an embeddable property graph database management system built for query speed and scalability.
> 
> Kùzu has a permissive (MIT) open source license and implements [Cypher](https://en.wikipedia.org/wiki/Cypher_(query_language)), a declarative graph query language that allows for expressive and efficient data querying in a property graph.
> It uses columnar storage and its query processor contains novel join algorithms that allow it to scale to very large graphs without sacrificing query performance.
> 
> This notebook shows how to use LLMs to provide a natural language interface to [Kùzu](https://kuzudb.com) database with Cypher.

## Setting up

Kùzu is an embedded database (it runs in-process), so there are no servers to manage.
Simply install it via its Python package:

```bash
pip install kuzu
```

Create a database on the local machine and connect to it:


```python
import kuzu

db = kuzu.Database("test_db")
conn = kuzu.Connection(db)
```

First, we create the schema for a simple movie database:


```python
conn.execute("CREATE NODE TABLE Movie (name STRING, PRIMARY KEY(name))")
conn.execute(
    "CREATE NODE TABLE Person (name STRING, birthDate STRING, PRIMARY KEY(name))"
)
conn.execute("CREATE REL TABLE ActedIn (FROM Person TO Movie)")
```




    <kuzu.query_result.QueryResult at 0x103a72290>



Then we can insert some data.


```python
conn.execute("CREATE (:Person {name: 'Al Pacino', birthDate: '1940-04-25'})")
conn.execute("CREATE (:Person {name: 'Robert De Niro', birthDate: '1943-08-17'})")
conn.execute("CREATE (:Movie {name: 'The Godfather'})")
conn.execute("CREATE (:Movie {name: 'The Godfather: Part II'})")
conn.execute(
    "CREATE (:Movie {name: 'The Godfather Coda: The Death of Michael Corleone'})"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Al Pacino' AND m.name = 'The Godfather Coda: The Death of Michael Corleone' CREATE (p)-[:ActedIn]->(m)"
)
conn.execute(
    "MATCH (p:Person), (m:Movie) WHERE p.name = 'Robert De Niro' AND m.name = 'The Godfather: Part II' CREATE (p)-[:ActedIn]->(m)"
)
```




    <kuzu.query_result.QueryResult at 0x103a9e750>



## Creating `KuzuQAChain`

We can now create the `KuzuGraph` and `KuzuQAChain`. To create the `KuzuGraph` we simply need to pass the database object to the `KuzuGraph` constructor.


```python
from langchain.chains import KuzuQAChain
from langchain_community.graphs import KuzuGraph
from langchain_openai import ChatOpenAI
```


```python
graph = KuzuGraph(db)
```


```python
chain = KuzuQAChain.from_llm(
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    graph=graph,
    verbose=True,
)
```

## Refresh graph schema information

If the schema of database changes, you can refresh the schema information needed to generate Cypher statements.
You can also display the schema of the Kùzu graph as demonstrated below.


```python
# graph.refresh_schema()
```


```python
print(graph.get_schema)
```

    Node properties: [{'properties': [('name', 'STRING')], 'label': 'Movie'}, {'properties': [('name', 'STRING'), ('birthDate', 'STRING')], 'label': 'Person'}]
    Relationships properties: [{'properties': [], 'label': 'ActedIn'}]
    Relationships: ['(:Person)-[:ActedIn]->(:Movie)']
    
    

## Querying the graph

We can now use the `KuzuQAChain` to ask questions of the graph.


```python
chain.invoke("Who acted in The Godfather: Part II?")
```

    
    
    [1m> Entering new KuzuQAChain chain...[0m
    Generated Cypher:
    [32;1m[1;3mMATCH (p:Person)-[:ActedIn]->(m:Movie)
    WHERE m.name = 'The Godfather: Part II'
    RETURN p.name[0m
    Full Context:
    [32;1m[1;3m[{'p.name': 'Al Pacino'}, {'p.name': 'Robert De Niro'}][0m
    
    [1m> Finished chain.[0m
    




    {'query': 'Who acted in The Godfather: Part II?',
     'result': 'Al Pacino, Robert De Niro acted in The Godfather: Part II.'}




```python
chain.invoke("Robert De Niro played in which movies?")
```

    
    
    [1m> Entering new KuzuQAChain chain...[0m
    Generated Cypher:
    [32;1m[1;3mMATCH (p:Person)-[:ActedIn]->(m:Movie)
    WHERE p.name = 'Robert De Niro'
    RETURN m.name[0m
    Full Context:
    [32;1m[1;3m[{'m.name': 'The Godfather: Part II'}][0m
    
    [1m> Finished chain.[0m
    




    {'query': 'Robert De Niro played in which movies?',
     'result': 'Robert De Niro played in The Godfather: Part II.'}




```python
chain.invoke("How many actors played in the Godfather: Part II?")
```

    
    
    [1m> Entering new KuzuQAChain chain...[0m
    Generated Cypher:
    [32;1m[1;3mMATCH (:Person)-[:ActedIn]->(:Movie {name: 'Godfather: Part II'})
    RETURN count(*)[0m
    Full Context:
    [32;1m[1;3m[{'COUNT_STAR()': 0}][0m
    
    [1m> Finished chain.[0m
    




    {'query': 'How many actors played in the Godfather: Part II?',
     'result': "I don't know the answer."}




```python
chain.invoke("Who is the oldest actor who played in The Godfather: Part II?")
```

    
    
    [1m> Entering new KuzuQAChain chain...[0m
    Generated Cypher:
    [32;1m[1;3mMATCH (p:Person)-[:ActedIn]->(m:Movie {name: 'The Godfather: Part II'})
    RETURN p.name
    ORDER BY p.birthDate ASC
    LIMIT 1[0m
    Full Context:
    [32;1m[1;3m[{'p.name': 'Al Pacino'}][0m
    
    [1m> Finished chain.[0m
    




    {'query': 'Who is the oldest actor who played in The Godfather: Part II?',
     'result': 'Al Pacino is the oldest actor who played in The Godfather: Part II.'}



## Use separate LLMs for Cypher and answer generation

You can specify `cypher_llm` and `qa_llm` separately to use different LLMs for Cypher generation and answer generation.


```python
chain = KuzuQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-4"),
    graph=graph,
    verbose=True,
)
```

    /Users/prrao/code/langchain/.venv/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 0.3.0. Use RunnableSequence, e.g., `prompt | llm` instead.
      warn_deprecated(
    


```python
chain.invoke("How many actors played in The Godfather: Part II?")
```

    
    
    [1m> Entering new KuzuQAChain chain...[0m
    

    /Users/prrao/code/langchain/.venv/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(
    

    Generated Cypher:
    [32;1m[1;3mMATCH (:Person)-[:ActedIn]->(:Movie {name: 'The Godfather: Part II'})
    RETURN count(*)[0m
    Full Context:
    [32;1m[1;3m[{'COUNT_STAR()': 2}][0m
    

    /Users/prrao/code/langchain/.venv/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: The method `Chain.__call__` was deprecated in langchain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
      warn_deprecated(
    

    
    [1m> Finished chain.[0m
    




    {'query': 'How many actors played in The Godfather: Part II?',
     'result': 'Two actors played in The Godfather: Part II.'}


