# Memgraph

>[Memgraph](https://github.com/memgraph/memgraph) is the open-source graph database, compatible with `Neo4j`.
>The database is using the `Cypher` graph query language, 
>
>[Cypher](https://en.wikipedia.org/wiki/Cypher_(query_language)) is a declarative graph query language that allows for expressive and efficient data querying in a property graph.

This notebook shows how to use LLMs to provide a natural language interface to a [Memgraph](https://github.com/memgraph/memgraph) database.


## Setting up

To complete this tutorial, you will need [Docker](https://www.docker.com/get-started/) and [Python 3.x](https://www.python.org/) installed.

Ensure you have a running Memgraph instance. To quickly run Memgraph Platform (Memgraph database + MAGE library + Memgraph Lab) for the first time, do the following:

On Linux/MacOS:
```
curl https://install.memgraph.com | sh
```

On Windows:
```
iwr https://windows.memgraph.com | iex
```

Both commands run a script that downloads a Docker Compose file to your system, builds and starts `memgraph-mage` and `memgraph-lab` Docker services in two separate containers. 

Read more about the installation process on [Memgraph documentation](https://memgraph.com/docs/getting-started/install-memgraph).

Now you can start playing with `Memgraph`!

Begin by installing and importing all the necessary packages. We'll use the package manager called [pip](https://pip.pypa.io/en/stable/installation/), along with the `--user` flag, to ensure proper permissions. If you've installed Python 3.4 or a later version, pip is included by default. You can install all the required packages using the following command:


```python
pip install langchain langchain-openai neo4j gqlalchemy --user
```

You can either run the provided code blocks in this notebook or use a separate Python file to experiment with Memgraph and LangChain.


```python
import os

from gqlalchemy import Memgraph
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
```

We're utilizing the Python library [GQLAlchemy](https://github.com/memgraph/gqlalchemy) to establish a connection between our Memgraph database and Python script. You can establish the connection to a running Memgraph instance with the Neo4j driver as well, since it's compatible with Memgraph. To execute queries with GQLAlchemy, we can set up a Memgraph instance as follows:


```python
memgraph = Memgraph(host="127.0.0.1", port=7687)
```

## Populating the database
You can effortlessly populate your new, empty database using the Cypher query language. Don't worry if you don't grasp every line just yet, you can learn Cypher from the documentation [here](https://memgraph.com/docs/cypher-manual/). Running the following script will execute a seeding query on the database, giving us data about a video game, including details like the publisher, available platforms, and genres. This data will serve as a basis for our work.


```python
# Creating and executing the seeding query
query = """
    MERGE (g:Game {name: "Baldur's Gate 3"})
    WITH g, ["PlayStation 5", "Mac OS", "Windows", "Xbox Series X/S"] AS platforms,
            ["Adventure", "Role-Playing Game", "Strategy"] AS genres
    FOREACH (platform IN platforms |
        MERGE (p:Platform {name: platform})
        MERGE (g)-[:AVAILABLE_ON]->(p)
    )
    FOREACH (genre IN genres |
        MERGE (gn:Genre {name: genre})
        MERGE (g)-[:HAS_GENRE]->(gn)
    )
    MERGE (p:Publisher {name: "Larian Studios"})
    MERGE (g)-[:PUBLISHED_BY]->(p);
"""

memgraph.execute(query)
```

## Refresh graph schema

You're all set to instantiate the Memgraph-LangChain graph using the following script. This interface will allow us to query our database using LangChain, automatically creating the required graph schema for generating Cypher queries through LLM.


```python
graph = MemgraphGraph(url="bolt://localhost:7687", username="", password="")
```

If necessary, you can manually refresh the graph schema as follows.


```python
graph.refresh_schema()
```

To familiarize yourself with the data and verify the updated graph schema, you can print it using the following statement.


```python
print(graph.schema)
```

```
Node properties are the following:
Node name: 'Game', Node properties: [{'property': 'name', 'type': 'str'}]
Node name: 'Platform', Node properties: [{'property': 'name', 'type': 'str'}]
Node name: 'Genre', Node properties: [{'property': 'name', 'type': 'str'}]
Node name: 'Publisher', Node properties: [{'property': 'name', 'type': 'str'}]

Relationship properties are the following:

The relationships are the following:
['(:Game)-[:AVAILABLE_ON]->(:Platform)']
['(:Game)-[:HAS_GENRE]->(:Genre)']
['(:Game)-[:PUBLISHED_BY]->(:Publisher)']
```

## Querying the database

To interact with the OpenAI API, you must configure your API key as an environment variable using the Python [os](https://docs.python.org/3/library/os.html) package. This ensures proper authorization for your requests. You can find more information on obtaining your API key [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key).


```python
os.environ["OPENAI_API_KEY"] = "your-key-here"
```

You should create the graph chain using the following script, which will be utilized in the question-answering process based on your graph data. While it defaults to GPT-3.5-turbo, you might also consider experimenting with other models like [GPT-4](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4) for notably improved Cypher queries and outcomes. We'll utilize the OpenAI chat, utilizing the key you previously configured. We'll set the temperature to zero, ensuring predictable and consistent answers. Additionally, we'll use our Memgraph-LangChain graph and set the verbose parameter, which defaults to False, to True to receive more detailed messages regarding query generation.


```python
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, model_name="gpt-3.5-turbo"
)
```

Now you can start asking questions!


```python
response = chain.run("Which platforms is Baldur's Gate 3 available on?")
print(response)
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (g:Game {name: 'Baldur\'s Gate 3'})-[:AVAILABLE_ON]->(p:Platform)
RETURN p.name
Full Context:
[{'p.name': 'PlayStation 5'}, {'p.name': 'Mac OS'}, {'p.name': 'Windows'}, {'p.name': 'Xbox Series X/S'}]

> Finished chain.
Baldur's Gate 3 is available on PlayStation 5, Mac OS, Windows, and Xbox Series X/S.
```


```python
response = chain.run("Is Baldur's Gate 3 available on Windows?")
print(response)
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (:Game {name: 'Baldur\'s Gate 3'})-[:AVAILABLE_ON]->(:Platform {name: 'Windows'})
RETURN true
Full Context:
[{'true': True}]

> Finished chain.
Yes, Baldur's Gate 3 is available on Windows.
```

## Chain modifiers

To modify the behavior of your chain and obtain more context or additional information, you can modify the chain's parameters.

#### Return direct query results
The `return_direct` modifier specifies whether to return the direct results of the executed Cypher query or the processed natural language response.


```python
# Return the result of querying the graph directly
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, return_direct=True
)
```


```python
response = chain.run("Which studio published Baldur's Gate 3?")
print(response)
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (:Game {name: 'Baldur\'s Gate 3'})-[:PUBLISHED_BY]->(p:Publisher)
RETURN p.name

> Finished chain.
[{'p.name': 'Larian Studios'}]
```

#### Return query intermediate steps
The `return_intermediate_steps` chain modifier enhances the returned response by including the intermediate steps of the query in addition to the initial query result.


```python
# Return all the intermediate steps of query execution
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, return_intermediate_steps=True
)
```


```python
response = chain("Is Baldur's Gate 3 an Adventure game?")
print(f"Intermediate steps: {response['intermediate_steps']}")
print(f"Final response: {response['result']}")
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (g:Game {name: 'Baldur\'s Gate 3'})-[:HAS_GENRE]->(genre:Genre {name: 'Adventure'})
RETURN g, genre
Full Context:
[{'g': {'name': "Baldur's Gate 3"}, 'genre': {'name': 'Adventure'}}]

> Finished chain.
Intermediate steps: [{'query': "MATCH (g:Game {name: 'Baldur\\'s Gate 3'})-[:HAS_GENRE]->(genre:Genre {name: 'Adventure'})\nRETURN g, genre"}, {'context': [{'g': {'name': "Baldur's Gate 3"}, 'genre': {'name': 'Adventure'}}]}]
Final response: Yes, Baldur's Gate 3 is an Adventure game.
```

#### Limit the number of query results
The `top_k` modifier can be used when you want to restrict the maximum number of query results.


```python
# Limit the maximum number of results returned by query
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, top_k=2
)
```


```python
response = chain.run("What genres are associated with Baldur's Gate 3?")
print(response)
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (:Game {name: 'Baldur\'s Gate 3'})-[:HAS_GENRE]->(g:Genre)
RETURN g.name
Full Context:
[{'g.name': 'Adventure'}, {'g.name': 'Role-Playing Game'}]

> Finished chain.
Baldur's Gate 3 is associated with the genres Adventure and Role-Playing Game.
```

# Advanced querying

As the complexity of your solution grows, you might encounter different use-cases that require careful handling. Ensuring your application's scalability is essential to maintain a smooth user flow without any hitches.

Let's instantiate our chain once again and attempt to ask some questions that users might potentially ask.


```python
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True, model_name="gpt-3.5-turbo"
)
```


```python
response = chain.run("Is Baldur's Gate 3 available on PS5?")
print(response)
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (g:Game {name: 'Baldur\'s Gate 3'})-[:AVAILABLE_ON]->(p:Platform {name: 'PS5'})
RETURN g.name, p.name
Full Context:
[]

> Finished chain.
I'm sorry, but I don't have the information to answer your question.
```

The generated Cypher query looks fine, but we didn't receive any information in response. This illustrates a common challenge when working with LLMs - the misalignment between how users phrase queries and how data is stored. In this case, the difference between user perception and the actual data storage can cause mismatches. Prompt refinement, the process of honing the model's prompts to better grasp these distinctions, is an efficient solution that tackles this issue. Through prompt refinement, the model gains increased proficiency in generating precise and pertinent queries, leading to the successful retrieval of the desired data.

### Prompt refinement

To address this, we can adjust the initial Cypher prompt of the QA chain. This involves adding guidance to the LLM on how users can refer to specific platforms, such as PS5 in our case. We achieve this using the LangChain [PromptTemplate](/docs/how_to#prompt-templates), creating a modified initial prompt. This modified prompt is then supplied as an argument to our refined Memgraph-LangChain instance.


```python
CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
If the user asks about PS5, Play Station 5 or PS 5, that is the platform called PlayStation 5.

The question is:
{question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)
```


```python
chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    graph=graph,
    verbose=True,
    model_name="gpt-3.5-turbo",
)
```


```python
response = chain.run("Is Baldur's Gate 3 available on PS5?")
print(response)
```

```
> Entering new GraphCypherQAChain chain...
Generated Cypher:
MATCH (g:Game {name: 'Baldur\'s Gate 3'})-[:AVAILABLE_ON]->(p:Platform {name: 'PlayStation 5'})
RETURN g.name, p.name
Full Context:
[{'g.name': "Baldur's Gate 3", 'p.name': 'PlayStation 5'}]

> Finished chain.
Yes, Baldur's Gate 3 is available on PlayStation 5.
```

Now, with the revised initial Cypher prompt that includes guidance on platform naming, we are obtaining accurate and relevant results that align more closely with user queries. 

This approach allows for further improvement of your QA chain. You can effortlessly integrate extra prompt refinement data into your chain, thereby enhancing the overall user experience of your app.
