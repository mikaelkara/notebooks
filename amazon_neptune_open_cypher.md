# Amazon Neptune with Cypher

>[Amazon Neptune](https://aws.amazon.com/neptune/) is a high-performance graph analytics and serverless database for superior scalability and availability.
>
>This example shows the QA chain that queries the `Neptune` graph database using `openCypher` and returns a human-readable response.
>
>[Cypher](https://en.wikipedia.org/wiki/Cypher_(query_language)) is a declarative graph query language that allows for expressive and efficient data querying in a property graph.
>
>[openCypher](https://opencypher.org/) is an open-source implementation of Cypher.# Neptune Open Cypher QA Chain
This QA chain queries Amazon Neptune using openCypher and returns human readable response

LangChain supports both [Neptune Database](https://docs.aws.amazon.com/neptune/latest/userguide/intro.html) and [Neptune Analytics](https://docs.aws.amazon.com/neptune-analytics/latest/userguide/what-is-neptune-analytics.html) with `NeptuneOpenCypherQAChain` 


Neptune Database is a serverless graph database designed for optimal scalability and availability. It provides a solution for graph database workloads that need to scale to 100,000 queries per second, Multi-AZ high availability, and multi-Region deployments. You can use Neptune Database for social networking, fraud alerting, and Customer 360 applications.

Neptune Analytics is an analytics database engine that can quickly analyze large amounts of graph data in memory to get insights and find trends. Neptune Analytics is a solution for quickly analyzing existing graph databases or graph datasets stored in a data lake. It uses popular graph analytic algorithms and low-latency analytic queries.

## Using Neptune Database


```python
from langchain_community.graphs import NeptuneGraph

host = "<neptune-host>"
port = 8182
use_https = True

graph = NeptuneGraph(host=host, port=port, use_https=use_https)
```

### Using Neptune Analytics


```python
from langchain_community.graphs import NeptuneAnalyticsGraph

graph = NeptuneAnalyticsGraph(graph_identifier="<neptune-analytics-graph-id>")
```

## Using NeptuneOpenCypherQAChain

This QA chain queries Neptune graph database using openCypher and returns human readable response.


```python
from langchain.chains import NeptuneOpenCypherQAChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4")

chain = NeptuneOpenCypherQAChain.from_llm(llm=llm, graph=graph)

chain.invoke("how many outgoing routes does the Austin airport have?")
```




    'The Austin airport has 98 outgoing routes.'


