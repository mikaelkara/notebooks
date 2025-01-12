# Elasticsearch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain/blob/master/docs/docs/use_cases/qa_structured/integrations/elasticsearch.ipynb)

We can use LLMs to interact with Elasticsearch analytics databases in natural language.

This chain builds search queries via the Elasticsearch DSL API (filters and aggregations).

The Elasticsearch client must have permissions for index listing, mapping description and search queries.

See [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) for instructions on how to run Elasticsearch locally.


```python
! pip install langchain langchain-experimental openai elasticsearch

# Set env var OPENAI_API_KEY or load from a .env file
# import dotenv

# dotenv.load_dotenv()
```


```python
from elasticsearch import Elasticsearch
from langchain.chains.elasticsearch_database import ElasticsearchDatabaseChain
from langchain_openai import ChatOpenAI
```


```python
# Initialize Elasticsearch python client.
# See https://elasticsearch-py.readthedocs.io/en/v8.8.2/api.html#elasticsearch.Elasticsearch
ELASTIC_SEARCH_SERVER = "https://elastic:pass@localhost:9200"
db = Elasticsearch(ELASTIC_SEARCH_SERVER)
```

Uncomment the next cell to initially populate your db.


```python
# customers = [
#     {"firstname": "Jennifer", "lastname": "Walters"},
#     {"firstname": "Monica","lastname":"Rambeau"},
#     {"firstname": "Carol","lastname":"Danvers"},
#     {"firstname": "Wanda","lastname":"Maximoff"},
#     {"firstname": "Jennifer","lastname":"Takeda"},
# ]
# for i, customer in enumerate(customers):
#     db.create(index="customers", document=customer, id=i)
```


```python
llm = ChatOpenAI(model="gpt-4", temperature=0)
chain = ElasticsearchDatabaseChain.from_llm(llm=llm, database=db, verbose=True)
```


```python
question = "What are the first names of all the customers?"
chain.run(question)
```

We can customize the prompt.


```python
from langchain.prompts.prompt import PromptTemplate

PROMPT_TEMPLATE = """Given an input question, create a syntactically correct Elasticsearch query to run. Unless the user specifies in their question a specific number of examples they wish to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Unless told to do not query for all the columns from a specific index, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the mapping description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which index. Return the query as valid json.

Use the following format:

Question: Question here
ESQuery: Elasticsearch Query formatted as json
"""

PROMPT = PromptTemplate.from_template(
    PROMPT_TEMPLATE,
)
chain = ElasticsearchDatabaseChain.from_llm(llm=llm, database=db, query_prompt=PROMPT)
```
