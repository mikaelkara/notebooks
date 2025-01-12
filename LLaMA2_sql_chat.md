## LLaMA2 chat with SQL

Open source, local LLMs are great to consider for any application that demands data privacy.

SQL is one good example. 

This cookbook shows how to perform text-to-SQL using various local versions of LLaMA2 run locally.

## Packages


```python
! pip install langchain replicate
```

## LLM

There are a few ways to access LLaMA2.

To run locally, we use Ollama.ai. 

See [here](/docs/integrations/chat/ollama) for details on installation and setup.

Also, see [here](/docs/guides/development/local_llms) for our full guide on local LLMs.
 
To use an external API, which is not private, we can use Replicate.


```python
# Local
from langchain_community.chat_models import ChatOllama

llama2_chat = ChatOllama(model="llama2:13b-chat")
llama2_code = ChatOllama(model="codellama:7b-instruct")

# API
from langchain_community.llms import Replicate

# REPLICATE_API_TOKEN = getpass()
# os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
llama2_chat_replicate = Replicate(
    model=replicate_id, input={"temperature": 0.01, "max_length": 500, "top_p": 1}
)
```

    Init param `input` is deprecated, please use `model_kwargs` instead.
    


```python
# Simply set the LLM we want to use
llm = llama2_chat
```

## DB

Connect to a SQLite DB.

To create this particular DB, you can use the code and follow the steps shown [here](https://github.com/facebookresearch/llama-recipes/blob/main/demo_apps/StructuredLlama.ipynb).


```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info=0)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)
```

## Query a SQL Database 

Follow the runnables workflow [here](https://python.langchain.com/docs/expression_language/cookbook/sql_db).


```python
# Prompt
from langchain_core.prompts import ChatPromptTemplate

# Update the template based on the type of SQL Database like MySQL, Microsoft SQL Server and so on
template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        ("human", template),
    ]
)

# Chain to query
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

sql_response.invoke({"question": "What team is Klay Thompson on?"})
```




    ' SELECT "Team" FROM nba_roster WHERE "NAME" = \'Klay Thompson\';'



We can review the results:

* [LangSmith trace](https://smith.langchain.com/public/afa56a06-b4e2-469a-a60f-c1746e75e42b/r) LLaMA2-13 Replicate API
* [LangSmith trace](https://smith.langchain.com/public/2d4ecc72-6b8f-4523-8f0b-ea95c6b54a1d/r) LLaMA2-13 local 



```python
# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

full_chain.invoke({"question": "How many unique teams are there?"})
```




    AIMessage(content=' Based on the table schema and SQL query, there are 30 unique teams in the NBA.')



We can review the results:

* [LangSmith trace](https://smith.langchain.com/public/10420721-746a-4806-8ecf-d6dc6399d739/r) LLaMA2-13 Replicate API
* [LangSmith trace](https://smith.langchain.com/public/5265ebab-0a22-4f37-936b-3300f2dfa1c1/r) LLaMA2-13 local 

## Chat with a SQL DB 

Next, we can add memory.


```python
# Prompt
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = """Given an input question, convert it to a SQL query. No pre-amble. Based on the table schema below, write a SQL query that would answer the user's question:
{schema}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory
from langchain_core.runnables import RunnableLambda

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save
sql_response_memory.invoke({"question": "What team is Klay Thompson on?"})
```




    ' SELECT "Team" FROM nba_roster WHERE "NAME" = \'Klay Thompson\';'




```python
# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response_memory)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

full_chain.invoke({"question": "What is his salary?"})
```




    AIMessage(content=' Sure! Here\'s the natural language response based on the given input:\n\n"Klay Thompson\'s salary is $43,219,440."')



Here is the [trace](https://smith.langchain.com/public/54794d18-2337-4ce2-8b9f-3d8a2df89e51/r).
