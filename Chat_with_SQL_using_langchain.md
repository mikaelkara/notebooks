##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Chat with SQL using LangChain

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/langchain/Chat_with_SQL_using_langchain.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

Reading an SQL database can be challenging for humans. However, with accurate prompts, Gemini models can generate answers based on the data. Through the use of the Gemini API, you will be able retrieve necessary information by chatting with a SQL database.


```
!pip install -U -q "google-generativeai>=0.7.2" langchain langchain-community langchain-google-genai
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m158.8/158.8 kB[0m [31m1.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m974.0/974.0 kB[0m [31m11.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m27.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m314.7/314.7 kB[0m [31m6.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m124.9/124.9 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.2/49.2 kB[0m [31m906.2 kB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.0/53.0 kB[0m [31m2.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m142.7/142.7 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [?25h


```
import sqlite3

from langchain.chains import create_sql_query_chain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

import google.generativeai as genai
from IPython.display import Markdown
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

## Setting up the database
To query a database, you first need to set one up.

1. **Load the California Housing Dataset:** Load the dataset from sklearn.datasets and extract it into a DataFrame.



```
from sklearn.datasets import fetch_california_housing

california_housing_bunch = fetch_california_housing(as_frame=True)
california_housing_df = california_housing_bunch.frame
```

2. **Connect to the SQLite database:** The database will be stored in the specified file.


```
conn = sqlite3.connect("mydatabase.db")

# Write the DataFrame to a SQL table named 'housing'.
california_housing_df.to_sql("housing", conn, index=False)
```




    20640




```
# Create an SQLDatabase object
db = SQLDatabase.from_uri("sqlite:///mydatabase.db")
```

## Question to query
With the database connection established, the `SQLDatabase` object now contains information about our database, which the model can access.

You can now start asking the LLM to generate queries.



```
# you can see what information is available
Markdown(db.get_table_info())
```





CREATE TABLE housing (
	"MedInc" REAL, 
	"HouseAge" REAL, 
	"AveRooms" REAL, 
	"AveBedrms" REAL, 
	"Population" REAL, 
	"AveOccup" REAL, 
	"Latitude" REAL, 
	"Longitude" REAL, 
	"MedHouseVal" REAL
)

/*
3 rows from housing table:
MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude	MedHouseVal
8.3252	41.0	6.984126984126984	1.0238095238095237	322.0	2.5555555555555554	37.88	-122.23	4.526
8.3014	21.0	6.238137082601054	0.9718804920913884	2401.0	2.109841827768014	37.86	-122.22	3.585
7.2574	52.0	8.288135593220339	1.073446327683616	496.0	2.8022598870056497	37.85	-122.24	3.521
*/




```
# Define query chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
write_query_chain = create_sql_query_chain(llm, db)
```

You use `create_sql_query_chain` that fits our database. It provides default prompts for various types of SQL including Oracle, Google SQL, MySQL and more.


In this case, default prompt is suitable for the task. However, feel free to experiment with writing this part of our chain yourself to suit your preferences.


```
Markdown(write_query_chain.get_prompts()[0].template)
```




You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use the following tables:
{table_info}

Question: {input}




```
response = write_query_chain.invoke({"question": "What is the total population?"})
response
```




    'Question: What is the total population?\nSQLQuery: SELECT SUM("Population") FROM housing'




```
db.run('SELECT SUM("Population") FROM housing')
```




    '[(29421840.0,)]'



Great! The SQL query is correct, but it needs proper formatting before it can be executed directly by the database.


## Validating the query
You will pass the output of the previous query to a model that will extract just the SQL query and ensure its validity.


```
validate_prompt = PromptTemplate(
    input_variables=["not_formatted_query"],
    template="""You are going to receive a text that contains a SQL query. Extract that query.
    Make sure that it is a valid SQL command that can be passed directly to the Database.
    Avoid using Markdown for this task.
    Text: {not_formatted_query}"""
)
```


```
validate_chain = write_query_chain | validate_prompt | llm | StrOutputParser()
validate_chain.invoke({"question": "What is the total population?"})
```




    'SELECT SUM("Population") FROM housing \n'



## Automatic querying
Now, let's automate the process of querying the database using *QuerySQLDataBaseTool*. This tool can receive text from previous parts of the chain, execute the query, and return the answer.



```
execute_query = QuerySQLDataBaseTool(db=db)
execute_chain = validate_chain | execute_query
execute_chain.invoke({"question": "What is the total population?"})
```




    '[(29421840.0,)]'



## Generating answer
You are almost done!

To enhance our output, you'll use LLM not only to get the number but to get properly formatted and natural language response.


```
answer_prompt = PromptTemplate.from_template(
    """You are going to receive a original user question, generated SQL query, and result of said query. You should use this information to answer the original question. Use only information provided to you.

Original Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer_chain = (
    RunnablePassthrough.assign(query=validate_chain).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt | llm | StrOutputParser()
)

answer_chain.invoke({"question": "What is the total population?"})
```




    'The total population is 29,421,840. \n'



## Next steps

Congratulations! You've successfully created a functional chain to interact with SQL. Now, feel free to explore further by asking different questions.
