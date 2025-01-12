# Using JSON Mode and Function Calling for SQL Querying

With the rise of Large Language Models (LLMs), one of the first practical applications has been the "chat with X" app. In this notebook we will explore methods for building "chat with my database" tools with Groq API, exploring benefits and drawbacks of each and leveraging Groq API's [JSON mode](https://console.groq.com/docs/text-chat#json-mode) and [tool use](https://console.groq.com/docs/tool-use) feature for function calling.

We will show two methods for using Groq API to query a database, and how leveraging tool use for function calling can improve the predictability and reliability of our outputs. We will use the [DuckDB](https://duckdb.org/) query language on local CSV files in this example, but the general framework could also work against standard data warehouse platforms like [BigQuery](https://cloud.google.com/bigquery).

### Setup


```python
from groq import Groq
import os 
import json
import sqlparse
from IPython.display import Markdown
import duckdb
import glob
import yaml
```

We will use the ```llama3-70b-8192``` model in this demo. Note that you will need a Groq API Key to proceed and can create an account [here](https://console.groq.com/) to generate one for free.


```python
client = Groq(api_key = os.getenv('GROQ_API_KEY'))
model = 'llama3-70b-8192'
```

### Text-To-SQL

The first method is a standard **Text-To-SQL** implementation. With Text-To-SQL, we describe the database schema to the LLM, ask it to answer a question, and let it write an on-the-fly SQL query against the database to answer that question. Let's see how we can use the Groq API to build a Text-To-SQL pipeline:

First, we have our system prompt. A system prompt is an initial input or instruction given to the model, setting the context or specifying the task it needs to perform, essentially guiding the model's response generation. In our case, our system prompt will serve 3 purposes:

1. Provide the metadata schemas for our database tables
2. Indicate any relevant context or tips for querying the DuckDB language or our database schema specifically
3. Define our desired JSON output (note that to use JSON mode, we must include 'JSON' in the prompt)


```python
system_prompt = '''
You are Groq Advisor, and you are tasked with generating SQL queries for DuckDB based on user questions about data stored in two tables derived from CSV files:

Table: employees.csv
Columns:
employee_id (INTEGER): A unique identifier for each employee.
name (VARCHAR): The full name of the employee.
email (VARCHAR): employee's email address

Table: purchases.csv
Columns:
purchase_id (INTEGER): A unique identifier for each purchase.
purchase_date (DATE): Date of purchase
employee_id (INTEGER): References the employee_id from the employees table, indicating which employee made the purchase.
amount (FLOAT): The monetary value of the purchase.
product_name (STRING): The name of the product purchased

Given a user's question about this data, write a valid DuckDB SQL query that accurately extracts or calculates the requested information from these tables and adheres to SQL best practices for DuckDB, optimizing for readability and performance where applicable.

Here are some tips for writing DuckDB queries:
* DuckDB syntax requires querying from the .csv file itself, i.e. employees.csv and purchases.csv. For example: SELECT * FROM employees.csv as employees
* All tables referenced MUST be aliased
* DuckDB does not implicitly include a GROUP BY clause
* CURRENT_DATE gets today's date
* Aggregated fields like COUNT(*) must be appropriately named

And some rules for querying the dataset:
* Never include employee_id in the output - show employee name instead

Also note that:
* Valid values for product_name include 'Tesla','iPhone' and 'Humane pin'


Question:
--------
{user_question}
--------
Reminder: Generate a DuckDB SQL to answer to the question:
* respond as a valid JSON Document
* [Best] If the question can be answered with the available tables: {"sql": <sql here>}
* If the question cannot be answered with the available tables: {"error": <explanation here>}
* Ensure that the entire output is returned on only one single line
* Keep your query as simple and straightforward as possible; do not use subqueries
'''
```

Now we will define a ```text_to_sql``` function which takes in the system prompt and the user's question and outputs the LLM-generated DuckDB SQL query. Note that since we are using Groq API's [JSON mode](https://console.groq.com/docs/text-chat#json-mode-object-object) to format our output, we must indicate our expected JSON output format in either the system or user prompt.


```python
def text_to_sql(client,system_prompt,user_question,model):

    completion = client.chat.completions.create(
        model = model,
        response_format = {"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    )
  
    return completion.choices[0].message.content
```

...and a function for executing the DuckDB query that was generated:


```python
def execute_duckdb_query(query):
    original_cwd = os.getcwd()
    os.chdir('data')
    
    try:
        conn = duckdb.connect(database=':memory:', read_only=False)
        query_result = conn.execute(query).fetchdf().reset_index(drop=True)
    finally:
        os.chdir(original_cwd)


    return query_result
```

Now, we can query our database just by asking a question about the data. Here, the LLM generates a valid SQL query that reasonably answers the question:


```python
user_question = 'What are the most recent purchases?'


llm_response = text_to_sql(client,system_prompt,user_question,model)
sql_json = json.loads(llm_response)
parsed_sql = sqlparse.format(sql_json['sql'], reindent=True, keyword_case='upper')
formatted_sql = f"```sql\n{parsed_sql.strip()}\n```"
display(Markdown(formatted_sql)) 

execute_duckdb_query(parsed_sql)
```


```sql
SELECT e.name,
       p.purchase_date,
       p.product_name,
       p.amount
FROM purchases.csv AS p
JOIN employees.csv AS e ON p.employee_id = e.employee_id
ORDER BY p.purchase_date DESC
LIMIT 10;
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>purchase_date</th>
      <th>product_name</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jared Dunn</td>
      <td>2024-02-05</td>
      <td>Tesla</td>
      <td>75000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bertram Gilfoyle</td>
      <td>2024-02-04</td>
      <td>iPhone</td>
      <td>700</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dinesh Chugtai</td>
      <td>2024-02-03</td>
      <td>Humane pin</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Erlich Bachman</td>
      <td>2024-02-02</td>
      <td>Tesla</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Richard Hendricks</td>
      <td>2024-02-01</td>
      <td>iPhone</td>
      <td>750</td>
    </tr>
  </tbody>
</table>
</div>



Note, however, that due to the non-deterministic nature of LLMs, we cannot guarantee a reliable or consistent result every time. I might get a different result than you, and I might get a totally different query tomorrow. How should "most recent purchases" be defined? Which fields should be returned?

Obviously, this is not ideal for making any kind of data-driven decisions. It's hard enough to land on a reliable source-of-truth data model, and even harder when your AI analyst cannot give you a consistent result. While text-to-SQL can be great for generating ad-hoc insights, the non-determinism feature of LLMs makes raw text-to-SQL an impractical solution for a production environment.

### Function Calling for Verified Queries

A different approach is to leverage the LLM to call on pre-vetted queries that can answer a set of questions. Since you wouldn't trust a traditional business intelligence tool without rigorously developed and validated SQL, a "chat with your data" app should be no different. For this example, we will use the verified queries stored [here](https://github.com/groq/groq-api-cookbook/tree/main/function-calling-sql/verified-queries).


```python
def get_verified_queries(directory_path):
    verified_queries_yaml_files = glob.glob(os.path.join(directory_path, '*.yaml'))
    verified_queries_dict = {}
    for file in verified_queries_yaml_files:
        with open(file, 'r') as stream:
            try:
                file_name = file[len(directory_path):-5]
                verified_queries_dict[file_name] = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                continue
        
    return verified_queries_dict

directory_path = 'verified-queries/'
verified_queries_dict = get_verified_queries(directory_path)
verified_queries_dict
```




    {'most-recent-purchases': {'description': 'Five most recent purchases',
      'sql': 'SELECT \n       purchases.purchase_date,\n       purchases.product_name,\n       purchases.amount,\n       employees.name\nFROM purchases.csv AS purchases\nJOIN employees.csv AS employees ON purchases.employee_id = employees.employee_id\nORDER BY purchases.purchase_date DESC\nLIMIT 5;\n'},
     'most-expensive-purchase': {'description': 'Employee with the most expensive purchase',
      'sql': 'SELECT employees.name AS employee_name,\n      MAX(amount) AS max_purchase_amount\nFROM purchases.csv AS purchases\nJOIN employees.csv AS employees ON purchases.employee_id = employees.employee_id\nGROUP BY employees.name\nORDER BY max_purchase_amount DESC\nLIMIT 1\n'},
     'number-of-teslas': {'description': 'Number of Teslas purchased',
      'sql': "SELECT COUNT(*) as number_of_teslas\nFROM purchases.csv AS p\nJOIN employees.csv AS e ON e.employee_id = p.employee_id\nWHERE p.product_name = 'Tesla'\n"},
     'employees-without-purchases': {'description': 'Employees without a purchase since Feb 1, 2024',
      'sql': "SELECT employees.name as employees_without_purchases\nFROM employees.csv AS employees\nLEFT JOIN purchases.csv AS purchases ON employees.employee_id = purchases.employee_id\nAND purchases.purchase_date > '2024-02-01'\nWHERE purchases.purchase_id IS NULL\n"}}



Note that each of these queries are stored in ```yaml``` files with some additional metadata, like a description. This metadata is important for when the LLM needs to select the most appropriate query for the question at hand.

Now, let's define a new function for executing SQL - this one is tweaked slightly to extract the SQL query from ```verified_queries_dict``` inside the function, given a query name:


```python
def execute_duckdb_query_function_calling(query_name,verified_queries_dict):
    
    original_cwd = os.getcwd()
    os.chdir('data')

    query = verified_queries_dict[query_name]['sql']
    
    try:
        conn = duckdb.connect(database=':memory:', read_only=False)
        query_result = conn.execute(query).fetchdf().reset_index(drop=True)
    finally:
        os.chdir(original_cwd)

    return query_result
```

Finally, we will write a function to utilize Groq API's [Tool Use](https://console.groq.com/docs/tool-use) functionality to call the ```execute_duckdb_query_function_calling``` with the appropriate query name. We will provide the query/description mappings from ```verified_queries_dict``` in the system prompt so that the LLM can determine which query most appropriately answers the user's question:


```python
def call_verified_sql(user_question,verified_queries_dict,model):
    
    #Simplify verified_queries_dict to just show query name and description
    query_description_mapping = {key: subdict['description'] for key, subdict in verified_queries_dict.items()}
    
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": '''You are a function calling LLM that uses the data extracted from the execute_duckdb_query_function_calling function to answer questions around a DuckDB dataset.
    
            Extract the query_name parameter from this mapping by finding the one whose description best matches the user's question: 
            {query_description_mapping}
            '''.format(query_description_mapping=query_description_mapping)
        },
        {
            "role": "user",
            "content": user_question,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_duckdb_query_function_calling",
                "description": "Executes a verified DuckDB SQL Query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_name": {
                            "type": "string",
                            "description": "The name of the verified query (i.e. 'most-recent-purchases')",
                        }
                    },
                    "required": ["query_name"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  
        max_tokens=4096
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    available_functions = {
        "execute_duckdb_query_function_calling": execute_duckdb_query_function_calling,
    }
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        print('Query found: ',function_args.get("query_name"))
        function_response = function_to_call(
            query_name=function_args.get("query_name"),
            verified_queries_dict=verified_queries_dict
        )
    
    return function_response
```

Now, when we ask the LLM "What were the most recent purchases?", we will get the same logic every time:


```python
user_prompt = 'What were the most recent purchases?'
call_verified_sql(user_prompt,verified_queries_dict,model)
```

    Query found:  most-recent-purchases
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>purchase_date</th>
      <th>product_name</th>
      <th>amount</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-05</td>
      <td>Tesla</td>
      <td>75000</td>
      <td>Jared Dunn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-04</td>
      <td>iPhone</td>
      <td>700</td>
      <td>Bertram Gilfoyle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-03</td>
      <td>Humane pin</td>
      <td>500</td>
      <td>Dinesh Chugtai</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-02</td>
      <td>Tesla</td>
      <td>70000</td>
      <td>Erlich Bachman</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-01</td>
      <td>iPhone</td>
      <td>750</td>
      <td>Richard Hendricks</td>
    </tr>
  </tbody>
</table>
</div>



The downside with using verified queries, of course, is having to write and verify them, which takes away from the magic of watching an LLM generate a SQL query on the fly. But in an environment where reliability is critical, function calling against verified queries is a much more consistent way to chat with your data than Text-To-SQL.

This is a simple example, but you could even take it a step further by defining parameters for each query (that you might find in a WHERE clause), and doing another function call once the verified query is found to find the parameter(s) to inject in the query from the user prompt. Go ahead and try it out!

### Conclusion

In this notebook we've explored two techniques for writing and executing SQL with LLMs using Groq API: Text-to-SQL (where the LLM generates SQL in the moment) and Verified Queries (where the LLM determines which verified query is most appropriate for your question and executes it). But perhaps the best approach is a blend - for ad-hoc reporting, there is still a lot of power in Text-to-SQL for quick answers. For user questions where there is no good verified query, you could default to using Text-To-SQL and then add that query to your dictionary of verified queries if it looks good. Either way, using LLMs on top of your data will lead to better and faster insights - just be sure to follow good data governance practices.
