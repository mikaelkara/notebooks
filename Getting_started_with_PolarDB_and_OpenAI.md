# Using PolarDB-PG as a vector database for OpenAI embeddings

This notebook guides you step by step on using PolarDB-PG as a vector database for OpenAI embeddings.

This notebook presents an end-to-end process of:
1. Using precomputed embeddings created by OpenAI API.
2. Storing the embeddings in a cloud instance of PolarDB-PG.
3. Converting raw text query to an embedding with OpenAI API.
4. Using PolarDB-PG to perform the nearest neighbour search in the created collection.

### What is PolarDB-PG

[PolarDB-PG](https://www.alibabacloud.com/help/en/polardb/latest/what-is-polardb-2) is a high-performance vector database that adopts a read-write separation architecture. It is a cloud-native database managed by Alibaba Cloud, 100% compatible with PostgreSQL, and highly compatible with Oracle syntax. It supports processing massive vector data storage and queries, and greatly improves the efficiency of vector calculations through optimization of underlying execution algorithms, providing users with fast, elastic, high-performance, massive storage, and secure and reliable vector database services. Additionally, PolarDB-PG also supports multi-dimensional and multi-modal spatiotemporal information engines and geographic information engines.At the same time, PolarDB-PG is equipped with complete OLAP functionality and service level agreements, which has been recognized and used by many users;







### Deployment options

- Using [PolarDB-PG Cloud Vector Database](https://www.alibabacloud.com/product/polardb-for-postgresql). [Click here](https://www.alibabacloud.com/product/polardb-for-postgresql?spm=a3c0i.147400.6791778070.243.9f204881g5cjpP) to fast deploy it.

## Prerequisites

For the purposes of this exercise we need to prepare a couple of things:

1. PolarDB-PG cloud server instance.
2. The 'psycopg2' library to interact with the vector database. Any other postgresql client library is ok.
3. An [OpenAI API key](https://beta.openai.com/account/api-keys).

We might validate if the server was launched successfully by running a simple curl command:

### Install requirements

This notebook obviously requires the `openai` and `psycopg2` packages, but there are also some other additional libraries we will use. The following command installs them all:



```python
! pip install openai psycopg2 pandas wget
```

Prepare your OpenAI API key
The OpenAI API key is used for vectorization of the documents and queries.

If you don't have an OpenAI API key, you can get one from https://beta.openai.com/account/api-keys.

Once you get your key, please add it to your environment variables as OPENAI_API_KEY.

If you have any doubts about setting the API key through environment variables, please refer to [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).


```python
# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

    OPENAI_API_KEY is ready
    

## Connect to PolarDB
First add it to your environment variables. or you can just change the "psycopg2.connect" parameters below

Connecting to a running instance of PolarDB server is easy with the official Python library:


```python
import os
import psycopg2

# Note. alternatively you can set a temporary env variable like this:
# os.environ["PGHOST"] = "your_host"
# os.environ["PGPORT"] "5432"),
# os.environ["PGDATABASE"] "postgres"),
# os.environ["PGUSER"] "user"),
# os.environ["PGPASSWORD"] "password"),

connection = psycopg2.connect(
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
    database=os.environ.get("PGDATABASE", "postgres"),
    user=os.environ.get("PGUSER", "user"),
    password=os.environ.get("PGPASSWORD", "password")
)

# Create a new cursor object
cursor = connection.cursor()
```

We can test the connection by running any available method:


```python
# Execute a simple query to test the connection
cursor.execute("SELECT 1;")
result = cursor.fetchone()

# Check the query result
if result == (1,):
    print("Connection successful!")
else:
    print("Connection failed.")
```

    Connection successful!
    


```python
import wget

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)
```




    'vector_database_wikipedia_articles_embedded.zip'



The downloaded file has to be then extracted:


```python
import zipfile
import os
import re
import tempfile

current_directory = os.getcwd()
zip_file_path = os.path.join(current_directory, "vector_database_wikipedia_articles_embedded.zip")
output_directory = os.path.join(current_directory, "../../data")

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(output_directory)


# check the csv file exist
file_name = "vector_database_wikipedia_articles_embedded.csv"
data_directory = os.path.join(current_directory, "../../data")
file_path = os.path.join(data_directory, file_name)


if os.path.exists(file_path):
    print(f"The file {file_name} exists in the data directory.")
else:
    print(f"The file {file_name} does not exist in the data directory.")
```

    The file vector_database_wikipedia_articles_embedded.csv exists in the data directory.
    

## Index data

PolarDB stores data in __relation__ where each object is described by at least one vector. Our relation will be called **articles** and each object will be described by both **title** and **content** vectors. 

We will start with creating a relation and create a vector index on both **title** and **content**, and then we will fill it with our precomputed embeddings.


```python
create_table_sql = '''
CREATE TABLE IF NOT EXISTS public.articles (
    id INTEGER NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector vector(1536),
    content_vector vector(1536),
    vector_id INTEGER
);

ALTER TABLE public.articles ADD PRIMARY KEY (id);
'''

# SQL statement for creating indexes
create_indexes_sql = '''
CREATE INDEX ON public.articles USING ivfflat (content_vector) WITH (lists = 1000);

CREATE INDEX ON public.articles USING ivfflat (title_vector) WITH (lists = 1000);
'''

# Execute the SQL statements
cursor.execute(create_table_sql)
cursor.execute(create_indexes_sql)

# Commit the changes
connection.commit()
```

## Load data

In this section we are going to load the data prepared previous to this session, so you don't have to recompute the embeddings of Wikipedia articles with your own credits.


```python
import io

# Path to your local CSV file
csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'

# Define a generator function to process the file line by line
def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line

# Create a StringIO object to store the modified lines
modified_lines = io.StringIO(''.join(list(process_file(csv_file_path))))

# Create the COPY command for the copy_expert method
copy_command = '''
COPY public.articles (id, url, title, content, title_vector, content_vector, vector_id)
FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',');
'''

# Execute the COPY command using the copy_expert method
cursor.copy_expert(copy_command, modified_lines)

# Commit the changes
connection.commit()
```


```python
# Check the collection size to make sure all the points have been stored
count_sql = """select count(*) from public.articles;"""
cursor.execute(count_sql)
result = cursor.fetchone()
print(f"Count:{result[0]}")
```

    Count:25000
    

## Search data

Once the data is put into Qdrant we will start querying the collection for the closest vectors. We may provide an additional parameter `vector_name` to switch from title to content based search. Since the precomputed embeddings were created with `text-embedding-3-small` OpenAI model we also have to use it during search.


```python
def query_polardb(query, collection_name, vector_name="title_vector", top_k=20):

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]

    # Convert the embedded_query to PostgreSQL compatible format
    embedded_query_pg = "[" + ",".join(map(str, embedded_query)) + "]"

    # Create SQL query
    query_sql = f"""
    SELECT id, url, title, l2_distance({vector_name},'{embedded_query_pg}'::VECTOR(1536)) AS similarity
    FROM {collection_name}
    ORDER BY {vector_name} <-> '{embedded_query_pg}'::VECTOR(1536)
    LIMIT {top_k};
    """
    # Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()

    return results
```


```python
import openai

query_results = query_polardb("modern art in Europe", "Articles")
for i, result in enumerate(query_results):
    print(f"{i + 1}. {result[2]} (Score: {round(1 - result[3], 3)})")
```

    1. Museum of Modern Art (Score: 0.5)
    2. Western Europe (Score: 0.485)
    3. Renaissance art (Score: 0.479)
    4. Pop art (Score: 0.472)
    5. Northern Europe (Score: 0.461)
    6. Hellenistic art (Score: 0.457)
    7. Modernist literature (Score: 0.447)
    8. Art film (Score: 0.44)
    9. Central Europe (Score: 0.439)
    10. European (Score: 0.437)
    11. Art (Score: 0.437)
    12. Byzantine art (Score: 0.436)
    13. Postmodernism (Score: 0.434)
    14. Eastern Europe (Score: 0.433)
    15. Europe (Score: 0.433)
    16. Cubism (Score: 0.432)
    17. Impressionism (Score: 0.432)
    18. Bauhaus (Score: 0.431)
    19. Surrealism (Score: 0.429)
    20. Expressionism (Score: 0.429)
    


```python
# This time we'll query using content vector
query_results = query_polardb("Famous battles in Scottish history", "Articles", "content_vector")
for i, result in enumerate(query_results):
    print(f"{i + 1}. {result[2]} (Score: {round(1 - result[3], 3)})")
```

    1. Battle of Bannockburn (Score: 0.489)
    2. Wars of Scottish Independence (Score: 0.474)
    3. 1651 (Score: 0.457)
    4. First War of Scottish Independence (Score: 0.452)
    5. Robert I of Scotland (Score: 0.445)
    6. 841 (Score: 0.441)
    7. 1716 (Score: 0.441)
    8. 1314 (Score: 0.429)
    9. 1263 (Score: 0.428)
    10. William Wallace (Score: 0.426)
    11. Stirling (Score: 0.419)
    12. 1306 (Score: 0.419)
    13. 1746 (Score: 0.418)
    14. 1040s (Score: 0.414)
    15. 1106 (Score: 0.412)
    16. 1304 (Score: 0.411)
    17. David II of Scotland (Score: 0.408)
    18. Braveheart (Score: 0.407)
    19. 1124 (Score: 0.406)
    20. July 27 (Score: 0.405)
    
