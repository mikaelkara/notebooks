# Using AnalyticDB as a vector database for OpenAI embeddings

This notebook guides you step by step on using AnalyticDB as a vector database for OpenAI embeddings.

This notebook presents an end-to-end process of:
1. Using precomputed embeddings created by OpenAI API.
2. Storing the embeddings in a cloud instance of AnalyticDB.
3. Converting raw text query to an embedding with OpenAI API.
4. Using AnalyticDB to perform the nearest neighbour search in the created collection.

### What is AnalyticDB

[AnalyticDB](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview) is a high-performance distributed vector database. Fully compatible with PostgreSQL syntax, you can effortlessly utilize it. AnalyticDB is Alibaba Cloud managed cloud-native database with strong-performed vector compute engine. Absolute out-of-box experience allow to scale into billions of data vectors processing with rich features including indexing algorithms, structured & non-structured data features, realtime update, distance metrics, scalar filtering, time travel searches etc. Also equipped with full OLAP database functionality and SLA commitment for production usage promise;

### Deployment options

- Using [AnalyticDB Cloud Vector Database](https://www.alibabacloud.com/help/zh/analyticdb-for-postgresql/latest/overview-2). [Click here](https://www.alibabacloud.com/product/hybriddb-postgresql) to fast deploy it.


## Prerequisites

For the purposes of this exercise we need to prepare a couple of things:

1. AnalyticDB cloud server instance.
2. The 'psycopg2' library to interact with the vector database. Any other postgresql client library is ok.
3. An [OpenAI API key](https://beta.openai.com/account/api-keys).



We might validate if the server was launched successfully by running a simple curl command:


### Install requirements

This notebook obviously requires the `openai` and `psycopg2` packages, but there are also some other additional libraries we will use. The following command installs them all:



```python
! pip install openai psycopg2 pandas wget
```

### Prepare your OpenAI API key

The OpenAI API key is used for vectorization of the documents and queries.

If you don't have an OpenAI API key, you can get one from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys).

Once you get your key, please add it to your environment variables as `OPENAI_API_KEY`.


```python
# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.
import os

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

    OPENAI_API_KEY is ready
    

## Connect to AnalyticDB
First add it to your environment variables. or you can just change the "psycopg2.connect" parameters below

Connecting to a running instance of AnalyticDB server is easy with the official Python library:


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

    100% [......................................................................] 698933052 / 698933052




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

AnalyticDB stores data in __relation__ where each object is described by at least one vector. Our relation will be called **articles** and each object will be described by both **title** and **content** vectors. \

We will start with creating a relation and create a vector index on both **title** and **content**, and then we will fill it with our precomputed embeddings.


```python
create_table_sql = '''
CREATE TABLE IF NOT EXISTS public.articles (
    id INTEGER NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector REAL[],
    content_vector REAL[],
    vector_id INTEGER
);

ALTER TABLE public.articles ADD PRIMARY KEY (id);
'''

# SQL statement for creating indexes
create_indexes_sql = '''
CREATE INDEX ON public.articles USING ann (content_vector) WITH (distancemeasure = l2, dim = '1536', pq_segments = '64', hnsw_m = '100', pq_centers = '2048');

CREATE INDEX ON public.articles USING ann (title_vector) WITH (distancemeasure = l2, dim = '1536', pq_segments = '64', hnsw_m = '100', pq_centers = '2048');
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
            # Replace '[' with '{' and ']' with '}'
            modified_line = line.replace('[', '{').replace(']', '}')
            yield modified_line

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
def query_analyticdb(query, collection_name, vector_name="title_vector", top_k=20):

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]

    # Convert the embedded_query to PostgreSQL compatible format
    embedded_query_pg = "{" + ",".join(map(str, embedded_query)) + "}"

    # Create SQL query
    query_sql = f"""
    SELECT id, url, title, l2_distance({vector_name},'{embedded_query_pg}'::real[]) AS similarity
    FROM {collection_name}
    ORDER BY {vector_name} <-> '{embedded_query_pg}'::real[]
    LIMIT {top_k};
    """
    # Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()

    return results
```


```python
import openai

query_results = query_analyticdb("modern art in Europe", "Articles")
for i, result in enumerate(query_results):
    print(f"{i + 1}. {result[2]} (Score: {round(1 - result[3], 3)})")
```

    1. Museum of Modern Art (Score: 0.75)
    2. Western Europe (Score: 0.735)
    3. Renaissance art (Score: 0.728)
    4. Pop art (Score: 0.721)
    5. Northern Europe (Score: 0.71)
    6. Hellenistic art (Score: 0.706)
    7. Modernist literature (Score: 0.694)
    8. Art film (Score: 0.687)
    9. Central Europe (Score: 0.685)
    10. European (Score: 0.683)
    11. Art (Score: 0.683)
    12. Byzantine art (Score: 0.682)
    13. Postmodernism (Score: 0.68)
    14. Eastern Europe (Score: 0.679)
    15. Europe (Score: 0.678)
    16. Cubism (Score: 0.678)
    17. Impressionism (Score: 0.677)
    18. Bauhaus (Score: 0.676)
    19. Surrealism (Score: 0.674)
    20. Expressionism (Score: 0.674)
    


```python
# This time we'll query using content vector
query_results = query_analyticdb("Famous battles in Scottish history", "Articles", "content_vector")
for i, result in enumerate(query_results):
    print(f"{i + 1}. {result[2]} (Score: {round(1 - result[3], 3)})")
```

    1. Battle of Bannockburn (Score: 0.739)
    2. Wars of Scottish Independence (Score: 0.723)
    3. 1651 (Score: 0.705)
    4. First War of Scottish Independence (Score: 0.699)
    5. Robert I of Scotland (Score: 0.692)
    6. 841 (Score: 0.688)
    7. 1716 (Score: 0.688)
    8. 1314 (Score: 0.674)
    9. 1263 (Score: 0.673)
    10. William Wallace (Score: 0.671)
    11. Stirling (Score: 0.663)
    12. 1306 (Score: 0.662)
    13. 1746 (Score: 0.661)
    14. 1040s (Score: 0.656)
    15. 1106 (Score: 0.654)
    16. 1304 (Score: 0.653)
    17. David II of Scotland (Score: 0.65)
    18. Braveheart (Score: 0.649)
    19. 1124 (Score: 0.648)
    20. July 27 (Score: 0.646)
    


```python

```
