# Using Hologres as a vector database for OpenAI embeddings

This notebook guides you step by step on using Hologres as a vector database for OpenAI embeddings.

This notebook presents an end-to-end process of:
1. Using precomputed embeddings created by OpenAI API.
2. Storing the embeddings in a cloud instance of Hologres.
3. Converting raw text query to an embedding with OpenAI API.
4. Using Hologres to perform the nearest neighbour search in the created collection.
5. Provide large language models with the search results as context in prompt engineering

### What is Hologres

[Hologres](https://www.alibabacloud.com/help/en/hologres/latest/what-is-hologres) is a unified real-time data warehousing service developed by Alibaba Cloud. You can use Hologres to write, update, process, and analyze large amounts of data in real time. Hologres supports standard SQL syntax, is compatible with PostgreSQL, and supports most PostgreSQL functions. Hologres supports online analytical processing (OLAP) and ad hoc analysis for up to petabytes of data, and provides high-concurrency and low-latency online data services. Hologres supports fine-grained isolation of multiple workloads and enterprise-level security capabilities. Hologres is deeply integrated with MaxCompute, Realtime Compute for Apache Flink, and DataWorks, and provides full-stack online and offline data warehousing solutions for enterprises.

Hologres provides vector database functionality by adopting [Proxima](https://www.alibabacloud.com/help/en/hologres/latest/vector-processing).

Proxima is a high-performance software library developed by Alibaba DAMO Academy. It allows you to search for the nearest neighbors of vectors. Proxima provides higher stability and performance than similar open source software such as Facebook AI Similarity Search (Faiss). Proxima provides basic modules that have leading performance and effects in the industry and allows you to search for similar images, videos, or human faces. Hologres is deeply integrated with Proxima to provide a high-performance vector search service.

### Deployment options

- [Click here](https://www.alibabacloud.com/product/hologres) to fast deploy [Hologres data warehouse](https://www.alibabacloud.com/help/en/hologres/latest/getting-started).


## Prerequisites

For the purposes of this exercise we need to prepare a couple of things:

1. Hologres cloud server instance.
2. The 'psycopg2-binary' library to interact with the vector database. Any other postgresql client library is ok.
3. An [OpenAI API key](https://beta.openai.com/account/api-keys).



We might validate if the server was launched successfully by running a simple curl command:


### Install requirements

This notebook obviously requires the `openai` and `psycopg2-binary` packages, but there are also some other additional libraries we will use. The following command installs them all:



```python
! pip install openai psycopg2-binary pandas wget
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
    

## Connect to Hologres
First add it to your environment variables. or you can just change the "psycopg2.connect" parameters below

Connecting to a running instance of Hologres server is easy with the official Python library:


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
connection.set_session(autocommit=True)

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
    

## Load data

In this section we are going to load the data prepared previous to this session, so you don't have to recompute the embeddings of Wikipedia articles with your own credits.


```python
!unzip -n vector_database_wikipedia_articles_embedded.zip
!ls -lh vector_database_wikipedia_articles_embedded.csv
```

    Archive:  vector_database_wikipedia_articles_embedded.zip
    -rw-r--r--@ 1 geng  staff   1.7G Jan 31 01:19 vector_database_wikipedia_articles_embedded.csv
    

Take a look at the data.


```python
import pandas, json
data = pandas.read_csv('../../data/vector_database_wikipedia_articles_embedded.csv')
data
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
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>text</th>
      <th>title_vector</th>
      <th>content_vector</th>
      <th>vector_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://simple.wikipedia.org/wiki/April</td>
      <td>April</td>
      <td>April is the fourth month of the year in the J...</td>
      <td>[0.001009464613161981, -0.020700545981526375, ...</td>
      <td>[-0.011253940872848034, -0.013491976074874401,...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://simple.wikipedia.org/wiki/August</td>
      <td>August</td>
      <td>August (Aug.) is the eighth month of the year ...</td>
      <td>[0.0009286514250561595, 0.000820168002974242, ...</td>
      <td>[0.0003609954728744924, 0.007262262050062418, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>https://simple.wikipedia.org/wiki/Art</td>
      <td>Art</td>
      <td>Art is a creative activity that expresses imag...</td>
      <td>[0.003393713850528002, 0.0061537534929811954, ...</td>
      <td>[-0.004959689453244209, 0.015772193670272827, ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>https://simple.wikipedia.org/wiki/A</td>
      <td>A</td>
      <td>A or a is the first letter of the English alph...</td>
      <td>[0.0153952119871974, -0.013759135268628597, 0....</td>
      <td>[0.024894846603274345, -0.022186409682035446, ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>https://simple.wikipedia.org/wiki/Air</td>
      <td>Air</td>
      <td>Air refers to the Earth's atmosphere. Air is a...</td>
      <td>[0.02224554680287838, -0.02044147066771984, -0...</td>
      <td>[0.021524671465158463, 0.018522677943110466, -...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>98295</td>
      <td>https://simple.wikipedia.org/wiki/Geneva</td>
      <td>Geneva</td>
      <td>Geneva (,  ,  ,  , ) is the second biggest cit...</td>
      <td>[-0.015773078426718712, 0.01737344264984131, 0...</td>
      <td>[0.008000412955880165, 0.02008531428873539, 0....</td>
      <td>24995</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>98316</td>
      <td>https://simple.wikipedia.org/wiki/Concubinage</td>
      <td>Concubinage</td>
      <td>Concubinage is the state of a woman in a relat...</td>
      <td>[-0.00519518880173564, 0.005898841191083193, 0...</td>
      <td>[-0.01736736111342907, -0.002740012714639306, ...</td>
      <td>24996</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>98318</td>
      <td>https://simple.wikipedia.org/wiki/Mistress%20%...</td>
      <td>Mistress (lover)</td>
      <td>A mistress is a man's long term female sexual ...</td>
      <td>[-0.023164259269833565, -0.02052430994808674, ...</td>
      <td>[-0.017878392711281776, -0.0004517830966506153...</td>
      <td>24997</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>98326</td>
      <td>https://simple.wikipedia.org/wiki/Eastern%20Front</td>
      <td>Eastern Front</td>
      <td>Eastern Front can be one of the following:\n\n...</td>
      <td>[-0.00681863259524107, 0.002171179046854377, 8...</td>
      <td>[-0.0019235472427681088, -0.004023272544145584...</td>
      <td>24998</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>98327</td>
      <td>https://simple.wikipedia.org/wiki/Italian%20Ca...</td>
      <td>Italian Campaign</td>
      <td>Italian Campaign can mean the following:\n\nTh...</td>
      <td>[-0.014151256531476974, -0.008553029969334602,...</td>
      <td>[-0.011758845299482346, -0.01346028596162796, ...</td>
      <td>24999</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 7 columns</p>
</div>




```python
title_vector_length = len(json.loads(data['title_vector'].iloc[0]))
content_vector_length = len(json.loads(data['content_vector'].iloc[0]))

print(title_vector_length, content_vector_length)
```

    1536 1536
    

### Create table and proxima vector index

Hologres stores data in __tables__ where each object is described by at least one vector. Our table will be called **articles** and each object will be described by both **title** and **content** vectors.

We will start with creating a table and create proxima indexes on both **title** and **content**, and then we will fill it with our precomputed embeddings.


```python
cursor.execute('CREATE EXTENSION IF NOT EXISTS proxima;')
create_proxima_table_sql = '''
BEGIN;
DROP TABLE IF EXISTS articles;
CREATE TABLE articles (
    id INT PRIMARY KEY NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector float4[] check(
        array_ndims(title_vector) = 1 and 
        array_length(title_vector, 1) = 1536
    ), -- define the vectors
    content_vector float4[] check(
        array_ndims(content_vector) = 1 and 
        array_length(content_vector, 1) = 1536
    ),
    vector_id INT
);

-- Create indexes for the vector fields.
call set_table_property(
    'articles',
    'proxima_vectors', 
    '{
        "title_vector":{"algorithm":"Graph","distance_method":"Euclidean","builder_params":{"min_flush_proxima_row_count" : 10}},
        "content_vector":{"algorithm":"Graph","distance_method":"Euclidean","builder_params":{"min_flush_proxima_row_count" : 10}}
    }'
);  

COMMIT;
'''

# Execute the SQL statements (will autocommit)
cursor.execute(create_proxima_table_sql)
```

### Upload data

Now let's upload the data to the Hologres cloud instance using [COPY statement](https://www.alibabacloud.com/help/en/hologres/latest/use-the-copy-statement-to-import-or-export-data). This might take 5-10 minutes according to the network bandwidth.


```python
import io

# Path to the unzipped CSV file
csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'

# In SQL, arrays are surrounded by {}, rather than []
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
```

The proxima index will be built in the background. We can do searching during this period but the query will be slow without the vector index. Use this command to wait for finish building the index.


```python
cursor.execute('vacuum articles;')
```


```python
# Check the collection size to make sure all the points have been stored
count_sql = "select count(*) from articles;"
cursor.execute(count_sql)
result = cursor.fetchone()
print(f"Count:{result[0]}")

```

    Count:25000
    

## Search data

Once the data is uploaded we will start querying the collection for the closest vectors. We may provide an additional parameter `vector_name` to switch from title to content based search. Since the precomputed embeddings were created with `text-embedding-3-small` OpenAI model we also have to use it during search.



```python
import openai
def query_knn(query, table_name, vector_name="title_vector", top_k=20):

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]

    # Convert the embedded_query to PostgreSQL compatible format
    embedded_query_pg = "{" + ",".join(map(str, embedded_query)) + "}"

    # Create SQL query
    query_sql = f"""
    SELECT id, url, title, pm_approx_euclidean_distance({vector_name},'{embedded_query_pg}'::float4[]) AS distance
    FROM {table_name}
    ORDER BY distance
    LIMIT {top_k};
    """
    # Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()

    return results
```


```python
query_results = query_knn("modern art in Europe", "Articles")
for i, result in enumerate(query_results):
    print(f"{i + 1}. {result[2]} (Score: {round(1 - result[3], 3)})")
```

    1. Museum of Modern Art (Score: 0.501)
    2. Western Europe (Score: 0.485)
    3. Renaissance art (Score: 0.479)
    4. Pop art (Score: 0.472)
    5. Northern Europe (Score: 0.461)
    6. Hellenistic art (Score: 0.458)
    7. Modernist literature (Score: 0.447)
    8. Art film (Score: 0.44)
    9. Central Europe (Score: 0.439)
    10. Art (Score: 0.437)
    11. European (Score: 0.437)
    12. Byzantine art (Score: 0.436)
    13. Postmodernism (Score: 0.435)
    14. Eastern Europe (Score: 0.433)
    15. Cubism (Score: 0.433)
    16. Europe (Score: 0.432)
    17. Impressionism (Score: 0.432)
    18. Bauhaus (Score: 0.431)
    19. Surrealism (Score: 0.429)
    20. Expressionism (Score: 0.429)
    


```python
# This time we'll query using content vector
query_results = query_knn("Famous battles in Scottish history", "Articles", "content_vector")
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
    


```python

```
