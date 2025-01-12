# Intro
This notebook is an example on how you can use SingleStoreDB vector storage and functions to build an interactive Q&A application with ChatGPT. If you start a [Trial](https://www.singlestore.com/cloud-trial/) in SingleStoreDB, you can find the same notebook in our sample notebooks with native connection.

## First let's talk directly to ChatGPT and try and get back a response


```python
!pip install openai --quiet

```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m23.1.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpython3.11 -m pip install --upgrade pip[0m
    


```python
import openai

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

```

## Let's connect to OpenAI and see the result we get when asking for a date beyond 2021


```python
openai.api_key = 'OPENAI API KEY'

response = openai.ChatCompletion.create(
  model=GPT_MODEL,
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the gold medal for curling in Olymics 2022?"},
    ]
)

print(response['choices'][0]['message']['content'])

```

    I'm sorry, I cannot provide information about events that have not occurred yet. The Winter Olympics 2022 will be held in Beijing, China from February 4 to 20, 2022. The curling events will take place during this time and the results will not be known until after the competition has concluded.
    

# Get the data about Winter Olympics and provide the information to ChatGPT as context

## 1. Setup


```python
!pip install matplotlib plotly.express scikit-learn tabulate tiktoken wget --quiet

```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m23.1.2[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpython3.11 -m pip install --upgrade pip[0m
    


```python
import pandas as pd
import os
import wget
import ast

```

## Step 1 - Grab the data from CSV and prepare it


```python
# download pre-chunked text and pre-computed embeddings
# this file is ~200 MB, so may take a minute depending on your connection speed
embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
file_path = "winter_olympics_2022.csv"

if not os.path.exists(file_path):
    wget.download(embeddings_path, file_path)
    print("File downloaded successfully.")
else:
    print("File already exists in the local file system.")

```

    File downloaded successfully.
    


```python
df = pd.read_csv(
    "winter_olympics_2022.csv"
)

# convert embeddings from CSV str type back to list type
df['embedding'] = df['embedding'].apply(ast.literal_eval)

```


```python
df

```


```python
df.info(show_counts=True)

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6059 entries, 0 to 6058
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   text       6059 non-null   object
     1   embedding  6059 non-null   object
    dtypes: object(2)
    memory usage: 94.8+ KB
    

## 2. Set up SingleStore DB


```python
import singlestoredb as s2

conn = s2.connect("<user>:<Password>@<host>:3306/")

cur = conn.cursor()

```


```python
# Create database
stmt = """
    CREATE DATABASE IF NOT EXISTS winter_wikipedia2;
"""

cur.execute(stmt)

```




    1




```python
#create table
stmt = """
CREATE TABLE IF NOT EXISTS winter_wikipedia2.winter_olympics_2022 (
    id INT PRIMARY KEY,
    text TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
    embedding BLOB
);"""

cur.execute(stmt)

```




    0



## 3. Populate the Table with our dataframe df and use JSON_ARRAY_PACK to compact it


```python
%%time

# Prepare the statement
stmt = """
    INSERT INTO winter_wikipedia2.winter_olympics_2022 (
        id,
        text,
        embedding
    )
    VALUES (
        %s,
        %s,
        JSON_ARRAY_PACK_F64(%s)
    )
"""

# Convert the DataFrame to a NumPy record array
record_arr = df.to_records(index=True)

# Set the batch size
batch_size = 1000

# Iterate over the rows of the record array in batches
for i in range(0, len(record_arr), batch_size):
    batch = record_arr[i:i+batch_size]
    values = [(row[0], row[1], str(row[2])) for row in batch]
    cur.executemany(stmt, values)

```

    CPU times: user 8.79 s, sys: 4.63 s, total: 13.4 s
    Wall time: 11min 4s
    

## 4. Do a semantic search with the same question from above and use the response to send to OpenAI again



```python
from utils.embeddings_utils import get_embedding

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple:
    """Returns a list of strings and relatednesses, sorted from most related to least."""

    # Get the embedding of the query.
    query_embedding_response = get_embedding(query, EMBEDDING_MODEL)

    # Create the SQL statement.
    stmt = """
        SELECT
            text,
            DOT_PRODUCT_F64(JSON_ARRAY_PACK_F64(%s), embedding) AS score
        FROM winter_wikipedia2.winter_olympics_2022
        ORDER BY score DESC
        LIMIT %s
    """

    # Execute the SQL statement.
    results = cur.execute(stmt, [str(query_embedding_response), top_n])

    # Fetch the results
    results = cur.fetchall()

    strings = []
    relatednesses = []

    for row in results:
        strings.append(row[0])
        relatednesses.append(row[1])

    # Return the results.
    return strings[:top_n], relatednesses[:top_n]

```


```python
from tabulate import tabulate

strings, relatednesses = strings_ranked_by_relatedness(
    "curling gold medal",
    df,
    top_n=5
)

for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(tabulate([[string]], headers=['Result'], tablefmt='fancy_grid'))

```

## 5. Send the right context to ChatGPT for a more accurate answer


```python
import tiktoken

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from SingleStoreDB."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df, "winter_olympics_2022")
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a table of relevant texts and embeddings in SingleStoreDB."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message

```

## 6. Get an answer from Chat GPT


```python
from pprint import pprint

answer = ask('Who won the gold medal for curling in Olymics 2022?')

pprint(answer)

```

    ("There were three curling events at the 2022 Winter Olympics: men's, women's, "
     'and mixed doubles. The gold medalists for each event are:\n'
     '\n'
     "- Men's: Sweden (Niklas Edin, Oskar Eriksson, Rasmus Wranå, Christoffer "
     'Sundgren, Daniel Magnusson)\n'
     "- Women's: Great Britain (Eve Muirhead, Vicky Wright, Jennifer Dodds, Hailey "
     'Duff, Mili Smith)\n'
     '- Mixed doubles: Italy (Stefania Constantini, Amos Mosaner)')
    
