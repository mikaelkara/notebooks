## Setup Environment

### Python Modules

Install the following Python modules:

```bash
pip install ipykernel python-dotenv cassio pandas langchain_openai langchain langchain-community langchainhub langchain_experimental openai-multi-tool-use-parallel-patch
```

### Load the `.env` File

Connection is via `cassio` using `auto=True` parameter, and the notebook uses OpenAI. You should create a `.env` file accordingly.

For Cassandra, set:
```bash
CASSANDRA_CONTACT_POINTS
CASSANDRA_USERNAME
CASSANDRA_PASSWORD
CASSANDRA_KEYSPACE
```

For Astra, set:
```bash
ASTRA_DB_APPLICATION_TOKEN
ASTRA_DB_DATABASE_ID
ASTRA_DB_KEYSPACE
```

For example:

```bash
# Connection to Astra:
ASTRA_DB_DATABASE_ID=a1b2c3d4-...
ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
ASTRA_DB_KEYSPACE=notebooks

# Also set 
OPENAI_API_KEY=sk-....
```

(You may also modify the below code to directly connect with `cassio`.)


```python
from dotenv import load_dotenv

load_dotenv(override=True)
```

### Connect to Cassandra


```python
import os

import cassio

cassio.init(auto=True)
session = cassio.config.resolve_session()
if not session:
    raise Exception(
        "Check environment configuration or manually configure cassio connection parameters"
    )

keyspace = os.environ.get(
    "ASTRA_DB_KEYSPACE", os.environ.get("CASSANDRA_KEYSPACE", None)
)
if not keyspace:
    raise ValueError("a KEYSPACE environment variable must be set")

session.set_keyspace(keyspace)
```

## Setup Database

This needs to be done one time only!

### Download Data

The dataset used is from Kaggle, the [Environmental Sensor Telemetry Data](https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k?select=iot_telemetry_data.csv). The next cell will download and unzip the data into a Pandas dataframe. The following cell is instructions to download manually. 

The net result of this section is you should have a Pandas dataframe variable `df`.

#### Download Automatically


```python
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests

datasetURL = "https://storage.googleapis.com/kaggle-data-sets/788816/1355729/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240404%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240404T115828Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=2849f003b100eb9dcda8dd8535990f51244292f67e4f5fad36f14aa67f2d4297672d8fe6ff5a39f03a29cda051e33e95d36daab5892b8874dcd5a60228df0361fa26bae491dd4371f02dd20306b583a44ba85a4474376188b1f84765147d3b4f05c57345e5de883c2c29653cce1f3755cd8e645c5e952f4fb1c8a735b22f0c811f97f7bce8d0235d0d3731ca8ab4629ff381f3bae9e35fc1b181c1e69a9c7913a5e42d9d52d53e5f716467205af9c8a3cc6746fc5352e8fbc47cd7d18543626bd67996d18c2045c1e475fc136df83df352fa747f1a3bb73e6ba3985840792ec1de407c15836640ec96db111b173bf16115037d53fdfbfd8ac44145d7f9a546aa"

response = requests.get(datasetURL)
if response.status_code == 200:
    zip_file = ZipFile(BytesIO(response.content))
    csv_file_name = zip_file.namelist()[0]
else:
    print("Failed to download the file")

with zip_file.open(csv_file_name) as csv_file:
    df = pd.read_csv(csv_file)
```

#### Download Manually

You can download the `.zip` file and unpack the `.csv` contained within. Comment in the next line, and adjust the path to this `.csv` file appropriately.


```python
# df = pd.read_csv("/path/to/iot_telemetry_data.csv")
```

### Load Data into Cassandra

This section assumes the existence of a dataframe `df`, the following cell validates its structure. The Download section above creates this object.


```python
assert df is not None, "Dataframe 'df' must be set"
expected_columns = [
    "ts",
    "device",
    "co",
    "humidity",
    "light",
    "lpg",
    "motion",
    "smoke",
    "temp",
]
assert all(
    [column in df.columns for column in expected_columns]
), "DataFrame does not have the expected columns"
```

Create and load tables:


```python
from datetime import UTC, datetime

from cassandra.query import BatchStatement

# Create sensors table
table_query = """
CREATE TABLE IF NOT EXISTS iot_sensors (
    device text,
    conditions text,
    room text,
    PRIMARY KEY (device)
)
WITH COMMENT = 'Environmental IoT room sensor metadata.';
"""
session.execute(table_query)

pstmt = session.prepare(
    """
INSERT INTO iot_sensors (device, conditions, room)
VALUES (?, ?, ?)
"""
)

devices = [
    ("00:0f:00:70:91:0a", "stable conditions, cooler and more humid", "room 1"),
    ("1c:bf:ce:15:ec:4d", "highly variable temperature and humidity", "room 2"),
    ("b8:27:eb:bf:9d:51", "stable conditions, warmer and dryer", "room 3"),
]

for device, conditions, room in devices:
    session.execute(pstmt, (device, conditions, room))

print("Sensors inserted successfully.")

# Create data table
table_query = """
CREATE TABLE IF NOT EXISTS iot_data (
    day text,
    device text,
    ts timestamp,
    co double,
    humidity double,
    light boolean,
    lpg double,
    motion boolean,
    smoke double,
    temp double,
    PRIMARY KEY ((day, device), ts)
)
WITH COMMENT = 'Data from environmental IoT room sensors. Columns include device identifier, timestamp (ts) of the data collection, carbon monoxide level (co), relative humidity, light presence, LPG concentration, motion detection, smoke concentration, and temperature (temp). Data is partitioned by day and device.';
"""
session.execute(table_query)

pstmt = session.prepare(
    """
INSERT INTO iot_data (day, device, ts, co, humidity, light, lpg, motion, smoke, temp)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
)


def insert_data_batch(name, group):
    batch = BatchStatement()
    day, device = name
    print(f"Inserting batch for day: {day}, device: {device}")

    for _, row in group.iterrows():
        timestamp = datetime.fromtimestamp(row["ts"], UTC)
        batch.add(
            pstmt,
            (
                day,
                row["device"],
                timestamp,
                row["co"],
                row["humidity"],
                row["light"],
                row["lpg"],
                row["motion"],
                row["smoke"],
                row["temp"],
            ),
        )

    session.execute(batch)


# Convert columns to appropriate types
df["light"] = df["light"] == "true"
df["motion"] = df["motion"] == "true"
df["ts"] = df["ts"].astype(float)
df["day"] = df["ts"].apply(
    lambda x: datetime.fromtimestamp(x, UTC).strftime("%Y-%m-%d")
)

grouped_df = df.groupby(["day", "device"])

for name, group in grouped_df:
    insert_data_batch(name, group)

print("Data load complete")
```


```python
print(session.keyspace)
```

## Load the Tools

Python `import` statements for the demo:


```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits.cassandra_database.toolkit import (
    CassandraDatabaseToolkit,
)
from langchain_community.tools.cassandra_database.prompt import QUERY_PATH_PROMPT
from langchain_community.tools.cassandra_database.tool import (
    GetSchemaCassandraDatabaseTool,
    GetTableDataCassandraDatabaseTool,
    QueryCassandraDatabaseTool,
)
from langchain_community.utilities.cassandra_database import CassandraDatabase
from langchain_openai import ChatOpenAI
```

The `CassandraDatabase` object is loaded from `cassio`, though it does accept a `Session`-type parameter as an alternative.


```python
# Create a CassandraDatabase instance
db = CassandraDatabase(include_tables=["iot_sensors", "iot_data"])

# Create the Cassandra Database tools
query_tool = QueryCassandraDatabaseTool(db=db)
schema_tool = GetSchemaCassandraDatabaseTool(db=db)
select_data_tool = GetTableDataCassandraDatabaseTool(db=db)
```

The tools can be invoked directly:


```python
# Test the tools
print("Executing a CQL query:")
query = "SELECT * FROM iot_sensors LIMIT 5;"
result = query_tool.run({"query": query})
print(result)

print("\nGetting the schema for a keyspace:")
schema = schema_tool.run({"keyspace": keyspace})
print(schema)

print("\nGetting data from a table:")
table = "iot_data"
predicate = "day = '2020-07-14' and device = 'b8:27:eb:bf:9d:51'"
data = select_data_tool.run(
    {"keyspace": keyspace, "table": table, "predicate": predicate, "limit": 5}
)
print(data)
```

## Agent Configuration


```python
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
```


```python
from langchain import hub

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
toolkit = CassandraDatabaseToolkit(db=db)

# context = toolkit.get_context()
# tools = toolkit.get_tools()
tools = [schema_tool, select_data_tool, repl_tool]

input = (
    QUERY_PATH_PROMPT
    + f"""

Here is your task: In the {keyspace} keyspace, find the total number of times the temperature of each device has exceeded 23 degrees on July 14, 2020.
 Create a summary report including the name of the room. Use Pandas if helpful.
"""
)

prompt = hub.pull("hwchase17/openai-tools-agent")

# messages = [
#     HumanMessagePromptTemplate.from_template(input),
#     AIMessage(content=QUERY_PATH_PROMPT),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ]

# prompt = ChatPromptTemplate.from_messages(messages)
# print(prompt)

# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

print("Available tools:")
for tool in tools:
    print("\t" + tool.name + " - " + tool.description + " - " + str(tool))
```


```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({"input": input})

print(response["output"])
```
