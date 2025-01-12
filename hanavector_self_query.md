# SAP HANA Cloud Vector Engine

For more information on how to setup the SAP HANA vetor store, take a look at the [documentation](/docs/integrations/vectorstores/sap_hanavector.ipynb).

We use the same setup here:


```python
import os

# Use OPENAI_API_KEY env variable
# os.environ["OPENAI_API_KEY"] = "Your OpenAI API key"
from hdbcli import dbapi

# Use connection settings from the environment
connection = dbapi.connect(
    address=os.environ.get("HANA_DB_ADDRESS"),
    port=os.environ.get("HANA_DB_PORT"),
    user=os.environ.get("HANA_DB_USER"),
    password=os.environ.get("HANA_DB_PASSWORD"),
    autocommit=True,
    sslValidateCertificate=False,
)
```

To be able to self query with good performance we create additional metadata fields
for our vectorstore table in HANA:


```python
# Create custom table with attribute
cur = connection.cursor()
cur.execute("DROP TABLE LANGCHAIN_DEMO_SELF_QUERY", ignoreErrors=True)
cur.execute(
    (
        """CREATE TABLE "LANGCHAIN_DEMO_SELF_QUERY"  (
        "name" NVARCHAR(100), "is_active" BOOLEAN, "id" INTEGER, "height" DOUBLE,
        "VEC_TEXT" NCLOB, 
        "VEC_META" NCLOB, 
        "VEC_VECTOR" REAL_VECTOR
        )"""
    )
)
```

Let's add some documents.


```python
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Prepare some test documents
docs = [
    Document(
        page_content="First",
        metadata={"name": "adam", "is_active": True, "id": 1, "height": 10.0},
    ),
    Document(
        page_content="Second",
        metadata={"name": "bob", "is_active": False, "id": 2, "height": 5.7},
    ),
    Document(
        page_content="Third",
        metadata={"name": "jane", "is_active": True, "id": 3, "height": 2.4},
    ),
]

db = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="LANGCHAIN_DEMO_SELF_QUERY",
    specific_metadata_columns=["name", "is_active", "id", "height"],
)

# Delete already existing documents from the table
db.delete(filter={})
db.add_documents(docs)
```

## Self querying

Now for the main act: here is how to construct a SelfQueryRetriever for HANA vectorstore:


```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.hanavector import HanaTranslator
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

metadata_field_info = [
    AttributeInfo(
        name="name",
        description="The name of the person",
        type="string",
    ),
    AttributeInfo(
        name="is_active",
        description="Whether the person is active",
        type="boolean",
    ),
    AttributeInfo(
        name="id",
        description="The ID of the person",
        type="integer",
    ),
    AttributeInfo(
        name="height",
        description="The height of the person",
        type="float",
    ),
]

document_content_description = "A collection of persons"

hana_translator = HanaTranslator()

retriever = SelfQueryRetriever.from_llm(
    llm,
    db,
    document_content_description,
    metadata_field_info,
    structured_query_translator=hana_translator,
)
```

Let's use this retriever to prepare a (self) query for a person:


```python
query_prompt = "Which person is not active?"

docs = retriever.invoke(input=query_prompt)
for doc in docs:
    print("-" * 80)
    print(doc.page_content, " ", doc.metadata)
```

We can also take a look at how the query is being constructed:


```python
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
)
output_parser = StructuredQueryOutputParser.from_components()
query_constructor = prompt | llm | output_parser

sq = query_constructor.invoke(input=query_prompt)

print("Structured query: ", sq)

print("Translated for hana vector store: ", hana_translator.visit_structured_query(sq))
```
