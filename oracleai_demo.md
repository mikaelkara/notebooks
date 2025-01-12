# Oracle AI Vector Search with Document Processing
Oracle AI Vector Search is designed for Artificial Intelligence (AI) workloads that allows you to query data based on semantics, rather than keywords.
One of the biggest benefits of Oracle AI Vector Search is that semantic search on unstructured data can be combined with relational search on business data in one single system.
This is not only powerful but also significantly more effective because you don't need to add a specialized vector database, eliminating the pain of data fragmentation between multiple systems.

In addition, your vectors can benefit from all of Oracle Database’s most powerful features, like the following:

 * [Partitioning Support](https://www.oracle.com/database/technologies/partitioning.html)
 * [Real Application Clusters scalability](https://www.oracle.com/database/real-application-clusters/)
 * [Exadata smart scans](https://www.oracle.com/database/technologies/exadata/software/smartscan/)
 * [Shard processing across geographically distributed databases](https://www.oracle.com/database/distributed-database/)
 * [Transactions](https://docs.oracle.com/en/database/oracle/oracle-database/23/cncpt/transactions.html)
 * [Parallel SQL](https://docs.oracle.com/en/database/oracle/oracle-database/21/vldbg/parallel-exec-intro.html#GUID-D28717E4-0F77-44F5-BB4E-234C31D4E4BA)
 * [Disaster recovery](https://www.oracle.com/database/data-guard/)
 * [Security](https://www.oracle.com/security/database-security/)
 * [Oracle Machine Learning](https://www.oracle.com/artificial-intelligence/database-machine-learning/)
 * [Oracle Graph Database](https://www.oracle.com/database/integrated-graph-database/)
 * [Oracle Spatial and Graph](https://www.oracle.com/database/spatial/)
 * [Oracle Blockchain](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_blockchain_table.html#GUID-B469E277-978E-4378-A8C1-26D3FF96C9A6)
 * [JSON](https://docs.oracle.com/en/database/oracle/oracle-database/23/adjsn/json-in-oracle-database.html)

This guide demonstrates how Oracle AI Vector Search can be used with Langchain to serve an end-to-end RAG pipeline. This guide goes through examples of:

 * Loading the documents from various sources using OracleDocLoader
 * Summarizing them within/outside the database using OracleSummary
 * Generating embeddings for them within/outside the database using OracleEmbeddings
 * Chunking them according to different requirements using Advanced Oracle Capabilities from OracleTextSplitter
 * Storing and Indexing them in a Vector Store and querying them for queries in OracleVS

If you are just starting with Oracle Database, consider exploring the [free Oracle 23 AI](https://www.oracle.com/database/free/#resources) which provides a great introduction to setting up your database environment. While working with the database, it is often advisable to avoid using the system user by default; instead, you can create your own user for enhanced security and customization. For detailed steps on user creation, refer to our [end-to-end guide](https://github.com/langchain-ai/langchain/blob/master/cookbook/oracleai_demo.ipynb) which also shows how to set up a user in Oracle. Additionally, understanding user privileges is crucial for managing database security effectively. You can learn more about this topic in the official [Oracle guide](https://docs.oracle.com/en/database/oracle/oracle-database/19/admqs/administering-user-accounts-and-security.html#GUID-36B21D72-1BBB-46C9-A0C9-F0D2A8591B8D) on administering user accounts and security.

### Prerequisites

Please install Oracle Python Client driver to use Langchain with Oracle AI Vector Search. 


```python
# pip install oracledb
```

### Create Demo User
First, create a demo user with all the required privileges. 


```python
import sys

import oracledb

# Update with your username, password, hostname, and service_name
username = ""
password = ""
dsn = ""

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            begin
                -- Drop user
                begin
                    execute immediate 'drop user testuser cascade';
                exception
                    when others then
                        dbms_output.put_line('Error dropping user: ' || SQLERRM);
                end;
                
                -- Create user and grant privileges
                execute immediate 'create user testuser identified by testuser';
                execute immediate 'grant connect, unlimited tablespace, create credential, create procedure, create any index to testuser';
                execute immediate 'create or replace directory DEMO_PY_DIR as ''/scratch/hroy/view_storage/hroy_devstorage/demo/orachain''';
                execute immediate 'grant read, write on directory DEMO_PY_DIR to public';
                execute immediate 'grant create mining model to testuser';
                
                -- Network access
                begin
                    DBMS_NETWORK_ACL_ADMIN.APPEND_HOST_ACE(
                        host => '*',
                        ace => xs$ace_type(privilege_list => xs$name_list('connect'),
                                           principal_name => 'testuser',
                                           principal_type => xs_acl.ptype_db)
                    );
                end;
            end;
            """
        )
        print("User setup done!")
    except Exception as e:
        print(f"User setup failed with error: {e}")
    finally:
        cursor.close()
    conn.close()
except Exception as e:
    print(f"Connection failed with error: {e}")
    sys.exit(1)
```

    Connection successful!
    User setup done!
    

## Process Documents using Oracle AI
Consider the following scenario: users possess documents stored either in an Oracle Database or a file system and intend to utilize this data with Oracle AI Vector Search powered by Langchain.

To prepare the documents for analysis, a comprehensive preprocessing workflow is necessary. Initially, the documents must be retrieved, summarized (if required), and chunked as needed. Subsequent steps involve generating embeddings for these chunks and integrating them into the Oracle AI Vector Store. Users can then conduct semantic searches on this data.

The Oracle AI Vector Search Langchain library encompasses a suite of document processing tools that facilitate document loading, chunking, summary generation, and embedding creation.

In the sections that follow, we will detail the utilization of Oracle AI Langchain APIs to effectively implement each of these processes.

### Connect to Demo User
The following sample code will show how to connect to Oracle Database. By default, python-oracledb runs in a ‘Thin’ mode which connects directly to Oracle Database. This mode does not need Oracle Client libraries. However, some additional functionality is available when python-oracledb uses them. Python-oracledb is said to be in ‘Thick’ mode when Oracle Client libraries are used. Both modes have comprehensive functionality supporting the Python Database API v2.0 Specification. See the following [guide](https://python-oracledb.readthedocs.io/en/latest/user_guide/appendix_a.html#featuresummary) that talks about features supported in each mode. You might want to switch to thick-mode if you are unable to use thin-mode.


```python
import sys

import oracledb

# please update with your username, password, hostname and service_name
username = ""
password = ""
dsn = ""

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")
    sys.exit(1)
```

    Connection successful!
    

### Populate a Demo Table
Create a demo table and insert some sample documents.


```python
try:
    cursor = conn.cursor()

    drop_table_sql = """drop table demo_tab"""
    cursor.execute(drop_table_sql)

    create_table_sql = """create table demo_tab (id number, data clob)"""
    cursor.execute(create_table_sql)

    insert_row_sql = """insert into demo_tab values (:1, :2)"""
    rows_to_insert = [
        (
            1,
            "If the answer to any preceding questions is yes, then the database stops the search and allocates space from the specified tablespace; otherwise, space is allocated from the database default shared temporary tablespace.",
        ),
        (
            2,
            "A tablespace can be online (accessible) or offline (not accessible) whenever the database is open.\nA tablespace is usually online so that its data is available to users. The SYSTEM tablespace and temporary tablespaces cannot be taken offline.",
        ),
        (
            3,
            "The database stores LOBs differently from other data types. Creating a LOB column implicitly creates a LOB segment and a LOB index. The tablespace containing the LOB segment and LOB index, which are always stored together, may be different from the tablespace containing the table.\nSometimes the database can store small amounts of LOB data in the table itself rather than in a separate LOB segment.",
        ),
    ]
    cursor.executemany(insert_row_sql, rows_to_insert)

    conn.commit()

    print("Table created and populated.")
    cursor.close()
except Exception as e:
    print("Table creation failed.")
    cursor.close()
    conn.close()
    sys.exit(1)
```

    Table created and populated.
    

With the inclusion of a demo user and a populated sample table, the remaining configuration involves setting up embedding and summary functionalities. Users are presented with multiple provider options, including local database solutions and third-party services such as Ocigenai, Hugging Face, and OpenAI. Should users opt for a third-party provider, they are required to establish credentials containing the necessary authentication details. Conversely, if selecting a database as the provider for embeddings, it is necessary to upload an ONNX model to the Oracle Database. No additional setup is required for summary functionalities when using the database option.

### Load ONNX Model

Oracle accommodates a variety of embedding providers, enabling users to choose between proprietary database solutions and third-party services such as OCIGENAI and HuggingFace. This selection dictates the methodology for generating and managing embeddings.

***Important*** : Should users opt for the database option, they must upload an ONNX model into the Oracle Database. Conversely, if a third-party provider is selected for embedding generation, uploading an ONNX model to Oracle Database is not required.

A significant advantage of utilizing an ONNX model directly within Oracle is the enhanced security and performance it offers by eliminating the need to transmit data to external parties. Additionally, this method avoids the latency typically associated with network or REST API calls.

Below is the example code to upload an ONNX model into Oracle Database:


```python
from langchain_community.embeddings.oracleai import OracleEmbeddings

# please update with your related information
# make sure that you have onnx file in the system
onnx_dir = "DEMO_PY_DIR"
onnx_file = "tinybert.onnx"
model_name = "demo_model"

try:
    OracleEmbeddings.load_onnx_model(conn, onnx_dir, onnx_file, model_name)
    print("ONNX model loaded.")
except Exception as e:
    print("ONNX model loading failed!")
    sys.exit(1)
```

    ONNX model loaded.
    

### Create Credential

When selecting third-party providers for generating embeddings, users are required to establish credentials to securely access the provider's endpoints.

***Important:*** No credentials are necessary when opting for the 'database' provider to generate embeddings. However, should users decide to utilize a third-party provider, they must create credentials specific to the chosen provider.

Below is an illustrative example:


```python
try:
    cursor = conn.cursor()
    cursor.execute(
        """
       declare
           jo json_object_t;
       begin
           -- HuggingFace
           dbms_vector_chain.drop_credential(credential_name  => 'HF_CRED');
           jo := json_object_t();
           jo.put('access_token', '<access_token>');
           dbms_vector_chain.create_credential(
               credential_name   =>  'HF_CRED',
               params            => json(jo.to_string));

           -- OCIGENAI
           dbms_vector_chain.drop_credential(credential_name  => 'OCI_CRED');
           jo := json_object_t();
           jo.put('user_ocid','<user_ocid>');
           jo.put('tenancy_ocid','<tenancy_ocid>');
           jo.put('compartment_ocid','<compartment_ocid>');
           jo.put('private_key','<private_key>');
           jo.put('fingerprint','<fingerprint>');
           dbms_vector_chain.create_credential(
               credential_name   => 'OCI_CRED',
               params            => json(jo.to_string));
       end;
       """
    )
    cursor.close()
    print("Credentials created.")
except Exception as ex:
    cursor.close()
    raise
```

### Load Documents
Users have the flexibility to load documents from either the Oracle Database, a file system, or both, by appropriately configuring the loader parameters. For comprehensive details on these parameters, please consult the [Oracle AI Vector Search Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html#GUID-73397E89-92FB-48ED-94BB-1AD960C4EA1F).

A significant advantage of utilizing OracleDocLoader is its capability to process over 150 distinct file formats, eliminating the need for multiple loaders for different document types. For a complete list of the supported formats, please refer to the [Oracle Text Supported Document Formats](https://docs.oracle.com/en/database/oracle/oracle-database/23/ccref/oracle-text-supported-document-formats.html).

Below is a sample code snippet that demonstrates how to use OracleDocLoader


```python
from langchain_community.document_loaders.oracleai import OracleDocLoader
from langchain_core.documents import Document

# loading from Oracle Database table
# make sure you have the table with this specification
loader_params = {}
loader_params = {
    "owner": "testuser",
    "tablename": "demo_tab",
    "colname": "data",
}

""" load the docs """
loader = OracleDocLoader(conn=conn, params=loader_params)
docs = loader.load()

""" verify """
print(f"Number of docs loaded: {len(docs)}")
# print(f"Document-0: {docs[0].page_content}") # content
```

    Number of docs loaded: 3
    

### Generate Summary
Now that the user loaded the documents, they may want to generate a summary for each document. The Oracle AI Vector Search Langchain library offers a suite of APIs designed for document summarization. It supports multiple summarization providers such as Database, OCIGENAI, HuggingFace, among others, allowing users to select the provider that best meets their needs. To utilize these capabilities, users must configure the summary parameters as specified. For detailed information on these parameters, please consult the [Oracle AI Vector Search Guide book](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html#GUID-EC9DDB58-6A15-4B36-BA66-ECBA20D2CE57).

***Note:*** The users may need to set proxy if they want to use some 3rd party summary generation providers other than Oracle's in-house and default provider: 'database'. If you don't have proxy, please remove the proxy parameter when you instantiate the OracleSummary.


```python
# proxy to be used when we instantiate summary and embedder object
proxy = ""
```

The following sample code will show how to generate summary:


```python
from langchain_community.utilities.oracleai import OracleSummary
from langchain_core.documents import Document

# using 'database' provider
summary_params = {
    "provider": "database",
    "glevel": "S",
    "numParagraphs": 1,
    "language": "english",
}

# get the summary instance
# Remove proxy if not required
summ = OracleSummary(conn=conn, params=summary_params, proxy=proxy)

list_summary = []
for doc in docs:
    summary = summ.get_summary(doc.page_content)
    list_summary.append(summary)

""" verify """
print(f"Number of Summaries: {len(list_summary)}")
# print(f"Summary-0: {list_summary[0]}") #content
```

    Number of Summaries: 3
    

### Split Documents
The documents may vary in size, ranging from small to very large. Users often prefer to chunk their documents into smaller sections to facilitate the generation of embeddings. A wide array of customization options is available for this splitting process. For comprehensive details regarding these parameters, please consult the [Oracle AI Vector Search Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html#GUID-4E145629-7098-4C7C-804F-FC85D1F24240).

Below is a sample code illustrating how to implement this:


```python
from langchain_community.document_loaders.oracleai import OracleTextSplitter
from langchain_core.documents import Document

# split by default parameters
splitter_params = {"normalize": "all"}

""" get the splitter instance """
splitter = OracleTextSplitter(conn=conn, params=splitter_params)

list_chunks = []
for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    list_chunks.extend(chunks)

""" verify """
print(f"Number of Chunks: {len(list_chunks)}")
# print(f"Chunk-0: {list_chunks[0]}") # content
```

    Number of Chunks: 3
    

### Generate Embeddings
Now that the documents are chunked as per requirements, the users may want to generate embeddings for these chunks. Oracle AI Vector Search provides multiple methods for generating embeddings, utilizing either locally hosted ONNX models or third-party APIs. For comprehensive instructions on configuring these alternatives, please refer to the [Oracle AI Vector Search Guide](https://docs.oracle.com/en/database/oracle/oracle-database/23/arpls/dbms_vector_chain1.html#GUID-C6439E94-4E86-4ECD-954E-4B73D53579DE).

***Note:*** Users may need to configure a proxy to utilize third-party embedding generation providers, excluding the 'database' provider that utilizes an ONNX model.


```python
# proxy to be used when we instantiate summary and embedder object
proxy = ""
```

The following sample code will show how to generate embeddings:


```python
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_core.documents import Document

# using ONNX model loaded to Oracle Database
embedder_params = {"provider": "database", "model": "demo_model"}

# get the embedding instance
# Remove proxy if not required
embedder = OracleEmbeddings(conn=conn, params=embedder_params, proxy=proxy)

embeddings = []
for doc in docs:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        embed = embedder.embed_query(chunk)
        embeddings.append(embed)

""" verify """
print(f"Number of embeddings: {len(embeddings)}")
# print(f"Embedding-0: {embeddings[0]}") # content
```

    Number of embeddings: 3
    

## Create Oracle AI Vector Store
Now that you know how to use Oracle AI Langchain library APIs individually to process the documents, let us show how to integrate with Oracle AI Vector Store to facilitate the semantic searches.

First, let's import all the dependencies.


```python
import sys

import oracledb
from langchain_community.document_loaders.oracleai import (
    OracleDocLoader,
    OracleTextSplitter,
)
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.utilities.oracleai import OracleSummary
from langchain_community.vectorstores import oraclevs
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
```

Next, let's combine all document processing stages together. Here is the sample code below:


```python
"""
In this sample example, we will use 'database' provider for both summary and embeddings.
So, we don't need to do the followings:
    - set proxy for 3rd party providers
    - create credential for 3rd party providers

If you choose to use 3rd party provider, 
please follow the necessary steps for proxy and credential.
"""

# oracle connection
# please update with your username, password, hostname, and service_name
username = ""
password = ""
dsn = ""

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")
    sys.exit(1)


# load onnx model
# please update with your related information
onnx_dir = "DEMO_PY_DIR"
onnx_file = "tinybert.onnx"
model_name = "demo_model"
try:
    OracleEmbeddings.load_onnx_model(conn, onnx_dir, onnx_file, model_name)
    print("ONNX model loaded.")
except Exception as e:
    print("ONNX model loading failed!")
    sys.exit(1)


# params
# please update necessary fields with related information
loader_params = {
    "owner": "testuser",
    "tablename": "demo_tab",
    "colname": "data",
}
summary_params = {
    "provider": "database",
    "glevel": "S",
    "numParagraphs": 1,
    "language": "english",
}
splitter_params = {"normalize": "all"}
embedder_params = {"provider": "database", "model": "demo_model"}

# instantiate loader, summary, splitter, and embedder
loader = OracleDocLoader(conn=conn, params=loader_params)
summary = OracleSummary(conn=conn, params=summary_params)
splitter = OracleTextSplitter(conn=conn, params=splitter_params)
embedder = OracleEmbeddings(conn=conn, params=embedder_params)

# process the documents
chunks_with_mdata = []
for id, doc in enumerate(docs, start=1):
    summ = summary.get_summary(doc.page_content)
    chunks = splitter.split_text(doc.page_content)
    for ic, chunk in enumerate(chunks, start=1):
        chunk_metadata = doc.metadata.copy()
        chunk_metadata["id"] = chunk_metadata["_oid"] + "$" + str(id) + "$" + str(ic)
        chunk_metadata["document_id"] = str(id)
        chunk_metadata["document_summary"] = str(summ[0])
        chunks_with_mdata.append(
            Document(page_content=str(chunk), metadata=chunk_metadata)
        )

""" verify """
print(f"Number of total chunks with metadata: {len(chunks_with_mdata)}")
```

    Connection successful!
    ONNX model loaded.
    Number of total chunks with metadata: 3
    

At this point, we have processed the documents and generated chunks with metadata. Next, we will create Oracle AI Vector Store with those chunks.

Here is the sample code how to do that:


```python
# create Oracle AI Vector Store
vectorstore = OracleVS.from_documents(
    chunks_with_mdata,
    embedder,
    client=conn,
    table_name="oravs",
    distance_strategy=DistanceStrategy.DOT_PRODUCT,
)

""" verify """
print(f"Vector Store Table: {vectorstore.table_name}")
```

    Vector Store Table: oravs
    

The example provided illustrates the creation of a vector store using the DOT_PRODUCT distance strategy. Users have the flexibility to employ various distance strategies with the Oracle AI Vector Store, as detailed in our [comprehensive guide](https://python.langchain.com/v0.1/docs/integrations/vectorstores/oracle/).

With embeddings now stored in vector stores, it is advisable to establish an index to enhance semantic search performance during query execution.

***Note*** Should you encounter an "insufficient memory" error, it is recommended to increase the  ***vector_memory_size*** in your database configuration

Below is a sample code snippet for creating an index:


```python
oraclevs.create_index(
    conn, vectorstore, params={"idx_name": "hnsw_oravs", "idx_type": "HNSW"}
)

print("Index created.")
```

This example demonstrates the creation of a default HNSW index on embeddings within the 'oravs' table. Users may adjust various parameters according to their specific needs. For detailed information on these parameters, please consult the [Oracle AI Vector Search Guide book](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/manage-different-categories-vector-indexes.html).

Additionally, various types of vector indices can be created to meet diverse requirements. More details can be found in our [comprehensive guide](https://python.langchain.com/v0.1/docs/integrations/vectorstores/oracle/).


## Perform Semantic Search
All set!

We have successfully processed the documents and stored them in the vector store, followed by the creation of an index to enhance query performance. We are now prepared to proceed with semantic searches.

Below is the sample code for this process:


```python
query = "What is Oracle AI Vector Store?"
filter = {"document_id": ["1"]}

# Similarity search without a filter
print(vectorstore.similarity_search(query, 1))

# Similarity search with a filter
print(vectorstore.similarity_search(query, 1, filter=filter))

# Similarity search with relevance score
print(vectorstore.similarity_search_with_score(query, 1))

# Similarity search with relevance score with filter
print(vectorstore.similarity_search_with_score(query, 1, filter=filter))

# Max marginal relevance search
print(vectorstore.max_marginal_relevance_search(query, 1, fetch_k=20, lambda_mult=0.5))

# Max marginal relevance search with filter
print(
    vectorstore.max_marginal_relevance_search(
        query, 1, fetch_k=20, lambda_mult=0.5, filter=filter
    )
)
```

    [Document(page_content='The database stores LOBs differently from other data types. Creating a LOB column implicitly creates a LOB segment and a LOB index. The tablespace containing the LOB segment and LOB index, which are always stored together, may be different from the tablespace containing the table. Sometimes the database can store small amounts of LOB data in the table itself rather than in a separate LOB segment.', metadata={'_oid': '662f2f257677f3c2311a8ff999fd34e5', '_rowid': 'AAAR/xAAEAAAAAnAAC', 'id': '662f2f257677f3c2311a8ff999fd34e5$3$1', 'document_id': '3', 'document_summary': 'Sometimes the database can store small amounts of LOB data in the table itself rather than in a separate LOB segment.\n\n'})]
    []
    [(Document(page_content='The database stores LOBs differently from other data types. Creating a LOB column implicitly creates a LOB segment and a LOB index. The tablespace containing the LOB segment and LOB index, which are always stored together, may be different from the tablespace containing the table. Sometimes the database can store small amounts of LOB data in the table itself rather than in a separate LOB segment.', metadata={'_oid': '662f2f257677f3c2311a8ff999fd34e5', '_rowid': 'AAAR/xAAEAAAAAnAAC', 'id': '662f2f257677f3c2311a8ff999fd34e5$3$1', 'document_id': '3', 'document_summary': 'Sometimes the database can store small amounts of LOB data in the table itself rather than in a separate LOB segment.\n\n'}), 0.055675752460956573)]
    []
    [Document(page_content='If the answer to any preceding questions is yes, then the database stops the search and allocates space from the specified tablespace; otherwise, space is allocated from the database default shared temporary tablespace.', metadata={'_oid': '662f2f253acf96b33b430b88699490a2', '_rowid': 'AAAR/xAAEAAAAAnAAA', 'id': '662f2f253acf96b33b430b88699490a2$1$1', 'document_id': '1', 'document_summary': 'If the answer to any preceding questions is yes, then the database stops the search and allocates space from the specified tablespace; otherwise, space is allocated from the database default shared temporary tablespace.\n\n'})]
    [Document(page_content='If the answer to any preceding questions is yes, then the database stops the search and allocates space from the specified tablespace; otherwise, space is allocated from the database default shared temporary tablespace.', metadata={'_oid': '662f2f253acf96b33b430b88699490a2', '_rowid': 'AAAR/xAAEAAAAAnAAA', 'id': '662f2f253acf96b33b430b88699490a2$1$1', 'document_id': '1', 'document_summary': 'If the answer to any preceding questions is yes, then the database stops the search and allocates space from the specified tablespace; otherwise, space is allocated from the database default shared temporary tablespace.\n\n'})]
    
