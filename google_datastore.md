# Google Firestore in Datastore Mode

> [Firestore in Datastore Mode](https://cloud.google.com/datastore) is a NoSQL document database built for automatic scaling, high performance and ease of application development. Extend your database application to build AI-powered experiences leveraging Datastore's Langchain integrations.

This notebook goes over how to use [Firestore in Datastore Mode](https://cloud.google.com/datastore) to [save, load and delete langchain documents](/docs/how_to#document-loaders) with `DatastoreLoader` and `DatastoreSaver`.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-datastore-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-datastore-python/blob/main/docs/document_loader.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Datastore API](https://console.cloud.google.com/flows/enableapi?apiid=datastore.googleapis.com)
* [Create a Firestore in Datastore Mode database](https://cloud.google.com/datastore/docs/manage-databases)

After confirmed access to database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.

### 🦜🔗 Library Installation

The integration lives in its own `langchain-google-datastore` package, so we need to install it.


```python
%pip install -upgrade --quiet langchain-google-datastore
```

**Colab only**: Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.


```python
# # Automatically restart kernel after installs so that your environment can access the new packages
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

### ☁ Set Your Google Cloud Project
Set your Google Cloud project so that you can leverage Google Cloud resources within this notebook.

If you don't know your project ID, try the following:

* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).


```python
# @markdown Please fill in the value below with your Google Cloud project ID and then run the cell.

PROJECT_ID = "my-project-id"  # @param {type:"string"}

# Set the project id
!gcloud config set project {PROJECT_ID}
```

### 🔐 Authentication

Authenticate to Google Cloud as the IAM user logged into this notebook in order to access your Google Cloud Project.

- If you are using Colab to run this notebook, use the cell below and continue.
- If you are using Vertex AI Workbench, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).


```python
from google.colab import auth

auth.authenticate_user()
```

## Basic Usage

### Save documents

Save langchain documents with `DatastoreSaver.upsert_documents(<documents>)`. By default it will try to extract the entity key from the `key` in the Document metadata.


```python
from langchain_core.documents import Document
from langchain_google_datastore import DatastoreSaver

saver = DatastoreSaver()

data = [Document(page_content="Hello, World!")]
saver.upsert_documents(data)
```

#### Save documents without key

If a `kind` is specified the documents will be stored with an auto generated id.


```python
saver = DatastoreSaver("MyKind")

saver.upsert_documents(data)
```

### Load documents via Kind

Load langchain documents with `DatastoreLoader.load()` or `DatastoreLoader.lazy_load()`. `lazy_load` returns a generator that only queries database during the iteration. To initialize `DatastoreLoader` class you need to provide:
1. `source` - The source to load the documents. It can be an instance of Query or the name of the Datastore kind to read from.


```python
from langchain_google_datastore import DatastoreLoader

loader = DatastoreLoader("MyKind")
data = loader.load()
```

### Load documents via query

Other than loading documents from kind, we can also choose to load documents from query. For example:


```python
from google.cloud import datastore

client = datastore.Client(database="non-default-db", namespace="custom_namespace")
query_load = client.query(kind="MyKind")
query_load.add_filter("region", "=", "west_coast")

loader_document = DatastoreLoader(query_load)

data = loader_document.load()
```

### Delete documents

Delete a list of langchain documents from Datastore with `DatastoreSaver.delete_documents(<documents>)`.


```python
saver = DatastoreSaver()

saver.delete_documents(data)

keys_to_delete = [
    ["Kind1", "identifier"],
    ["Kind2", 123],
    ["Kind3", "identifier", "NestedKind", 456],
]
# The Documents will be ignored and only the document ids will be used.
saver.delete_documents(data, keys_to_delete)
```

## Advanced Usage

### Load documents with customized document page content & metadata

The arguments of `page_content_properties` and `metadata_properties` will specify the Entity properties to be written into LangChain Document `page_content` and `metadata`.


```python
loader = DatastoreLoader(
    source="MyKind",
    page_content_fields=["data_field"],
    metadata_fields=["metadata_field"],
)

data = loader.load()
```

### Customize Page Content Format

When the `page_content` contains only one field the information will be the field value only. Otherwise the `page_content` will be in JSON format.

### Customize Connection & Authentication


```python
from google.auth import compute_engine
from google.cloud.firestore import Client

client = Client(database="non-default-db", creds=compute_engine.Credentials())
loader = DatastoreLoader(
    source="foo",
    client=client,
)
```
