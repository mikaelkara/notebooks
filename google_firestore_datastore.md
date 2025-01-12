# Google Firestore (Datastore Mode)

> [Google Cloud Firestore in Datastore](https://cloud.google.com/datastore) is a serverless document-oriented database that scales to meet any demand. Extend your database application to build AI-powered experiences leveraging `Datastore's` Langchain integrations.

This notebook goes over how to use [Google Cloud Firestore in Datastore](https://cloud.google.com/datastore) to store chat message history with the `DatastoreChatMessageHistory` class.

Learn more about the package on [GitHub](https://github.com/googleapis/langchain-google-datastore-python/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googleapis/langchain-google-datastore-python/blob/main/docs/chat_message_history.ipynb)

## Before You Begin

To run this notebook, you will need to do the following:

* [Create a Google Cloud Project](https://developers.google.com/workspace/guides/create-project)
* [Enable the Datastore API](https://console.cloud.google.com/flows/enableapi?apiid=datastore.googleapis.com)
* [Create a Datastore database](https://cloud.google.com/datastore/docs/manage-databases)

After confirming access to the database in the runtime environment of this notebook, filling the following values and run the cell before running example scripts.

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

### API Enablement
The `langchain-google-datastore` package requires that you [enable the Datastore API](https://console.cloud.google.com/flows/enableapi?apiid=datastore.googleapis.com) in your Google Cloud Project.


```python
# enable Datastore API
!gcloud services enable datastore.googleapis.com
```

## Basic Usage

### DatastoreChatMessageHistory

To initialize the `DatastoreChatMessageHistory` class you need to provide only 3 things:

1. `session_id` - A unique identifier string that specifies an id for the session.
1. `kind` - The name of the Datastore kind to write into. This is an optional value and by default, it will use `ChatHistory` as the kind.
1. `collection` - The single `/`-delimited path to a Datastore collection.


```python
from langchain_google_datastore import DatastoreChatMessageHistory

chat_history = DatastoreChatMessageHistory(
    session_id="user-session-id", collection="HistoryMessages"
)

chat_history.add_user_message("Hi!")
chat_history.add_ai_message("How can I help you?")
```


```python
chat_history.messages
```

#### Cleaning up
When the history of a specific session is obsolete and can be deleted from the database and memory, it can be done the following way.

**Note:** Once deleted, the data is no longer stored in Datastore and is gone forever.


```python
chat_history.clear()
```

### Custom Client

The client is created by default using the available environment variables. A [custom client](https://cloud.google.com/python/docs/reference/datastore/latest/client) can be passed to the constructor.


```python
from google.auth import compute_engine
from google.cloud import datastore

client = datastore.Client(
    project="project-custom",
    database="non-default-database",
    credentials=compute_engine.Credentials(),
)

history = DatastoreChatMessageHistory(
    session_id="session-id", collection="History", client=client
)

history.add_user_message("New message")

history.messages

history.clear()
```
