# Vertex AI Search and Conversation Data Store Status Checker

_Using Google Cloud Discovery Engine APIs for Vertex AI Search and Conversation_

<table align="left">

  <td>
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/conversation/data-store-status-checker/data_store_checker.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo"> Run in Colab
    </a>
  </td>
  <td>
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/conversation/data-store-status-checker/data_store_checker.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
      View on GitHub
    </a>
  </td>
  <td>
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/conversation/data-store-status-checker/data_store_checker.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo">
      Open in Vertex AI Workbench
    </a>
  </td>
</table>
<br><br><br>

## What is a Data Store?

A [Data Store](https://cloud.google.com/generative-ai-app-builder/docs/create-datastore-ingest) in [Vertex AI Search and Conversation](https://cloud.google.com/generative-ai-app-builder) is a collection of websites or documents, both structured and unstructured, that can be indexed for search and retrieval actions.

Data Stores are the fundamental building block behind [Vertex AI Search](https://cloud.google.com/enterprise-search) and [Vertex AI Conversation](https://cloud.google.com/generative-ai-app-builder/docs/agent-usage).

## Data Store Indexing Time

With each website or set of documents added, the Data Store needs to index the site and/or documents in order for them to be searchable. This can take up to 4 hours for new data store web content to be indexed.

Using the attached example notebook, you can query your Data Store ID to see if indexing is complete.
Once complete, you can additionally use the notebook to search your Data Store for specific pages or documents.

## Limitations

This notebook cannot be used for `Search` applications that use standard website indexing.


| | |
|-|-|
|Author(s) | [Patrick Marlow](https://github.com/kmaphoenix) |

## Objective

Simple notebook that uses the Cloud Discovery Engine API to check a Data Store for indexed docs.

---

This notebook utilizes the [`google-cloud-discoveryengine`](https://cloud.google.com/python/docs/reference/discoveryengine/latest) Python library.  
This notebook allows the user to perform the following tasks:

- ✅ Check Indexing Status of given Data Store ID.
- ✅ List all documents in a given Data Store ID.
- ✅ List all indexed URLs for a given Data Store ID
- ✅ Search all indexed URLs for a specific URL within a given Data Store ID.

---

**References:**

- [Google Cloud Discovery Engine API](https://cloud.google.com/python/docs/reference/discoveryengine/latest)

---

- Author: Patrick Marlow
- Created: 07/17/2023

---


# Install PreReqs and Authentication



```
%pip install --upgrade google-cloud-discoveryengine humanize
```


```
import sys

if "google.colab" in sys.modules:
    from google.auth import default
    from google.colab import auth

    auth.authenticate_user()
    creds, _ = default()
else:
    # Otherwise, attempt to discover local credentials as described on https://cloud.google.com/docs/authentication/application-default-credentials
    pass
```

# Helper Methods

Run the below cell to setup the helper methods for this notebook.



```
import re
import time

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1beta as discoveryengine
import humanize


def _call_list_documents(
    project_id: str, location: str, datastore_id: str, page_token: str | None = None
) -> discoveryengine.ListDocumentsResponse:
    """Build the List Docs Request payload."""
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    request = discoveryengine.ListDocumentsRequest(
        parent=client.branch_path(project_id, location, datastore_id, "default_branch"),
        page_size=1000,
        page_token=page_token,
    )

    return client.list_documents(request=request)


def list_documents(
    project_id: str, location: str, datastore_id: str, rate_limit: int = 1
) -> list[discoveryengine.Document]:
    """Gets a list of docs in a datastore."""

    res = _call_list_documents(project_id, location, datastore_id)

    # setup the list with the first batch of docs
    docs = res.documents

    while res.next_page_token:
        # implement a rate_limit to prevent quota exhaustion
        time.sleep(rate_limit)

        res = _call_list_documents(
            project_id, location, datastore_id, res.next_page_token
        )
        docs.extend(res.documents)

    return docs


def list_indexed_urls(
    docs: list[discoveryengine.Document] | None = None,
    project_id: str | None = None,
    location: str | None = None,
    datastore_id: str | None = None,
) -> list[str]:
    """Get the list of docs in data store, then parse to only urls."""
    if not docs:
        docs = list_documents(project_id, location, datastore_id)
    urls = [doc.content.uri for doc in docs]

    return urls


def search_url(urls: list[str], url: str) -> None:
    """Searches a url in a list of urls."""
    for item in urls:
        if url in item:
            print(item)


def search_doc_id(
    doc_id: str,
    docs: list[discoveryengine.Document] | None = None,
    project_id: str | None = None,
    location: str | None = None,
    datastore_id: str | None = None,
) -> None:
    """Searches a doc_id in a list of docs."""
    if not docs:
        docs = list_documents(project_id, location, datastore_id)

    doc_found = False
    for doc in docs:
        if doc.parent_document_id == doc_id:
            doc_found = True
            print(doc)

    if not doc_found:
        print(f"Document not found for provided Doc ID: `{doc_id}`")


def estimate_data_store_size(
    urls: list[str] | None = None,
    docs: list[discoveryengine.Document] | None = None,
    project_id: str | None = None,
    location: str | None = None,
    datastore_id: str | None = None,
) -> None:
    """For Advanced Website Indexing data stores only."""
    if not urls:
        if not docs:
            docs = list_documents(project_id, location, datastore_id)
        urls = list_indexed_urls(docs=docs)

    # Filter to only include website urls.
    urls = list(filter(lambda x: re.search(r"https?://", x), urls))

    if not urls:
        print(
            "No urls found. Make sure this data store is for websites with advanced indexing."
        )
        return

    # For website indexing, each page is calculated as 500KB.
    size = len(urls) * 500_000
    print(f"Estimated data store size: {humanize.naturalsize(size)}")


PENDING_MESSAGE = """
No docs found.\n\nIt\'s likely one of the following issues: \n  [1] Your data store is not finished indexing. \n  [2] Your data store failed indexing. \n  [3] Your data store is for website data without advanced indexing.\n\n
If you just added your data store, it can take up to 4 hours before it will become available.
"""
```

# User Inputs

You can find your `datastore_id` by going following these steps:

1. Click on Vertex AI Search & Conversation
2. Select your App / Engine

![image.png](image.png)

3. Select your Available Data Store

![image-3.png](image-3.png)

4. Find your Data Store ID

![image-2.png](image-2.png)



```
project_id = "YOUR_PROJECT_ID"
location = "global"  # Options: "global", "us", "eu"
datastore_id = "YOUR_DATA_STORE_ID"
```

# Check Data Store Index Status

Using the `list_documents` method, we can do a check to see if the data store has finished indexing.



```
docs = list_documents(project_id, location, datastore_id)

if len(docs) == 0:
    print(PENDING_MESSAGE)
else:
    SUCCESS_MESSAGE = f"""
  Success! 🎉\n
  Your indexing is complete.\n
  Your index contains {len(docs)} documents.
  """
    print(SUCCESS_MESSAGE)
```

    
    No docs found.
    
    It's likely one of two issues: 
      [1] Your data store is not finished indexing. 
      [2] Your data store failed indexing.
    
    If you just added your data store, it can take up to 4 hours before it will become available.
    
    

# List Documents

List all the documents for a given Data Store ID



```
docs = list_documents(project_id, location, datastore_id)
docs[0]
```




    struct_data {
    }
    name: "projects/772105163160/locations/global/collections/default_collection/dataStores/cgc_1690226759325/branches/0/documents/000a98558b6fe9aef7992c9023fb7fdb"
    id: "000a98558b6fe9aef7992c9023fb7fdb"
    schema_id: "default_schema"
    content {
      uri: "https://cloud.google.com/docs/security/data-loss-prevention/revoking-user-access?hl=es-419"
      mime_type: "text/html"
    }
    parent_document_id: "000a98558b6fe9aef7992c9023fb7fdb"



# Search Data Store by Doc ID

Search through all Docs in a given Data Store and find a specific Doc ID.



```
document_id = "000a98558b6fe9aef7992c9023fb7fdb"

search_doc_id(document_id, docs=docs)
```

    struct_data {
    }
    name: "projects/772105163160/locations/global/collections/default_collection/dataStores/cgc_1690226759325/branches/0/documents/000a98558b6fe9aef7992c9023fb7fdb"
    id: "000a98558b6fe9aef7992c9023fb7fdb"
    schema_id: "default_schema"
    content {
      uri: "https://cloud.google.com/docs/security/data-loss-prevention/revoking-user-access?hl=es-419"
      mime_type: "text/html"
    }
    parent_document_id: "000a98558b6fe9aef7992c9023fb7fdb"
    
    

# List Indexed URLs



```
urls = list_indexed_urls(docs=docs)
urls[0]
```




    'https://cloud.google.com/docs/security/data-loss-prevention/revoking-user-access?hl=es-419'



# Search Indexed URLs



```
search_url(urls, "https://cloud.google.com/docs/terraform/samples")
```

    https://cloud.google.com/docs/terraform/samples?hl=ko
    https://cloud.google.com/docs/terraform/samples?hl=fr
    https://cloud.google.com/docs/terraform/samples?hl=pt-br
    https://cloud.google.com/docs/terraform/samples?hl=de
    https://cloud.google.com/docs/terraform/samples?hl=zh-cn
    https://cloud.google.com/docs/terraform/samples?hl=ja
    https://cloud.google.com/docs/terraform/samples
    https://cloud.google.com/docs/terraform/samples?hl=es-419
    https://cloud.google.com/docs/terraform/samples?hl=it
    


```
search_url(urls, "terraform")
```

    https://cloud.google.com/docs/terraform/getting-support?hl=de
    https://cloud.google.com/docs/terraform/basic-commands?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=es-419
    https://cloud.google.com/docs/terraform/samples?hl=ko
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=de
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=ko
    https://cloud.google.com/docs/terraform/deploy-foundation-using-terraform-from-console?hl=pt-br
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=de
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=it
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=pt-br
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=es-419
    https://cloud.google.com/docs/terraform/getting-support?hl=he
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=zh-cn
    https://cloud.google.com/docs/terraform/resources
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation?hl=pt-br
    https://cloud.google.com/docs/terraform/resources?hl=it
    https://cloud.google.com/docs/terraform/get-started-with-terraform
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=it
    https://cloud.google.com/docs/terraform/basic-commands?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=he
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=de
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=zh-cn
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=ja
    https://cloud.google.com/docs/terraform/resource-management/import?hl=ko
    https://cloud.google.com/docs/terraform/deploy-foundation-using-terraform-from-console?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=de
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=de
    https://cloud.google.com/docs/terraform/resource-management/export?hl=de
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=ko
    https://cloud.google.com/docs/terraform
    https://cloud.google.com/docs/terraform/?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=fr
    https://cloud.google.com/docs/terraform/release-notes
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=fr
    https://cloud.google.com/docs/terraform/basic-commands?hl=pt-br
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=de
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=zh-cn
    https://cloud.google.com/docs/terraform/resources?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=zh-cn
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=fr
    https://cloud.google.com/docs/terraform/resource-management/export?hl=zh-cn
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=es-419
    https://cloud.google.com/docs/terraform/resource-management/export?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=zh-cn
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=ja
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=ja
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=de
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=zh-cn
    https://cloud.google.com/docs/terraform/samples?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation?hl=he
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=pt-br
    https://cloud.google.com/docs/terraform/samples?hl=pt-br
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=de
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=zh-cn
    https://cloud.google.com/docs/terraform/policy-validation?hl=de
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=it
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=ko
    https://cloud.google.com/docs/terraform/resource-management/store-state
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=de
    https://cloud.google.com/docs/terraform/resource-management/import?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=ja
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=ko
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=zh-cn
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=it
    https://cloud.google.com/docs/terraform?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=ja
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=de
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=pt-br
    https://cloud.google.com/docs/terraform/resources?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=ko
    https://cloud.google.com/docs/terraform/resource-management/import
    https://cloud.google.com/docs/terraform/samples?hl=de
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=zh-cn
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=zh-cn
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=ja
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=it
    https://cloud.google.com/docs/terraform/basic-commands
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=fr
    https://cloud.google.com/docs/terraform?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=ko
    https://cloud.google.com/docs/terraform/getting-support
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints
    https://cloud.google.com/docs/terraform/policy-validation?hl=it
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator
    https://cloud.google.com/docs/terraform/resource-management/export?hl=fr
    https://cloud.google.com/docs/terraform/basic-commands?hl=he
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=fr
    https://cloud.google.com/docs/terraform?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=it
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies
    https://cloud.google.com/docs/terraform/getting-support?hl=zh-cn
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=ko
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=it
    https://cloud.google.com/docs/terraform/resource-management/import?hl=ja
    https://cloud.google.com/docs/terraform/getting-support?hl=pt-br
    https://cloud.google.com/docs/terraform/basic-commands?hl=zh-cn
    https://cloud.google.com/docs/terraform?hl=it
    https://cloud.google.com/docs/terraform/getting-support?hl=fr
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=ja
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=zh-cn
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=de
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=ko
    https://cloud.google.com/docs/terraform/get-started-with-terraform?hl=de
    https://cloud.google.com/docs/terraform/resource-management/import?hl=es-419
    https://cloud.google.com/docs/terraform/resource-management/export?hl=ko
    https://cloud.google.com/docs/terraform/resources?hl=pt-br
    https://cloud.google.com/docs/terraform/resource-management/import?hl=pt-br
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=es-419
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=ja
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=fr
    https://cloud.google.com/docs/terraform/samples?hl=zh-cn
    https://cloud.google.com/docs/terraform/basic-commands?hl=it
    https://cloud.google.com/docs/terraform/resource-management/import?hl=it
    https://cloud.google.com/docs/terraform/basic-commands?hl=de
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=he
    https://cloud.google.com/docs/terraform/samples?hl=ja
    https://cloud.google.com/docs/terraform?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting
    https://cloud.google.com/docs/terraform/deploy-foundation-using-terraform-from-console?hl=fr
    https://cloud.google.com/docs/terraform/resources?hl=he
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints
    https://cloud.google.com/docs/terraform/resource-management/import?hl=he
    https://cloud.google.com/docs/terraform/samples
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=ja
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=es-419
    https://cloud.google.com/docs/terraform/resources?hl=de
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=pt-br
    https://cloud.google.com/docs/terraform/resource-management/export?hl=he
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=pt-br
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=he
    https://cloud.google.com/docs/terraform?hl=pt-br
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=it
    https://cloud.google.com/docs/terraform?hl=de
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library
    https://cloud.google.com/docs/terraform/getting-support?hl=it
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=ja
    https://cloud.google.com/docs/terraform/best-practices-for-terraform
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=it
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console
    https://cloud.google.com/docs/terraform/resources?hl=es-419
    https://cloud.google.com/docs/terraform/getting-support?hl=es-419
    https://cloud.google.com/docs/terraform?hl=zh-cn
    https://cloud.google.com/docs/terraform/getting-support?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=es-419
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=de
    https://cloud.google.com/docs/terraform/resource-management/import?hl=de
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=ko
    https://cloud.google.com/docs/terraform/samples?hl=es-419
    https://cloud.google.com/docs/terraform/resources?hl=zh-cn
    https://cloud.google.com/docs/terraform/basic-commands?hl=ja
    https://cloud.google.com/docs/terraform/basic-commands?hl=es-419
    https://cloud.google.com/docs/terraform/resource-management/export
    https://cloud.google.com/docs/terraform/resource-management/export?hl=ja
    https://cloud.google.com/docs/terraform?hl=ja
    https://cloud.google.com/docs/terraform/policy-validation/migrate-from-terraform-validator?hl=pt-br
    https://cloud.google.com/docs/terraform/blueprints/terraform-blueprints?hl=he
    https://cloud.google.com/docs/terraform/resource-management/import?hl=zh-cn
    https://cloud.google.com/docs/terraform/deploy-foundation-using-terraform-from-console?hl=he
    https://cloud.google.com/docs/enterprise/deploy-foundation-using-terraform-from-console?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/create-policy-library?hl=it
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=ko
    https://cloud.google.com/docs/terraform/policy-validation?hl=zh-cn
    https://cloud.google.com/docs/terraform/policy-validation?hl=ja
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=it
    https://cloud.google.com/docs/terraform/policy-validation/quickstart?hl=he
    https://cloud.google.com/docs/terraform/resources?hl=ja
    https://cloud.google.com/docs/terraform/policy-validation/create-terraform-constraints?hl=zh-cn
    https://cloud.google.com/docs/terraform/deploy-foundation-using-terraform-from-console?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/create-cai-constraints?hl=he
    https://cloud.google.com/docs/terraform/samples?hl=it
    https://cloud.google.com/docs/terraform/getting-support?hl=ja
    https://cloud.google.com/docs/terraform/resource-management/managing-infrastructure-as-code?hl=ja
    https://cloud.google.com/docs/terraform/policy-validation/troubleshooting?hl=it
    https://cloud.google.com/docs/terraform/resource-management/export?hl=it
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=ja
    https://cloud.google.com/docs/terraform/resource-management/store-state?hl=ko
    https://cloud.google.com/docs/terraform/resource-management/export?hl=es-419
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=zh-cn
    https://cloud.google.com/docs/terraform/best-practices-for-terraform?hl=es-419
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=fr
    https://cloud.google.com/docs/terraform/policy-validation/quickstart
    https://cloud.google.com/docs/terraform/policy-validation/validate-policies?hl=es-419
    

# Estimate Data Store Size

Only for Website data stores with Advanced Website Indexing



```
estimate_data_store_size(urls=urls)
```

    Estimated data store size: 247.5 MB
    
