# Create a Vertex AI Datastore and Search Engine

<table align="left">

  <td>
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/search/create_datastore_and_search.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo"> Run in Colab
    </a>
  </td>
  <td>
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/create_datastore_and_search.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
      View on GitHub
    </a>
  </td>
  <td>
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/search/create_datastore_and_search.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo">
      Open in Vertex AI Workbench
    </a>
  </td>
</table>

---

* Author(s): [Kara Greenfield](https://github.com/kgreenfield2)
* Created: 22 Nov 2023
* Updated: 31 Oct 2024

---

## Objective

This notebook shows how to create and populate a Vertex AI Search Datastore, how to create a search app connected to that datastore, and how to submit queries through the search engine.


Services used in the notebook:

- ✅ Vertex AI Search for document search and retrieval

## Install pre-requisites

If running in Colab install the pre-requisites into the runtime. Otherwise it is assumed that the notebook is running in Vertex AI Workbench.


```
%pip install --upgrade --user -q google-cloud-discoveryengine
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Restart kernel after installs so that your environment can access the new packages

import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


## Authenticate

If running in Colab authenticate with `google.colab.google.auth` otherwise assume that running on Vertex AI Workbench.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

## Configure notebook environment


```
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine

PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "global"
```

Set [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)


```
!gcloud auth application-default login --project {PROJECT_ID}
```

## Create and Populate a Datastore


```
def create_data_store(
    project_id: str, location: str, data_store_name: str, data_store_id: str
):
    # Create a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.DataStoreServiceClient(client_options=client_options)

    # Initialize request argument(s)
    data_store = discoveryengine.DataStore(
        display_name=data_store_name,
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
    )

    operation = client.create_data_store(
        request=discoveryengine.CreateDataStoreRequest(
            parent=client.collection_path(
                project_id, location, "default_collection"
            ),
            data_store=data_store,
            data_store_id=data_store_id,
        )
    )

    # Make the request
    # The try block is necessary to prevent execution from halting due to an error being thrown when the datastore takes a while to instantiate
    try:
        response = operation.result(timeout=90)
    except:
        print("long-running operation error.")
```


```
# The datastore name can only contain lowercase letters, numbers, and hyphens
DATASTORE_NAME = "alphabet-contracts"
DATASTORE_ID = f"{DATASTORE_NAME}-id"

create_data_store(PROJECT_ID, LOCATION, DATASTORE_NAME, DATASTORE_ID)
```


```
def import_documents(
    project_id: str,
    location: str,
    data_store_id: str,
    gcs_uri: str,
):
    # Create a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.DocumentServiceClient(client_options=client_options)

    # The full resource name of the search engine branch.
    # e.g. projects/{project}/locations/{location}/dataStores/{data_store_id}/branches/{branch}
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )

    source_documents = [f"{gcs_uri}/*"]

    request = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(
            input_uris=source_documents, data_schema="content"
        ),
        # Options: `FULL`, `INCREMENTAL`
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    )

    # Make the request
    operation = client.import_documents(request=request)

    response = operation.result()

    # Once the operation is complete,
    # get information from operation metadata
    metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)

    # Handle the response
    return operation.operation.name
```


```
source_documents_gs_uri = (
    "gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs"
)

import_documents(PROJECT_ID, LOCATION, DATASTORE_ID, source_documents_gs_uri)
```

## Create a Search Engine

This is used to set the `search_tier` to enterprise and to enable advanced LLM features.

Enterprise tier is required to get extractive answers from a search query and advanced LLM features are required to summarize search results.


```
def create_engine(
    project_id: str, location: str, engine_name: str, engine_id: str, data_store_id: str
):
    # Create a client
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )
    client = discoveryengine.EngineServiceClient(client_options=client_options)

    # Initialize request argument(s)
    engine = discoveryengine.Engine(
        display_name=engine_name,
        solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        data_store_ids=[data_store_id],
        search_engine_config=discoveryengine.Engine.SearchEngineConfig(
            search_tier=discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE,
            search_add_ons=[discoveryengine.SearchAddOn.SEARCH_ADD_ON_LLM],
        ),
    )

    request = discoveryengine.CreateEngineRequest(
        parent=client.collection_path(project_id, location, "default_collection"),
        engine=engine,
        engine_id=engine.display_name,
    )

    # Make the request
    operation = client.create_engine(request=request)
    response = operation.result(timeout=90)
```


```
ENGINE_NAME = DATASTORE_NAME
ENGINE_ID = DATASTORE_ID
create_engine(PROJECT_ID, LOCATION, ENGINE_NAME, ENGINE_ID, DATASTORE_ID)
```

## Query your Search Engine

Note: The Engine will take some time to be ready to query.

If you recently created an engine and you receive an error similar to:

`404 Engine {ENGINE_NAME} is not found`

Then wait a few minutes and try your query again.


```
def search_sample(
    project_id: str,
    location: str,
    engine_id: str,
    search_query: str,
) -> list[discoveryengine.SearchResponse]:
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if LOCATION != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search engine serving config
    # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_search"

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=10,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)
    return response
```


```
query = "Who is the CEO of Google?"

response = search_sample(PROJECT_ID, LOCATION, ENGINE_ID, query)
print(response.summary.summary_text)
```
