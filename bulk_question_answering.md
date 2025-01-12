# Bulk Question Answering with Vertex AI Search

Answer questions from a CSV using a Vertex AI Search data store.

<table align="left">

  <td>
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/search/bulk-question-answering/bulk_question_answering.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td>
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/search/bulk-question-answering/bulk_question_answering.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td>
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/search/bulk-question-answering/bulk_question_answering.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Ruchika Kharwar](https://github.com/rasalt), [Holt Skinner](https://github.com/holtskinner) |

## Install pre-requisites

If running in Colab install the pre-requisites into the runtime. Otherwise it is assumed that the notebook is running in Vertex AI Workbench. In that case it is recommended to install the pre-requisites from a terminal using the `--user` option.


```
%pip install google-cloud-discoveryengine google-auth pandas --upgrade --user -q
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
    from google.colab import auth as google_auth

    google_auth.authenticate_user()
```

## Definitions for the index values

- "Query"
- "Golden Doc"
- "Golden Doc Page Number"
- "Golden Answer"
- "Top 5 Docs"
- "Top 5 extractive answers"
- "Top 5 extractive segments"
- "Answer / Summary"

# Import libraries


```
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1beta as discoveryengine
import pandas as pd
```

### Set the following constants to reflect your environment
* The queries used in the examples here relate to a GCS bucket containing Alphabet investor PDFs, but these should be customized to your own data.


```
PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "global"  # @param {type:"string"}
DATA_STORE_ID = "YOUR_DATA_STORE_ID"  # @param {type:"string"}
```

## Function to search the Vertex AI Search data store


```
def search_data_store(
    project_id: str,
    location: str,
    data_store_id: str,
    search_query: str,
) -> discoveryengine.SearchResponse:
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search engine serving config
    # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}
    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        serving_config="default_config",
    )

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True
        ),
        extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
            max_extractive_answer_count=5,
            max_extractive_segment_count=1,
        ),
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=False,
            ignore_non_summary_seeking_query=False,
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=5,
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

# Function to load results into DataFrame


```
def answer_questions(
    row, project_id: str, location: str, data_store_id: str, top_n: int = 5
) -> None:
    """This function returns the top 5 docs, extractive segments, answers"""
    # Perform search with Query
    response = search_data_store(project_id, location, data_store_id, row["Query"])

    row["Answer / Summary"] = response.summary.summary_text

    top5docs, top5answers, top5segments = [], [], []
    ext_ans_cnt, ext_seg_cnt = 0, 0

    for result in response.results:
        doc_data = getattr(result.document, "derived_struct_data", None)
        if not doc_data:
            continue

        # Process extractive answers
        for chunk in doc_data.get("extractive_answers", []):
            content = chunk.get("content", "").replace("\n", "")
            top5answers.append(content)
            top5docs.append(
                f"Doc: {doc_data.get('link', '')}  Page: {chunk.get('pageNumber', '')}"
            )
            ext_ans_cnt += 1

        # Process extractive segments
        for chunk in doc_data.get("extractive_segments", []):
            data = chunk.get("content", "").replace("\n", "")
            top5segments.append(data)
            ext_seg_cnt += 1

        if ext_ans_cnt >= top_n and ext_seg_cnt >= top_n:
            break

    row["Top 5 Docs"] = "\n\n".join(top5docs)
    row["Top 5 extractive answers"] = "\n\n".join(top5answers)
    row["Top 5 extractive segments"] = "\n\n".join(top5segments)
```

### Gather all of the Vertex AI Search results

- Read in CSV as Pandas DataFrame
- Send questions to Vertex AI Search
- Load Summary, top 5 docs, extractive answers, extractive segments to DataFrame
- Output DataFrame to TSV
  - [Example TSV](https://storage.googleapis.com/github-repo/search/bulk-question-answering/bulk_question_answering_output.tsv)


```
# Open the CSV file and read column values
df = pd.read_csv(
    "gs://github-repo/search/bulk-question-answering/bulk_question_answering_input.csv",
    header=0,
    dtype=str,
)

# Make Vertex AI Search request for each question
df.apply(
    lambda row: answer_questions(row, PROJECT_ID, LOCATION, DATA_STORE_ID, top_n=5),
    axis=1,
)

# Output results to new TSV file
df.to_csv("bulk_question_answering_output.tsv", index=False, sep="\t")

df
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
      <th>Query</th>
      <th>Golden Doc</th>
      <th>Golden Doc Page Number</th>
      <th>Golden Answer</th>
      <th>Top 5 Docs</th>
      <th>Top 5 extractive answers</th>
      <th>Top 5 extractive segments</th>
      <th>Answer / Summary</th>
      <th>Feedback from customer / account team about returned docs and answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What was Google's revenue in 2021?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Doc: gs://cloud-samples-data/gen-app-builder/s...</td>
      <td>Google Cloud had an Operating Loss of $890 mil...</td>
      <td>Within Other Revenues, we are pleased with the...</td>
      <td>Google's revenue for the full year 2021 was $5...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What was Google's revenue in 2022?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Doc: gs://cloud-samples-data/gen-app-builder/s...</td>
      <td>Other Revenues were $8.2 billion, up 22%, driv...</td>
      <td>Let me now turn to our segment financial resul...</td>
      <td>Google's total revenue was $282.8 billion in 2...</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


