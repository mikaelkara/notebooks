```
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Use BigQuery DataFrames with Generative AI for code generation

<table align="left">
    <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_llm_code_generation.ipynb">
        <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
    </td>
    <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Flanguage%2Fuse-cases%2Fapplying-llms-to-data%2Fbigquery_dataframes_llm_code_generation.ipynb">
        <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
    </td>
    <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_llm_code_generation.ipynb">
        <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
    </td>
    <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_llm_code_generation.ipynb">
        <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
    </td>
    <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_llm_code_generation.ipynb">
        <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
    </td>
</table>

| | |
|-|-|
|Author(s) | [Ashley Xu](https://github.com/ashleyxuu) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.10

## Overview

Use this notebook to walk through an example use case of generating sample code by using BigQuery DataFrames and its integration with Generative AI support on Vertex AI.

Learn more about [BigQuery DataFrames](https://cloud.google.com/python/docs/reference/bigframes/latest).

### Objective

In this tutorial, you create a CSV file containing sample code for calling a given set of APIs.

The steps include:

- Defining an LLM model in BigQuery DataFrames, specifically the [`text-bison` model of the PaLM API](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text), using `bigframes.ml.llm`.
- Creating a DataFrame by reading in data from Cloud Storage.
- Manipulating data in the DataFrame to build LLM prompts.
- Sending DataFrame prompts to the LLM model using the `predict` method.
- Creating and using a custom function to transform the output provided by the LLM model response.
- Exporting the resulting transformed DataFrame as a CSV file.

### Dataset

This tutorial uses a dataset listing the names of various pandas DataFrame and Series APIs.

### Costs

This tutorial uses billable components of Google Cloud:

* BigQuery
* Generative AI support on Vertex AI
* Cloud Functions

Learn about [BigQuery compute pricing](https://cloud.google.com/bigquery/pricing#analysis_pricing_models),
[Generative AI support on Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing#generative_ai_models), and [Cloud Functions pricing](https://cloud.google.com/functions/pricing), and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Installation

Install the following packages, which are required to run this notebook:


```
%pip install bigframes --upgrade --quiet
```

## Before you begin

Complete the tasks in this section to set up your environment.

### Set up your Google Cloud project

**The following steps are required, regardless of your notebook environment.**

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 credit towards your compute/storage costs.

2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

3. [Click here](https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com,bigqueryconnection.googleapis.com,cloudfunctions.googleapis.com,run.googleapis.com,artifactregistry.googleapis.com,cloudbuild.googleapis.com,cloudresourcemanager.googleapis.com) to enable the following APIs:

  * BigQuery API
  * BigQuery Connection API
  * Cloud Functions API
  * Cloud Run API
  * Artifact Registry API
  * Cloud Build API
  * Cloud Resource Manager API
  * Vertex AI API

4. If you are running this notebook locally, install the [Cloud SDK](https://cloud.google.com/sdk).

#### Set your project ID

If you don't know your project ID, try the following:
* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113).


```
PROJECT_ID = ""  # @param {type:"string"}

# Set the project id
! gcloud config set project {PROJECT_ID}
```

#### Set the region

You can also change the `REGION` variable used by BigQuery. Learn more about [BigQuery regions](https://cloud.google.com/bigquery/docs/locations#supported_locations).


```
REGION = "US"  # @param {type: "string"}
```

### Authenticate your Google Cloud account

Depending on your Jupyter environment, you might have to manually authenticate. Follow the relevant instructions below.

**Vertex AI Workbench**

Do nothing, you are already authenticated.

**Local JupyterLab instance**

Uncomment and run the following cell:


```
# ! gcloud auth login
```

**Colab**

Uncomment and run the following cell:


```
# from google.colab import auth
# auth.authenticate_user()
```

### Import libraries


```
import bigframes.pandas as bf
from google.cloud import bigquery_connection_v1 as bq_connection
```

### Set BigQuery DataFrames options


```
bf.options.bigquery.project = PROJECT_ID
bf.options.bigquery.location = REGION
```

If you want to reset the location of the created DataFrame or Series objects, reset the session by executing `bf.reset_session()`. After that, you can reuse `bf.options.bigquery.location` to specify another location.

# Define the LLM model

BigQuery DataFrames provides integration with [`text-bison` model of the PaLM API](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text) via Vertex AI.

This section walks through a few steps required in order to use the model in your notebook.

## Create a BigQuery Cloud resource connection

You need to create a [Cloud resource connection](https://cloud.google.com/bigquery/docs/create-cloud-resource-connection) to enable BigQuery DataFrames to interact with Vertex AI services.


```
CONN_NAME = "bqdf-llm"

client = bq_connection.ConnectionServiceClient()
new_conn_parent = f"projects/{PROJECT_ID}/locations/{REGION}"
exists_conn_parent = f"projects/{PROJECT_ID}/locations/{REGION}/connections/{CONN_NAME}"
cloud_resource_properties = bq_connection.CloudResourceProperties({})

try:
    request = client.get_connection(
        request=bq_connection.GetConnectionRequest(name=exists_conn_parent)
    )
    CONN_SERVICE_ACCOUNT = f"serviceAccount:{request.cloud_resource.service_account_id}"
except Exception:
    connection = bq_connection.types.Connection(
        {"friendly_name": CONN_NAME, "cloud_resource": cloud_resource_properties}
    )
    request = bq_connection.CreateConnectionRequest(
        {
            "parent": new_conn_parent,
            "connection_id": CONN_NAME,
            "connection": connection,
        }
    )
    response = client.create_connection(request)
    CONN_SERVICE_ACCOUNT = (
        f"serviceAccount:{response.cloud_resource.service_account_id}"
    )
print(CONN_SERVICE_ACCOUNT)
```

## Set permissions for the service account

The resource connection service account requires certain project-level permissions:
 - `roles/aiplatform.user` and `roles/bigquery.connectionUser`: These roles are required for the connection to create a model definition using the LLM model in Vertex AI ([documentation](https://cloud.google.com/bigquery/docs/generate-text#give_the_service_account_access)).
 - `roles/run.invoker`: This role is required for the connection to have read-only access to Cloud Run services that back custom/remote functions ([documentation](https://cloud.google.com/bigquery/docs/remote-functions#grant_permission_on_function)).

Set these permissions by running the following `gcloud` commands:


```
!gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member={CONN_SERVICE_ACCOUNT} --role='roles/bigquery.connectionUser'
!gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member={CONN_SERVICE_ACCOUNT} --role='roles/aiplatform.user'
!gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member={CONN_SERVICE_ACCOUNT} --role='roles/run.invoker'
```

## Define the model

Use `bigframes.ml.llm` to define the model:


```
from bigframes.ml.llm import PaLM2TextGenerator

session = bf.get_global_session()
connection = f"{PROJECT_ID}.{REGION}.{CONN_NAME}"
model = PaLM2TextGenerator(session=session, connection_name=connection)
```

# Read data from Cloud Storage into BigQuery DataFrames

You can create a BigQuery DataFrames DataFrame by reading data from any of the following locations:

* A local data file
* Data stored in a BigQuery table
* A data file stored in Cloud Storage
* An in-memory pandas DataFrame

In this tutorial, you create BigQuery DataFrames DataFrames by reading two CSV files stored in Cloud Storage, one containing a list of DataFrame API names and one containing a list of Series API names.


```
df_api = bf.read_csv("gs://cloud-samples-data/vertex-ai/bigframe/df.csv")
series_api = bf.read_csv("gs://cloud-samples-data/vertex-ai/bigframe/series.csv")
```

Take a peek at a few rows of data for each file:


```
df_api.head(2)
```


```
series_api.head(2)
```

# Generate code using the LLM model

Prepare the prompts and send them to the LLM model for prediction.

## Prompt design in BigQuery DataFrames

Designing prompts for LLMs is a fast growing area and you can read more in [this documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/introduction-prompt-design).

For this tutorial, you use a simple prompt to ask the LLM model for sample code for each of the API methods (or rows) from the last step's DataFrames. The output is the new DataFrames `df_prompt` and `series_prompt`, which contain the full prompt text.


```
df_prompt_prefix = "Generate Pandas sample code for DataFrame."
series_prompt_prefix = "Generate Pandas sample code for Series."

df_prompt = df_prompt_prefix + df_api["API"]
series_prompt = series_prompt_prefix + series_api["API"]

df_prompt.head(2)
```

## Make predictions using the LLM model

Use the BigQuery DataFrames DataFrame containing the full prompt text as the input to the `predict` method. The `predict` method calls the LLM model and returns its generated text output back to two new BigQuery DataFrames DataFrames, `df_pred` and `series_pred`.

Note: The predictions might take a few minutes to run.


```
df_pred = model.predict(df_prompt.to_frame(), max_output_tokens=1024)
series_pred = model.predict(series_prompt.to_frame(), max_output_tokens=1024)
```

Once the predictions are processed, take a look at the sample output from the LLM, which provides code samples for the API names listed in the DataFrames dataset.


```
print(df_pred["ml_generate_text_llm_result"].iloc[0])
```

# Manipulate LLM output using a remote function

The output that the LLM provides often contains additional text beyond the code sample itself. Using BigQuery DataFrames, you can deploy custom Python functions that process and transform this output.

Running the cell below creates a custom function that you can use to process the LLM output data in two ways:
1. Strip the LLM text output to include only the code block.
2. Substitute `import pandas as pd` with `import bigframes.pandas as bf` so that the resulting code block works with BigQuery DataFrames.


```
@bf.remote_function([str], str, bigquery_connection=CONN_NAME)
def extract_code(text: str):
    try:
        res = text[text.find("\n") + 1 : text.find("```", 3)]
        res = res.replace("import pandas as pd", "import bigframes.pandas as bf")
        if "import bigframes.pandas as bf" not in res:
            res = "import bigframes.pandas as bf\n" + res
        return res
    except:
        return ""
```

The custom function is deployed as a Cloud Function, and then integrated with BigQuery as a [remote function](https://cloud.google.com/bigquery/docs/remote-functions). Save both of the function names so that you can clean them up at the end of this notebook.


```
CLOUD_FUNCTION_NAME = format(extract_code.bigframes_cloud_function)
print("Cloud Function Name " + CLOUD_FUNCTION_NAME)
REMOTE_FUNCTION_NAME = format(extract_code.bigframes_remote_function)
print("Remote Function Name " + REMOTE_FUNCTION_NAME)
```

Apply the custom function to each LLM output DataFrame to get the processed results:


```
df_code = df_pred.assign(
    code=df_pred["ml_generate_text_llm_result"].apply(extract_code)
)
series_code = series_pred.assign(
    code=series_pred["ml_generate_text_llm_result"].apply(extract_code)
)
```

You can see the differences by inspecting the first row of data:


```
print(df_code["code"].iloc[0])
```

# Save the results to Cloud Storage

BigQuery DataFrames lets you save a BigQuery DataFrames DataFrame as a CSV file in Cloud Storage for further use. Try that now with your processed LLM output data.

Create a new Cloud Storage bucket with a unique name:


```
import uuid

BUCKET_ID = "code-samples-" + str(uuid.uuid1())

!gsutil mb gs://{BUCKET_ID}
```

Use `to_csv` to write each BigQuery DataFrames DataFrame as a CSV file in the Cloud Storage bucket:


```
df_code[["code"]].to_csv(f"gs://{BUCKET_ID}/df_code*.csv")
series_code[["code"]].to_csv(f"gs://{BUCKET_ID}/series_code*.csv")
```

You can navigate to the Cloud Storage bucket browser to download the two files and view them.

Run the following cell, and then follow the link to your Cloud Storage bucket browser:


```
print(f"https://console.developers.google.com/storage/browser/{BUCKET_ID}/")
```

# Summary and next steps

You've used BigQuery DataFrames' integration with LLM models (`bigframes.ml.llm`) to generate code samples, and have transformed LLM output by creating and using a custom function in BigQuery DataFrames.

Learn more about BigQuery DataFrames in the [documentation](https://cloud.google.com/python/docs/reference/bigframes/latest) and find more sample notebooks in the [GitHub repo](https://github.com/googleapis/python-bigquery-dataframes/tree/main/notebooks).

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can uncomment the remaining cells and run them to delete the individual resources you created in this tutorial:


```
# # Delete the BigQuery Connection
# from google.cloud import bigquery_connection_v1 as bq_connection
# client = bq_connection.ConnectionServiceClient()
# CONNECTION_ID = f"projects/{PROJECT_ID}/locations/{REGION}/connections/{CONN_NAME}"
# client.delete_connection(name=CONNECTION_ID)
# print(f"Deleted connection '{CONNECTION_ID}'.")
```


```
# # Delete the Cloud Function
# ! gcloud functions delete {CLOUD_FUNCTION_NAME} --quiet
# # Delete the Remote Function
# REMOTE_FUNCTION_NAME = REMOTE_FUNCTION_NAME.replace(PROJECT_ID + ".", "")
# ! bq rm --routine --force=true {REMOTE_FUNCTION_NAME}
```


```
# # Delete the Google Cloud Storage bucket and files
# ! gsutil rm -r gs://{BUCKET_ID}
# print(f"Deleted bucket '{BUCKET_ID}'.")
```
