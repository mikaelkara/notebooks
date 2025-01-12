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

# BigQuery DataFrames ML: Drug Name Generation

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_ml_drug_name_generation.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Flanguage%2Fuse-cases%2Fapplying-llms-to-data%2Fbigquery_dataframes_ml_drug_name_generation.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_ml_drug_name_generation.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_ml_drug_name_generation.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_dataframes_ml_drug_name_generation.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Ashley Xu](https://github.com/ashleyxuu) |

**_NOTE_**: This notebook has been tested in the following environment:

* Python version = 3.9

## Overview

The goal of this notebook is to demonstrate an enterprise generative AI use case. A marketing user can provide information about a new pharmaceutical drug and its generic name, and receive ideas on marketing-oriented brand names for that drug.

Learn more about [BigQuery DataFrames](https://cloud.google.com/bigquery/docs/dataframes-quickstart).

### Objective

In this tutorial, you learn about Generative AI concepts such as prompting and few-shot learning, as well as how to use BigFrames ML for performing these tasks simply using an intuitive dataframe API.

The steps performed include:

1. Ask the user for the generic name and usage for the drug.
1. Use `bigframes` to query the FDA dataset of over 100,000 drugs, filtered on the brand name, generic name, and indications & usage columns.
1. Filter this dataset to find prototypical brand names that can be used as examples in prompt tuning.
1. Create a prompt with the user input, general instructions, examples and counter-examples for the desired brand name.
1. Use the `bigframes.ml.llm.PaLM2TextGenerator` to generate choices of brand names.

### Dataset

This notebook uses the [FDA dataset](https://cloud.google.com/blog/topics/healthcare-life-sciences/fda-mystudies-comes-to-google-cloud) available at [`bigquery-public-data.fda_drug`](https://console.cloud.google.com/bigquery?ws=!1m4!1m3!3m2!1sbigquery-public-data!2sfda_drug).

### Costs

This tutorial uses billable components of Google Cloud:

* BigQuery (compute)
* BigQuery ML

Learn about [BigQuery compute pricing](https://cloud.google.com/bigquery/pricing#analysis_pricing_models),
and [BigQuery ML pricing](https://cloud.google.com/bigquery/pricing#bqml),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Installation

Install the following packages required to execute this notebook.


```
%pip install -U --quiet bigframes
```

### Colab only: Uncomment the following cell to restart the kernel.


```
# # Automatically restart kernel after installs so that your environment can access the new packages
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

### Import libraries


```
from IPython.display import Markdown
from bigframes.ml.llm import PaLM2TextGenerator
import bigframes.pandas as bpd
from google.cloud import bigquery_connection_v1 as bq_connection
```

### Authenticate your Google Cloud account

Depending on your Jupyter environment, you may have to manually authenticate. Follow the relevant instructions below.

**1. Vertex AI Workbench**
* Do nothing as you are already authenticated.

**2. Local JupyterLab instance, uncomment and run:**


```
# ! gcloud auth login
```

**3. Colab, uncomment and run:**


```
# from google.colab import auth

# auth.authenticate_user()
```

## Before you begin

### Set up your Google Cloud project

**The following steps are required, regardless of your notebook environment.**

1. [Select or create a Google Cloud project](https://console.cloud.google.com/cloud-resource-manager). When you first create an account, you get a $300 free credit towards your compute/storage costs.

2. [Make sure that billing is enabled for your project](https://cloud.google.com/billing/docs/how-to/modify-project).

3. [Enable the BigQuery API](https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com).

4. If you are running this notebook locally, you need to install the [Cloud SDK](https://cloud.google.com/sdk).

#### Set your project ID

**If you don't know your project ID**, try the following:
* Run `gcloud config list`.
* Run `gcloud projects list`.
* See the support page: [Locate the project ID](https://support.google.com/googleapi/answer/7014113)


```
PROJECT_ID = "<your-project-id>"  # @param {type:"string"}

# Set the project id
! gcloud config set project {PROJECT_ID}
```

#### BigFrames configuration

Next, we will specify a [BigQuery connection](https://cloud.google.com/bigquery/docs/working-with-connections). If you already have a connection, you can simplify provide the name and skip the following creation steps.


```
# Please fill in these values.
LOCATION = "us"  # @param {type:"string"}
CONNECTION = "bigframes-ml"  # @param {type:"string"}

connection_name = f"{PROJECT_ID}.{LOCATION}.{CONNECTION}"
```

We will now try to use the provided connection, and if it doesn't exist, create a new one. We will also print the service account used.


```
# Initialize client and set request parameters
client = bq_connection.ConnectionServiceClient()
new_conn_parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
exists_conn_parent = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/connections/{CONNECTION}"
)
cloud_resource_properties = bq_connection.CloudResourceProperties({})

# Try to connect using provided connection
try:
    request = client.get_connection(
        request=bq_connection.GetConnectionRequest(name=exists_conn_parent)
    )
    CONN_SERVICE_ACCOUNT = f"serviceAccount:{request.cloud_resource.service_account_id}"
# Create a new connection on error
except Exception:
    connection = bq_connection.types.Connection(
        {"friendly_name": CONNECTION, "cloud_resource": cloud_resource_properties}
    )
    request = bq_connection.CreateConnectionRequest(
        {
            "parent": new_conn_parent,
            "connection_id": CONNECTION,
            "connection": connection,
        }
    )
    response = client.create_connection(request)
    CONN_SERVICE_ACCOUNT = (
        f"serviceAccount:{response.cloud_resource.service_account_id}"
    )
# Set service account permissions
!gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member={CONN_SERVICE_ACCOUNT} --role='roles/bigquery.connectionUser'
!gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member={CONN_SERVICE_ACCOUNT} --role='roles/aiplatform.user'
!gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member={CONN_SERVICE_ACCOUNT} --role='roles/run.invoker'

print(CONN_SERVICE_ACCOUNT)
```

### Initialize BigFrames client

Here, we set the project configuration based on the provided parameters.


```
bpd.options.bigquery.project = PROJECT_ID
bpd.options.bigquery.location = LOCATION
```

## Generate a name

Let's start with entering a generic name and description of the drug.


```
GENERIC_NAME = "Entropofloxacin"  # @param {type:"string"}
USAGE = "Entropofloxacin is a fluoroquinolone antibiotic that is used to treat a variety of bacterial infections, including: pneumonia, streptococcus infections, salmonella infections, escherichia coli infections, and pseudomonas aeruginosa infections It is taken by mouth or by injection. The dosage and frequency of administration will vary depending on the type of infection being treated. It should be taken for the full course of treatment, even if symptoms improve after a few days. Stopping the medication early may increase the risk of the infection coming back."  # @param {type:"string"}
NUM_NAMES = 10  # @param {type:"integer"}
TEMPERATURE = 0.5  # @param {type: "number"}
```

We can now create a prompt string, and populate it with the name and description.


```
zero_shot_prompt = f"""Provide {NUM_NAMES} unique and modern brand names in Markdown bullet point format. Do not provide any additional explanation.

Be creative with the brand names. Don't use English words directly; use variants or invented words.

The generic name is: {GENERIC_NAME}

The indications and usage are: {USAGE}."""

print(zero_shot_prompt)
```

Next, let's create a helper function to predict with our model. It will take a string input, and add it to a temporary BigFrames `DataFrame`. It will also return the string extracted from the response `DataFrame`.


```
def predict(prompt: str, temperature: float = TEMPERATURE) -> str:
    # Create dataframe
    input = bpd.DataFrame(
        {
            "prompt": [prompt],
        }
    )

    # Return response
    return model.predict(input, temperature).ml_generate_text_llm_result.iloc[0]
```

We can now initialize the model, and get a response to our prompt!


```
# Get BigFrames session
session = bpd.get_global_session()

# Define the model
model = PaLM2TextGenerator(session=session, connection_name=connection_name)

# Invoke LLM with prompt
response = predict(zero_shot_prompt)

# Print results as Markdown
Markdown(response)
```

We're off to a great start! Let's see if we can refine our response.

## Few-shot learning

Let's try using [few-shot learning](https://paperswithcode.com/task/few-shot-learning). We will provide a few examples of what we're looking for along with our prompt.

Our prompt will consist of 3 parts:
* General instructions (e.g. generate $n$ brand names)
* Multiple examples
* Information about the drug we'd like to generate a name for

Let's walk through how to construct this prompt.

Our first step will be to define how many examples we want to provide in the prompt.


```
# Specify number of examples to include

NUM_EXAMPLES = 3  # @param {type:"integer"}
```

Next, let's define a prefix that will set the overall context.


```
prefix_prompt = f"""Provide {NUM_NAMES} unique and modern brand names in Markdown bullet point format, related to the drug at the bottom of this prompt.

Be creative with the brand names. Don't use English words directly; use variants or invented words.

First, we will provide {NUM_EXAMPLES} examples to help with your thought process.

Then, we will provide the generic name and usage for the drug we'd like you to generate brand names for.
"""

print(prefix_prompt)
```

Our next step will be to include examples into the prompt.

We will start out by retrieving the raw data for the examples, by querying the BigQuery public dataset.


```
# Query 3 columns of interest from drug label dataset
df = bpd.read_gbq(
    "bigquery-public-data.fda_drug.drug_label",
    col_order=["openfda_generic_name", "openfda_brand_name", "indications_and_usage"],
)

# Exclude any rows with missing data
df = df.dropna()

# Drop duplicate rows
df = df.drop_duplicates()

# Print values
df.head()
```

Let's now filter the results to remove atypical names.


```
# Remove names with spaces
df = df[df["openfda_brand_name"].str.find(" ") == -1]

# Remove names with 5 or fewer characters
df = df[df["openfda_brand_name"].str.len() > 5]

# Remove names where the generic and brand name match (case-insensitive)
df = df[df["openfda_generic_name"].str.lower() != df["openfda_brand_name"].str.lower()]
```

Let's take `NUM_EXAMPLES` samples to include in the prompt.


```
# Take a sample and convert to a Pandas dataframe for local usage.
df_examples = df.sample(NUM_EXAMPLES, random_state=3).to_pandas()

df_examples
```

Let's now convert the data to a JSON structure, to enable embedding into a prompt. For consistency, we'll capitalize each example brand name.


```
examples = [
    {
        "brand_name": brand_name.capitalize(),
        "generic_name": generic_name,
        "usage": usage,
    }
    for brand_name, generic_name, usage in zip(
        df_examples["openfda_brand_name"],
        df_examples["openfda_generic_name"],
        df_examples["indications_and_usage"],
    )
]

print(examples)
```

We'll create a prompt template for each example, and view the first one.


```
example_prompt = ""
for example in examples:
    example_prompt += f"Generic name: {example['generic_name']}\nUsage: {example['usage']}\nBrand name: {example['brand_name']}\n\n"

example_prompt
```

Finally, we can create a suffix to our prompt. This will contain the generic name of the drug, its usage, ending with a request for brand names.


```
suffix_prompt = f"""Generic name: {GENERIC_NAME}
Usage: {USAGE}
Brand names:"""

print(suffix_prompt)
```

Let's pull it altogether into a few shot prompt.


```
# Define the prompt
few_shot_prompt = prefix_prompt + example_prompt + suffix_prompt

# Print the prompt
print(few_shot_prompt)
```

Now, let's pass our prompt to the LLM, and get a response!


```
response = predict(few_shot_prompt)

Markdown(response)
```

# Bulk generation

Let's take these experiments to the next level by generating many names in bulk. We'll see how to leverage BigFrames at scale!

We can start by finding drugs that are missing brand names. There are approximately 4,000 drugs that meet this criteria. We'll put a limit of 100 in this notebook.


```
# Query 3 columns of interest from drug label dataset
df_missing = bpd.read_gbq(
    "bigquery-public-data.fda_drug.drug_label",
    col_order=["openfda_generic_name", "openfda_brand_name", "indications_and_usage"],
)

# Exclude any rows with missing data
df_missing = df_missing.dropna()

# Include rows in which openfda_brand_name equals openfda_generic_name
df_missing = df_missing[
    df_missing["openfda_generic_name"] == df_missing["openfda_brand_name"]
]

# Limit the number of rows for demonstration purposes
df_missing = df_missing.head(100)

# Print values
df_missing.head()
```

We will create a column `prompt` with a customized prompt for each row.


```
df_missing["prompt"] = (
    "Provide a unique and modern brand name related to this pharmaceutical drug."
    + "Don't use English words directly; use variants or invented words. The generic name is: "
    + df_missing["openfda_generic_name"]
    + ". The indications and usage are: "
    + df_missing["indications_and_usage"]
    + "."
)
```

We'll create a new helper method, `batch_predict()` and query the LLM. The job may take a couple minutes to execute.


```
def batch_predict(
    input: bpd.DataFrame, temperature: float = TEMPERATURE
) -> bpd.DataFrame:
    return model.predict(input, temperature).ml_generate_text_llm_result


response = batch_predict(df_missing["prompt"])
```

Let's check the results for one of our responses!


```
# Pick a sample
k = 0

# Gather the prompt and response details
df_missing = df_missing.head(100).reset_index(drop=True)

prompt_generic = df_missing["openfda_generic_name"][k]
prompt_usage = df_missing["indications_and_usage"][k]
response_str = response[k]

# Print details
print(f"Generic name: {prompt_generic}")
print(f"Brand name: {prompt_usage}")
print(f"Response: {response_str}")
```

Congratulations! You have learned how to use generative AI to jump-start the creative process.

You've also seen how BigFrames can manage each step of the process, including gathering data, data manipulation, and querying the LLM.

## Cleaning up

To clean up all Google Cloud resources used in this project, you can [delete the Google Cloud
project](https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects) you used for the tutorial.

Otherwise, you can uncomment the remaining cells and run them to delete the individual resources you created in this tutorial:


```
# Delete the BigQuery Connection
from google.cloud import bigquery_connection_v1 as bq_connection

client = bq_connection.ConnectionServiceClient()
CONNECTION_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/connections/{CONNECTION}"
client.delete_connection(name=CONNECTION_ID)
print(f"Deleted connection {CONNECTION_ID}.")
```
