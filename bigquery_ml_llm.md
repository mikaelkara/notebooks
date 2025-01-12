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

# Using Vertex AI LLMs with data in BigQuery

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_ml_llm.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Flanguage%2Fuse-cases%2Fapplying-llms-to-data%2Fbigquery_ml_llm.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/use-cases/applying-llms-to-data/bigquery_ml_llm.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_ml_llm.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/applying-llms-to-data/bigquery_ml_llm.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Rachael Deacon-smith](https://github.com/rachael-ds) |

## Overview

You might wonder: how do you use LLMs with your data sitting in a data warehouse?

The latest integrations between [BigQuery ML](https://cloud.google.com/bigquery/docs/bqml-introduction) (BQML) and [Vertex AI LLMs](https://cloud.google.com/vertex-ai) (PaLM 2 for Text) mean that organizations can now use Vertex AI LLMs on their BigQuery data. Organizations can continue to use BigQuery for data analytics, while also taking advantage of the power of generative AI without the need to move their data.

In this tutorial, you will go through examples of how to use Vertex AI LLMs with data stored in BigQuery.

### Objectives
The objective is to demonstrate some of the many ways LLMs can be applied to your BigQuery data using BigQuery ML.


You will execute simple SQL statements that call the Vertex AI API with the `ML.GENERATE_TEXT` function to:

- Summarize and classify text
- Perform entity recognition
- Enrich data
- Run sentiment analysis


### Services and Costs
This tutorial uses the following Google Cloud data analytics and ML services, they are billable components of Google Cloud:

* BigQuery & BigQuery ML <a href="https://cloud.google.com/bigquery/pricing" target="_blank">(pricing)</a>
* Vertex AI API <a href="https://cloud.google.com/vertex-ai/pricing" target="_blank">(pricing)</a>

Check out the [BQML Pricing page](https://cloud.google.com/bigquery/pricing#bqml) for a breakdown of costs are applied across these services.

Use the [Pricing
Calculator](https://cloud.google.com/products/calculator)
to generate a cost estimate based on your projected usage.

### Installation

Install the following packages required to execute this notebook.


```
%pip install --upgrade --user google-cloud-bigquery-connection google-cloud-aiplatform
```

#### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
# Automatically restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ Before proceeding, please wait for the kernel to finish restarting ⚠️</b>
</div>

#### Set the project and BigQuery region

You will need to set the `PROJECT_ID` and the `REGION` variable when creating the BigQuery dataset and Cloud resource connection.

For now, only the `us` multi-region and `us-central1` single region are supported for remote model services in BigQuery.

**For this notebook, set the region to `US` to ensure access to all public datasets used below.**

Learn more about [BigQuery public dataset regions](https://cloud.google.com/bigquery/public-data?gad=1&gclid=CjwKCAjw_aemBhBLEiwAT98FMhtM2q0Il2M4xU_eLwO_mAJpaZuuzBlQCNEkHKDDI-snZyGguxqnaRoCBdYQAvD_BwE&gclsrc=aw.ds#public_dataset_locations).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
REGION = "US"  # @param {type: "string"}
```

#### Setup project variables

These variables will be used throughout this notebook


*   **DATASET_ID:** ID of BigQuery dataset
*   **CONN_NAME**: Name of a BigQuery connector that will be used to connect to Vertex AI services
*   **LLM_MODEL_NAME**: Name given to the LLM created in BigQuery



```
DATASET_ID = "bqml_llm"
CONN_NAME = "bqml_llm_conn"
LLM_MODEL_NAME = "bqml-vertex-llm"
```

### Authenticating your notebook environment
If you are using **Colab**, you will need to authenticate yourself first. The next cell will check if you are currently using Colab, and will start the authentication process.

If you are using **Vertex AI Workbench**, you will not require additional authentication. For more information, you can check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Import libraries



```
from google.cloud import bigquery
from google.cloud import bigquery_connection_v1 as bq_connection
import pandas as pd

pd.set_option("display.max_colwidth", 1000)
```

### Create BigQuery Cloud resource connection

You will need to create a [Cloud resource connection](https://cloud.google.com/bigquery/docs/create-cloud-resource-connection) to enable BigQuery to interact with Vertex AI services.

You may need to first [enable the BigQuery Connection API](https://console.developers.google.com/apis/api/bigqueryconnection.googleapis.com/overview).


```
client = bq_connection.ConnectionServiceClient()
new_conn_parent = f"projects/{PROJECT_ID}/locations/{REGION}"
exists_conn_parent = f"projects/{PROJECT_ID}/locations/{REGION}/connections/{CONN_NAME}"
cloud_resource_properties = bq_connection.CloudResourceProperties({})

# Try to use an existing connection if one already exists. If not, create a new one.
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

### Set permissions for Service Account
The resource connection service account requires certain project-level permissions which are outlined in the <a href="https://cloud.google.com/bigquery/docs/bigquery-ml-remote-model-tutorial#set_up_access" target="_blank">Vertex AI function documentation</a>.

<br>

**Note:** If you are using **Vertex AI Workbench**, the service account used by Vertex AI may not have sufficient permissions to add IAM policy bindings. You may see the error:
> `ERROR: (gcloud.projects.add-iam-policy-binding) User [12345-compute@developer.gserviceaccount.com] does not have permission to access projects instance [my-project-id:setIamPolicy] (or it may not exist): Policy update access denied.`

If you see the above error with Vertex AI Workbench, open a Terminal (File -> New -> Terminal), and then authenticate your Google Cloud user account with: `gcloud auth login` and follow the steps therein. Once authenticated, return to this notebook and run the cell below. Alternatively, you can set the IAM roles 


```
gcloud_serviceusage = f"""
gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member="{CONN_SERVICE_ACCOUNT}" --role="roles/serviceusage.serviceUsageConsumer"
"""

gcloud_bigquery = f"""
gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member="{CONN_SERVICE_ACCOUNT}" --role="roles/bigquery.connectionUser"
"""

gcloud_aiplatform = f"""
gcloud projects add-iam-policy-binding {PROJECT_ID} --condition=None --no-user-output-enabled --member="{CONN_SERVICE_ACCOUNT}" --role="roles/aiplatform.user"
"""

print(gcloud_serviceusage)
!$gcloud_serviceusage #execute gcloud script

print(gcloud_bigquery)
!$gcloud_bigquery #execute gcloud script

print(gcloud_aiplatform)
!$gcloud_aiplatform #execute gcloud script
```

You can confirm that the three IAM roles have been set by running the cell below:
- roles/aiplatform.user
- roles/bigquery.connectionUser
- roles/serviceusage.serviceUsageConsumer


```
!gcloud projects get-iam-policy $PROJECT_ID  \
--flatten="bindings[].members" \
--format="table(bindings.role)" \
--filter="bindings.members:$CONN_SERVICE_ACCOUNT"
```

## Prepare BigQuery Dataset

### Create a BigQuery Dataset
You will need a BigQuery dataset to store your ML model and tables. This dataset must be created in the same region used by the BigQuery Cloud resource connection.

Run the following to create a dataset within your project:


```
client = bigquery.Client(project=PROJECT_ID)

dataset_id = f"""{PROJECT_ID}.{DATASET_ID}"""
dataset = bigquery.Dataset(dataset_id)
dataset.location = REGION

dataset = client.create_dataset(dataset, exists_ok=True)

print(f"Dataset {dataset.dataset_id} created.")
```

Create a wrapper to use the BigQuery client to run queries and return the result:


```
# Wrapper to use BigQuery client to run query and return result


def run_bq_query(sql: str):
    """
    Input: SQL query, as a string, to execute in BigQuery
    Returns the query results or error, if any
    """
    try:
        query_job = client.query(sql)
        result = query_job.result()
        print(f"JOB ID: {query_job.job_id} STATUS: {query_job.state}")
        return result

    except Exception as e:
        raise Exception(str(e))
```

## Using LLMs with BigQuery ML

To use LLMs with BigQuery ML you will first need to configure the LLM and then execute the `ML.GENERATE_TEXT` function with a prompt. This can all be done in SQL.

### Configure Vertex AI Model

You can configure a Vertex AI remote model in BigQuery using the CREATE MODEL statement:


```
sql = f"""
      CREATE OR REPLACE MODEL
        `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`
        REMOTE WITH CONNECTION
          `{PROJECT_ID}.{REGION}.{CONN_NAME}`
          OPTIONS ( remote_service_type = 'CLOUD_AI_LARGE_LANGUAGE_MODEL_V1');
      """
result = run_bq_query(sql)
```

### Using the LLM
You can use the LLMs in BQML by executing the `ML.GENERATE_TEXT` function against free text or data stored in BigQuery.

[The BQML documentation](https://cloud.google.com/bigquery/docs/generate-text#generate_text) gives further details on the parameters used: `temperature, max_output_tokens, top_p and top_k.`

*Note: The table column with the input text must have the alias 'prompt'*



```
PROMPT = "Describe a cat in one paragraph"

sql = f"""
          SELECT
            *
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                '{PROMPT}' AS prompt
              ),
              STRUCT
              (
                1 AS temperature,
                1024 AS max_output_tokens,
                0.8 AS top_p,
                40 AS top_k,
                TRUE AS flatten_json_output
              ));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

In this case, the LLM responded to the simple prompt to describe a cat in one paragraph.

The LLM response is returned as a table of results in BigQuery. The table includes JSON that can be parsed to extract the content result.

Setting the `flatten_json_output` parameter to TRUE will return a flattened JSON as a string: `ml_generate_text_llm_result`.

For the rest of the examples, you can just display the prompt and `ml_generate_text_llm_result` for simplicity.

## Example Use Cases

The following examples explore using the BQML LLM for content creation, text summarization, classification, entity recognition, data enrichment and sentiment analysis.

When writing your own prompts, we recommend you first review these [Prompt Design best practices](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/prompts/intro_prompt_design.ipynb).

#### Text Classification

This example categorizes news articles into one of the following categories: tech, sport, business, politics, or entertainment. The articles are stored in the BigQuery BBC News public dataset.


```
PROMPT = "Please categorize this BBC news article into either tech, sport, business, politics, or entertainment and return the category. Here is an example. News article: Intel has unveiled research that could mean data is soon being moved around chips at the speed of light., Category: Tech "

sql = f"""
          SELECT
            body AS article_body,
            CONCAT('{PROMPT}','News article: ', 'article_body', ', Category:') as prompt_template,
            ml_generate_text_llm_result as llm_result
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT}','News article: ', body, ', Category:') AS prompt,
                body
              FROM
                `bigquery-public-data.bbc_news.fulltext`
              LIMIT
                5),
              STRUCT(1 AS temperature, 1024 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k, TRUE AS flatten_json_output));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

#### Text Summarization

This example rewrites news articles stored in the BigQuery BBC News public dataset into text that can be understood by a 12 year old.


```
PROMPT = "Please rewrite this article to enable easier understanding for a 12 year old. Article: "

sql = f"""
          SELECT
            body as article_before,
            CONCAT('{PROMPT}', 'article_before') as prompt_template,
            ml_generate_text_llm_result as llm_result
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT}', body) AS prompt,
                body
              FROM  `bigquery-public-data.bbc_news.fulltext`
              LIMIT 5
              ),
              STRUCT(1 AS temperature, 1024 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k, TRUE AS flatten_json_output));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

This example summarizes lengthy news articles stored in the BigQuery BBC News public dataset into 25 words or less


```
PROMPT = "Please summarize this BBC news article into 25 words or less: "

sql = f"""
          SELECT
            body as article_before,
            ARRAY_LENGTH(SPLIT(body, ' ')) AS word_count_before,
            CONCAT('{PROMPT}') as prompt_template,
            ml_generate_text_llm_result as article_after,
            ARRAY_LENGTH(SPLIT(ml_generate_text_llm_result, ' ')) AS word_count_after,
            1-ARRAY_LENGTH(SPLIT(ml_generate_text_llm_result, ' '))/ARRAY_LENGTH(SPLIT(body, ' ')) as percent_reduction_words
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT}', body) AS prompt,
                body
              FROM
                `bigquery-public-data.bbc_news.fulltext`
              LIMIT
                5),
              STRUCT(1 AS temperature, 1024 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k, TRUE AS flatten_json_output));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

The word count of the results may not always be within the 25 words requested and so further [prompt engineering](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/introduction-prompt-design) may be required.

#### Entity Recognition

This example extracts the sentences from news articles that contain a statistic. The articles are stored in the BigQuery BBC News public dataset.


```
PROMPT = "Please return a bullet-point list of all sentences in this article that cite a statistic: "

sql = f"""
          SELECT
            body AS article_body,
            CONCAT('{PROMPT}', 'article_body') AS prompt,
            ml_generate_text_llm_result AS llm_result
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT}', body) AS prompt,
                body
              FROM
                `bigquery-public-data.bbc_news.fulltext`
              LIMIT
                5),
              STRUCT(1 AS temperature, 1024 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k, TRUE AS flatten_json_output));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

This example extracts the brand names from product descriptions stored in the BigQuery thelook_ecommerce public dataset.


```
PROMPT = "Please return the brand name listed in this product description. Here is an example. Product: TYR Sport Mens Solid Durafast Jammer Swim Suit, Brand: TYR ; Product: "

sql = f"""
          SELECT
            name AS product_description,
            CONCAT('{PROMPT}', 'product_description,',' Brand: ') as prompt_template,
            ml_generate_text_llm_result as llm_result
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT}', name,' Brand: ') AS prompt,
                name
              FROM
                `bigquery-public-data.thelook_ecommerce.products`
              LIMIT
                5),
              STRUCT(1 AS temperature, 1024 AS max_output_tokens, 0.8 AS top_p, 40 AS top_k, TRUE AS flatten_json_output));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

#### Sentiment Analysis

This example runs sentiment analysis on movie reviews stored in the BigQuery IMDB public dataset to determine whether the movie review is positive, negative or neutral.


```
PROMPT = "Please categorize this movie review as either Positive, Negative or Neutral. Here is an example. Review: I dislike this movie, Sentiment: Negative"

sql = f"""
          SELECT
            review,
            CONCAT('{PROMPT}',' Review: review Sentiment:') as prompt_template,
            ml_generate_text_llm_result as llm_result

          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT}',' Review: ', review, ', Sentiment:') AS prompt,
                review
              FROM
                `bigquery-public-data.imdb.reviews`
              WHERE
                UPPER(title) = 'TROY'
              LIMIT
                10 ),
              STRUCT(0.2 AS temperature,
                50 AS max_output_tokens,
                0.8 AS top_p,
                40 AS top_k, TRUE AS flatten_json_output))
          """
result = run_bq_query(sql)
result.to_dataframe()
```

#### Content Creation


This examples creates a marketing campaign email based on recipient demographic and spending data. Commerce data is taken from BigQuery's [thelook eCommerce public dataset](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce).

First, you will need to join the order_items, products, and users tables of the dataset in order to get a table that includes the information for the email, including the item purchased, description of that item, and demographic data about the purchaser.


```
sql = f"""
      CREATE OR REPLACE TABLE
        `{PROJECT_ID}.{DATASET_ID}.purchases` AS
      SELECT
        u.id,
        u.first_name,
        u.email,
        u.postal_code,
        u.country,
        o.order_id,
        o.created_at,
        p.category,
        p.name
      FROM
        `bigquery-public-data.thelook_ecommerce.users` u
      JOIN (
        SELECT
          user_id,
          order_id,
          created_at,
          product_id,
          ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
        FROM
          `bigquery-public-data.thelook_ecommerce.order_items`
      ) o
      ON
        u.id = o.user_id
      JOIN
        `bigquery-public-data.thelook_ecommerce.products` p
      ON
        o.product_id = p.id
      WHERE
        o.rn = 1 AND p.category = "Active" AND u.country = "United States";
       """

result = run_bq_query(sql)
```

Querying the new table, you will see a comprehensive set of data for each purchase.


```
sql = f"""
        SELECT
            *
        FROM
            `{PROJECT_ID}.{DATASET_ID}.purchases`
        LIMIT
            10;
      """
result = run_bq_query(sql)
result.to_dataframe().head(10)
```

Now you will prepare the prompt for the LLM, incorporating the item's name and the purchaser's name and postal code.



```
PROMPT_PART1 = "A user bought a product with this description: "
PROMPT_PART2 = ' Write a follow up marketing email mentioning the high-level product category of their purchase in one word, for example "We hope you are enjoying your new t-shirt". '
PROMPT_PART3 = "Encourage the individual to shop with the store again using the coupon code RETURN10 for 10% off their next purchase. "
PROMPT_PART4 = "Provide two local outdoor activities they could pursue with their new purchase. They live in the zip code "
PROMPT_PART5 = '. Do not mention the brand of the product, just sign off the email with "-TheLook." Address the email to: '

sql = f"""
          SELECT
            prompt,
            ml_generate_text_llm_result
          FROM
            ML.GENERATE_TEXT(
              MODEL `{PROJECT_ID}.{DATASET_ID}.{LLM_MODEL_NAME}`,
              (
              SELECT
                CONCAT('{PROMPT_PART1}',name,'{PROMPT_PART2}','{PROMPT_PART3}','{PROMPT_PART4}',postal_code,'{PROMPT_PART5}',first_name) AS prompt
              FROM
                `{PROJECT_ID}.{DATASET_ID}.purchases`
              LIMIT
                5),
              STRUCT(1 AS temperature,
                1024 AS max_output_tokens,
                0.8 AS top_p,
                40 AS top_k,
                TRUE AS flatten_json_output));
        """
result = run_bq_query(sql)
result.to_dataframe()
```

## Cleaning Up
To clean up all Google Cloud resources used in this project, you can <a href="https://cloud.google.com/resource-manager/docs/creating-managing-projects#shutting_down_projects" target="_blank">delete the Google Cloud
project</a> you used for the tutorial.

Otherwise, you can delete the individual resources you created in this tutorial by uncommenting the below:


```
# # Delete BigQuery dataset, including the BigQuery ML models you just created, and the BigQuery Connection
# ! bq rm -r -f $PROJECT_ID:$DATASET_ID
# ! bq rm --connection --project_id=$PROJECT_ID --location=$REGION $CONN_NAME
```

## Wrap Up

In this tutorial we have shown how to integrate BQML with Vertex AI LLMs, and given examples of how the new `ML.GENERATE_TEXT` function can be applied directly to data stored in BigQuery.

Check out our [BigQuery ML LLM page](https://cloud.google.com/bigquery/docs/inference-overview#generative_ai) to learn more about remote models and generative AI in BigQuery.
