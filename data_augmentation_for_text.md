```
# Copyright 2024 Google LLC
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

# Data Augmentation for Text Data using BigQuery DataFrames

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-augmentation/data_augmentation_for_text.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fdata-augmentation%2Fdata_augmentation_for_text.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/data-augmentation/data_augmentation_for_text.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-augmentation/data_augmentation_for_text.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/bigquery/v1/32px.svg" alt="BigQuery Studio logo"><br> Open in BigQuery Studio
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/data-augmentation/data_augmentation_for_text.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | | |
|-|-|-|
|Author(s) | [Karl Weinmeister](https://github.com/kweinmeister)| [Kaz Sato](https://github.com/kazunori279)|

## Overview

Data augmentation is a technique used to expand the size and diversity of a training dataset with synthetic data. This is particularly beneficial when dealing with limited datasets, which can lead to overfitting.

It can be applied to multiple types of data through techniques including:
* Text: Synonym replacement, word shuffling, or adding typos can be used to create new variations of text data.
* Images: Flipping, rotating, cropping, scaling, adjusting brightness or contrast, and adding noise can all be used to generate new image variations.
* Audio: Time-warping, adding noise, or pitch-shifting can be used to modify audio data.

In this notebook, we will demonstrate 3 different techniques for augmenting data on text data: synonym replacement, back-translation, and noise injection.


## Getting Started

### Install BigQuery DataFrames

We will augment our data using Gemini for each of these techniques. Since we'll use a BigQuery public dataset, BigQuery DataFrames provides a natural interface to perform bulk LLM request operations on the original data.

Learn more about [BigQuery DataFrames](https://cloud.google.com/python/docs/reference/bigframes/latest).


```
%pip install --upgrade --user --quiet bigframes
```

### Restart runtime (Colab only)

To use the newly installed packages, you must restart the runtime on Google Colab.


```
import sys

if "google.colab" in sys.modules:
    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

Authenticate your environment on Google Colab.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Imports


```
from bigframes.ml.llm import GeminiTextGenerator
import bigframes.pandas as bpd
```

### Set Google Cloud project information

To get started using BigQuery DataFrames, you must have an existing Google Cloud project and [enable the BigQuery API](https://console.cloud.google.com/flows/enableapi?apiid=bigquery.googleapis.com).


```
PROJECT_ID = "your-project-id"  # @param {type:"string"}
LOCATION = "US"  # @param {type:"string"}

# Note: The project option is not required in all environments.
# On BigQuery Studio, the project ID is automatically detected.
bpd.options.bigquery.project = PROJECT_ID

# Note: The location option is not required.
# It defaults to the location of the first table or query
# passed to read_gbq(). For APIs where a location can't be
# auto-detected, the location defaults to the "US" location.
bpd.options.bigquery.location = LOCATION
```

## Load data

We will use the StackOverflow data on BigQuery Public Datasets, limiting to questions with the `python` tag, and accepted answers for answers since 2020-01-01.


```
stack_overflow_df = bpd.read_gbq_query(
    """SELECT
           CONCAT(q.title, q.body) AS input_text,
           a.body AS output_text
       FROM `bigquery-public-data.stackoverflow.posts_questions` q
       JOIN `bigquery-public-data.stackoverflow.posts_answers` a
         ON q.accepted_answer_id = a.id
       WHERE q.accepted_answer_id IS NOT NULL
         AND REGEXP_CONTAINS(q.tags, "python")
         AND a.creation_date >= "2020-01-01"
       LIMIT 550
    """
)

stack_overflow_df.peek()
```

## Synonym Replacement

Synonym replacement involves replacing words in a sentence with their synonyms to create variations of the original text. This helps to introduce diversity into the dataset, which can improve the model's ability to generalize with unseen data. For instance, if the original sentence is "Create easily interpretable topics with Large Language Models", a synonym replacement approach might replace "large" with "big" to create "Create easily interpretable topics with Big Language Models".


```
# Constants
n_sample_dataset_rows = 10  # How many rows to sample for synonym replacement
n_replacement_words = 5  # How many words per input text to replace

# Sample a number of rows from the original dataframe
df = stack_overflow_df.sample(n_sample_dataset_rows, random_state=42)
```

Define a Gemini text generator LLM [model](https://cloud.google.com/python/docs/reference/bigframes/latest/bigframes.ml.llm.GeminiTextGenerator):


```
model = GeminiTextGenerator()
```


```
# Create a prompt with the synonym replacement instructions and the input text
df["synonym_prompt"] = (
    f"Replace {n_replacement_words} words from the input text with synonyms, "
    + "keeping the overall meaning as close to the original text as possible."
    + "Only provide the synonymized text, with no additional explanation."
    + "Preserve the original formatting.\n\nInput text: "
    + df["input_text"]
)

# Run batch job and assign to a new column
df["input_text_with_synonyms"] = model.predict(
    df["synonym_prompt"]
).ml_generate_text_llm_result

# Compare the original and new columns
df.peek()[["input_text", "input_text_with_synonyms"]]
```

## Back translation

Back translation is a data augmentation technique used in NLP to artificially expand your training dataset. It works by translating your text data into another language and then translating it back to the original language. This process injects variations into your data due to the imperfections of machine translation.

Here's a breakdown of the steps involved:

* Original Text: You start with your original text data that you want to augment.
* Translation (Round Trip):  This text is then translated into a different language using a machine translation model.
* Back Translation: The translated text is then translated back to the original language.


```
# Create a prompt with the translation instructions and the input text
df["translation_prompt"] = (
    "Translate the input text from English to Spanish. The text may include HTML characters; preserve them in the text.\n\nInput text: "
    + df["input_text"]
)

# Run batch job and assign to a new column
df["input_text_translated"] = model.predict(
    df["translation_prompt"]
).ml_generate_text_llm_result

# Compare the original and new columns
df.peek()[["input_text", "input_text_translated"]]
```


```
# Create a prompt with the back-translation instructions and the input text
df["backtranslation_prompt"] = (
    "Translate the input text from Spanish to English. The text may include HTML characters; preserve them in the text.\n\nInput text: "
    + df["input_text_translated"]
)

# Run batch job and assign to a new column
df["input_text_backtranslated"] = model.predict(
    df["backtranslation_prompt"]
).ml_generate_text_llm_result

# Compare the original and new columns
df.peek()[["input_text", "input_text_translated", "input_text_backtranslated"]]
```

## Noise injection

Noise injection is the deliberate introduction of random perturbations or alterations into training data. The goal is to make the model more robust and prevent overfitting.

Here we will apply a 3-step noise injection approach described in the paper "[Understanding Back-Translation at Scale](https://arxiv.org/pdf/1808.09381.pdf)" by Edunov et al. (2018)

This process involves:
1. Word Deletion: Randomly removing a percentage of words from the input text.
1. Word Replacement: Replacing another percentage of words in the text with a filler token.
1. Word Rearranging: Randomly shuffling the order of words, with the restriction that words can't be moved more than a specific distance from their original position.


```
# Constants
delete_ratio = 0.1
filler_ratio = 0.1
filler_token = "BLANK"
swap_range = 3
```


```
# Create a prompt with the delete instructions and the input text
df["deletion_prompt"] = (
    f"Delete words from the input text with probability {delete_ratio}. Preserve all HTML tags.\n\nInput text:"
    + df["input_text"]
)

# Run batch job and assign to a new column
df["input_text_deleted"] = model.predict(
    df["deletion_prompt"]
).ml_generate_text_llm_result

# View results
df.peek()[["input_text", "input_text_deleted"]]
```


```
# Create a prompt with the filler instructions and the input text
df["filler_prompt"] = (
    f"Replace words from the input text with probability {filler_ratio} with this filler word {filler_token}. Preserve all HTML tags.\n\nInput text:"
    + df["input_text_deleted"]
)

# Run batch job and assign to a new column
df["input_text_filler"] = model.predict(df["filler_prompt"]).ml_generate_text_llm_result

# View results
df.peek()[["input_text", "input_text_filler"]]
```


```
# Create a prompt with the swap instructions and the input text
df["swap_prompt"] = (
    f"Rearrange words from the input text no more than {swap_range} words apart. Preserve all HTML tags.\n\nInput text:"
    + df["input_text_filler"]
)

# Run batch job and assign to a new column
df["input_text_swap"] = model.predict(df["swap_prompt"]).ml_generate_text_llm_result

# View results
df.peek()[["input_text", "input_text_swap"]]
```
