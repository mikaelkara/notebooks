##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Embedding Quickstart with REST

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/rest/Embeddings_REST.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook provides quick code examples that show you how to get started generating embeddings using `curl`.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata
```


```
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Embed content

Call the `embed_content` method with the `text-embedding-004` model to generate text embeddings:


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/text-embedding-004",
    "content": {
    "parts":[{
      "text": "Hello world"}]}, }' 2> /dev/null | head
```

    {
      "embedding": {
        "values": [
          0.013168523,
          -0.008711934,
          -0.046782676,
          0.00069968984,
          -0.009518873,
          -0.008720178,
          0.060103577,
    

# Batch embed content

You can embed a list of multiple prompts with one API call for efficiency.



```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"requests": [{
      "model": "models/text-embedding-004",
      "content": {
      "parts":[{
        "text": "What is the meaning of life?"}]}, },
      {
      "model": "models/text-embedding-004",
      "content": {
      "parts":[{
        "text": "How much wood would a woodchuck chuck?"}]}, },
      {
      "model": "models/text-embedding-004",
      "content": {
      "parts":[{
        "text": "How does the brain work?"}]}, }, ]}' 2> /dev/null | grep -C 5 values
```

    {
      "embeddings": [
        {
          "values": [
            -0.010632277,
            0.019375855,
            0.0209652,
            0.0007706424,
            -0.061464064,
    --
            -0.0071538696,
            -0.028534694
          ]
        },
        {
          "values": [
            0.018467998,
            0.0054281196,
            -0.017658804,
            0.013859266,
            0.053418662,
    --
            0.026714385,
            0.0018762538
          ]
        },
        {
          "values": [
            0.05808907,
            0.020941721,
            -0.108728774,
            -0.04039259,
            -0.04440443,
    

## Set the output dimensionality
If you're using `text-embeddings-004`, you can set the `output_dimensionality` parameter to create smaller embeddings.

* `output_dimensionality` truncates the embedding (e.g., `[1, 3, 5]` becomes `[1,3]` when `output_dimensionality=2`).



```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/text-embedding-004",
    "output_dimensionality":256,
    "content": {
    "parts":[{
      "text": "Hello world"}]}, }' 2> /dev/null | head
```

    {
      "embedding": {
        "values": [
          0.013168523,
          -0.008711934,
          -0.046782676,
          0.00069968984,
          -0.009518873,
          -0.008720178,
          0.060103577,
    

## Use `task_type` to provide a hint to the model how you'll use the embeddings

Let's look at all the parameters the embed_content method takes. There are four:

* `model`: Required. Must be `models/embedding-001`.
* `content`: Required. The content that you would like to embed.
* `task_type`: Optional. The task type for which the embeddings will be used. See below for possible values.
* `title`: The given text is a document from a corpus being searched. Optionally, set the `title` parameter with the title of the document. Can only be set when `task_type` is `RETRIEVAL_DOCUMENT`.

`task_type` is an optional parameter that provides a hint to the API about how you intend to use the embeddings in your application.

The following task_type parameters are accepted:

* `TASK_TYPE_UNSPECIFIED`: If you do not set the value, it will default to retrieval_query.
* `RETRIEVAL_QUERY` : The given text is a query in a search/retrieval setting.
* `RETRIEVAL_DOCUMENT`: The given text is a document from the corpus being searched.
* `SEMANTIC_SIMILARITY`: The given text will be used for Semantic Textual Similarity (STS).
* `CLASSIFICATION`: The given text will be classified.
* `CLUSTERING`: The embeddings will be used for clustering.



```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{"model": "models/text-embedding-004",
    "content": {
    "parts":[{
      "text": "Hello world"}]},
    "task_type": "RETRIEVAL_DOCUMENT",
    "title": "My title"}' 2> /dev/null | head
```

    {
      "embedding": {
        "values": [
          0.060187872,
          -0.031515103,
          -0.03244149,
          -0.019341845,
          0.057285223,
          0.037159503,
          0.035636507,
    

## Learning more

* Learn more about text-embeddings-004 [here](https://developers.googleblog.com/2024/04/gemini-15-pro-in-public-preview-with-new-features.html).
*   See the [REST API reference](https://ai.google.dev/api/rest) to learn more.
*   Explore more examples in the cookbook.

