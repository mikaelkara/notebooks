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

# Intro to Context Caching with the Gemini API

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/context-caching/intro_context_caching.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fcontext-caching%2Fintro_context_caching.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/context-caching/intro_context_caching.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/context-caching/intro_context_caching.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong)|

## Overview

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases.

### Context Caching

The Gemini API provides the context caching feature for developers to store frequently used input tokens in a dedicated cache and reference them for subsequent requests, eliminating the need to repeatedly pass the same set of tokens to a model. This feature can help reduce the number of tokens sent to the model, thereby lowering the cost of requests that contain repeat content with high input token counts.

### Objectives

In this tutorial, you learn how to use the Gemini API context caching feature in Vertex AI.

You will complete the following tasks:
- Create a context cache
- Retrieve and use a context cache
- Use context caching in Chat
- Update the expire time of a context cache
- Delete a context cache


## Get started

### Install Vertex AI SDK and other required packages



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

## Code Examples

### Import libraries


```
import datetime

import vertexai
from vertexai.generative_models import Part
from vertexai.preview import caching
from vertexai.preview.generative_models import GenerativeModel
```

### Create a context cache

**Note**: Context caching is only available for stable models with fixed versions (for example, `gemini-1.5-pro-001`). You must include the version postfix (for example, the `-001` in `gemini-1.5-pro-001`).

For more information, see [Available Gemini stable model versions](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning#stable-versions-available).



```
MODEL_ID = "gemini-1.5-pro-001"  # @param {type:"string"}
```

Context caching is particularly well suited to scenarios where a substantial initial context is referenced repeatedly by shorter requests.

- Cached content can be any of the MIME types supported by Gemini multimodal models. For example, you can cache a large amount of text, audio, or video. **Note**: The minimum size of a context cache is 32,769 tokens.
- The default expiration time of a context cache is 60 minutes. You can specify a different expiration time using the `ttl` (time to live) or the `expire_time` property.

This example shows how to create a context cache using two large research papers stored in a Cloud Storage bucket, and set the `ttl` to 60 minutes.

- Paper 1: [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)
- Paper 2: [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](https://arxiv.org/abs/2403.05530)



```
system_instruction = """
You are an expert researcher who has years of experience in conducting systematic literature surveys and meta-analyses of different topics.
You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts.
Now look at the research paper below, and answer the following questions in 1-2 sentences.
"""

contents = [
    Part.from_uri(
        "gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
        mime_type="application/pdf",
    ),
    Part.from_uri(
        "gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
        mime_type="application/pdf",
    ),
]

cached_content = caching.CachedContent.create(
    model_name=MODEL_ID,
    system_instruction=system_instruction,
    contents=contents,
    ttl=datetime.timedelta(minutes=60),
)
```

You can access the properties of the cached content as example below. You can use its `name` or `resource_name` to reference the contents of the context cache.

**Note**: The `name` of the context cache is also referred to as cache ID.


```
print(cached_content.name)
print(cached_content.resource_name)
print(cached_content.model_name)
print(cached_content.create_time)
print(cached_content.expire_time)
```

### Retrieve and use a context cache

You can use the property `name` or `resource_name` to reference the contents of the context cache. For example:
```
new_cached_content = caching.CachedContent(cached_content_name=cached_content.name)
```

To use the context cache, you construct a `GenerativeModel` with the context cache.


```
model = GenerativeModel.from_cached_content(cached_content=cached_content)
```

Then you can query the model with a prompt, and the cached content will be used as a prefix to the prompt.


```
response = model.generate_content(
    "What is the research goal shared by these research papers?"
)

print(response.text)
```

### Use context caching in Chat

You can use the context cache in a multi-turn chat session.



```
chat = model.start_chat()
```


```
prompt = """
How do the approaches to responsible AI development and mitigation strategies in Gemini 1.5 evolve from those in Gemini 1.0?
"""

response = chat.send_message(prompt)

print(response.text)
```


```
prompt = """
Given the advancements presented in Gemini 1.5, what are the key future research directions identified in both papers
for further improving multimodal AI models?
"""

response = chat.send_message(prompt)

print(response.text)
```

You can use `print(chat.history)` to print out the chat session history.

### Update the expiration time of a context cache


The default expiration time of a context cache is 60 minutes. To update the expiration time, update one of the following properties:

`ttl` - The number of seconds and nanoseconds that the cache lives after it's created or after the `ttl` is updated before it expires. When you set the `ttl`, the cache `expire_time` is updated.

`expire_time` - A Timestamp that specifies the absolute date and time when the context cache expires.


```
cached_content.update(ttl=datetime.timedelta(hours=1))

cached_content.refresh()

print(cached_content.expire_time)
```

### Delete a context cache

You can remove content from the cache using the delete operation.


```
cached_content.delete()
```
