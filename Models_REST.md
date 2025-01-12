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

# Gemini API: Models with REST

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/rest/Models_REST.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


This notebook demonstrates how to list the models that are available for you to use in the Gemini API, and how to find details about a model in `curl`.

You can run this in Google Colab, or you can copy/paste the curl commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata
```


```
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Model info

### List models

If you `GET` the models directory, it uses the `list` method to list all of the models available through the API, including both the Gemini and PaLM family models.


```bash
%%bash

curl https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY
```

### Get model

If you `GET` a model's URL, the API uses the `get` method to return information about that model such as version, display name, input token limit, etc.


```bash
%%bash

curl https://generativelanguage.googleapis.com/v1beta/models/gemini-pro?key=$GOOGLE_API_KEY
```

## Learning more

To learn how use a model for prompting, see the [Prompting](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Prompting_REST.ipynb) quickstart.

To learn how use a model for embedding, see the [Embedding](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Embeddings_REST.ipynb) quickstart.

For more information on models, visit the [Gemini models](https://ai.google.dev/models/gemini) documentation.
