##### Copyright 2023 Google LLC


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

## Setup


```
!pip install -U -q "google-generativeai>=0.8.2"
```


```
# import necessary modules.

import google.generativeai as genai

import base64
import json

try:
    # Mount google drive
    from google.colab import drive

    drive.mount("/gdrive")

    # The SDK will automatically read it from the GOOGLE_API_KEY environment variable.
    # In Colab get the key from Colab-secrets ("🔑" in the left panel).
    import os
    from google.colab import userdata

    os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
except ImportError:
    pass

# Parse the arguments

model = "gemini-1.5-flash"  # @param {isTemplate: true}
contents_b64 = b'W3sicGFydHMiOiBbeyJ0ZXh0IjogIkhlbGxvIn1dfV0='
generation_config_b64 = "e30="  # @param {isTemplate: true}
safety_settings_b64 = "e30="  # @param {isTemplate: true}

contents = json.loads(base64.b64decode(contents_b64))

generation_config = json.loads(base64.b64decode(generation_config_b64))
safety_settings = json.loads(base64.b64decode(safety_settings_b64))

stream = False

print(json.dumps(contents, indent=4))
```

## Call `generate_content`


```
from IPython.display import display
from IPython.display import Markdown

# Call the model and print the response.
gemini = genai.GenerativeModel(model_name=model)

response = gemini.generate_content(
    contents,
    generation_config=generation_config,
    safety_settings=safety_settings,
    stream=stream,
)

display(Markdown(response.text))
```

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://ai.google.dev/gemini-api/docs"><img src="https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png" height="32" width="32" />Docs on ai.google.dev</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/google-gemini/cookbook/blob/main/quickstarts"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />More notebooks in the Cookbook</a>
  </td>
</table>
