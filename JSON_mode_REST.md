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

# Gemini API: JSON Mode Quickstart with REST

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/rest/JSON_mode_REST.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

The Gemini API can be used to generate a JSON output if you set the schema that you would like to use. This notebook provides a code example that shows you how to get started with JSON mode using `curl`.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

### Authentication

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](../../quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata
```


```
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Activate JSON Mode

To activate JSON mode, set `response_mime_type` to `application/json` in the `generationConfig`.


```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
    "contents": [{
      "parts":[
        {"text": "List a few popular cookie recipes using this JSON schema:
          {'type': 'object', 'properties': { 'recipe_name': {'type': 'string'}}}"
          }
        ]
    }],
    "generationConfig": {
        "response_mime_type": "application/json",
    }
}' 2> /dev/null | head
```

    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "[{\"recipe_name\":\"Chocolate Chip Cookies\"},{\"recipe_name\":\"Peanut Butter Cookies\"},{\"recipe_name\":\"Oatmeal Raisin Cookies\"},{\"recipe_name\":\"Sugar Cookies\"},{\"recipe_name\":\"Shortbread Cookies\"}] \n"
              }
            ],
            "role": "model"
    

To turn off JSON mode, set `response_mime_type` to `text/plain` (or omit the `response_mime_type` parameter).

## Next Steps
### Useful API references:

Check the [structured ouput](https://ai.google.dev/gemini-api/docs/structured-output) documentation or the [`GenerationConfig`](https://ai.google.dev/api/generate-content#generationconfig) API reference for more details 

### Related examples

* The constrained output is used in the [Text summarization](../../examples/json_capabilities/Text_Summarization.ipynb) example to provide the model a format to summarize a story (genre, characters, etc...)
* The [Object detection](../../examples/Object_detection.ipynb) examples are using the JSON constrained output to normalize the output of the detection.

### Continue your discovery of the Gemini API

JSON is not the only way to constrain the output of the model, you can also use an [Enum](../../quickstarts/Enum.ipynb). [Function calling](../../quickstarts/Function_calling.ipynb) and [Code execution](../../quickstarts/Code_Execution.ipynb) are other ways to enhance your model by using your own functions or by letting the model write and run them.
