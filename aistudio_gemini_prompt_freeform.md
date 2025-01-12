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
import base64
import copy
import json
import pathlib
import requests


import PIL.Image
import IPython.display
from IPython.display import Markdown

try:
    # The SDK will automatically read it from the GOOGLE_API_KEY environment variable.
    # In Colab get the key from Colab-secrets ("🔑" in the left panel).
    import os
    from google.colab import userdata

    os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
except ImportError:
    pass

import google.generativeai as genai

# Parse the arguments

model = "gemini-1.5-flash"  # @param {isTemplate: true}
contents_b64 = "W3sicGFydHMiOiBbeyJ0ZXh0IjogIldoYXQncyBpbiB0aGlzIHBpY3R1cmU/In0sIHsiZmlsZV9kYXRhIjogeyJ1cmwiOiAiaHR0cHM6Ly9zdG9yYWdlLmdvb2dsZWFwaXMuY29tL2dlbmVyYXRpdmVhaS1kb3dubG9hZHMvaW1hZ2VzL3Njb25lcy5qcGciLCAibWltZV90eXBlIjogImltYWdlL2pwZWcifX1dfV0="  # @param {isTemplate: true}
generation_config_b64 = "e30="  # @param {isTemplate: true}
safety_settings_b64 = "e30="  # @param {isTemplate: true}

gais_contents = json.loads(base64.b64decode(contents_b64))

generation_config = json.loads(base64.b64decode(generation_config_b64))
safety_settings = json.loads(base64.b64decode(safety_settings_b64))

stream = False

# Convert and upload the files

tempfiles = pathlib.Path(f"tempfiles")
tempfiles.mkdir(parents=True, exist_ok=True)


drive = None
def upload_file_data(file_data, index):
    """Upload files to the Files API.

    For each file, Google AI Studio either sent:
    - a Google Drive ID,
    - a URL,
    - a file path, or
    - The raw bytes (`inline_data`).

    The API only understands `inline_data` or it's Files API.
    This code, uploads files to the files API where the API can access them.
    """

    mime_type = file_data["mime_type"]
    if drive_id := file_data.pop("drive_id", None):
        if drive is None:
          from google.colab import drive
          drive.mount("/gdrive")

        path = next(
            pathlib.Path(f"/gdrive/.shortcut-targets-by-id/{drive_id}").glob("*")
        )
        print("Uploading:", str(path))
        file_info = genai.upload_file(path=path, mime_type=mime_type)
        file_data["file_uri"] = file_info.uri
        return

    if url := file_data.pop("url", None):
        response = requests.get(url)
        data = response.content
        name = url.split("/")[-1]
        path = tempfiles / str(index)
        path.write_bytes(data)
        print("Uploading:", url)
        file_info = genai.upload_file(path, display_name=name, mime_type=mime_type)
        file_data["file_uri"] = file_info.uri
        return

    if name := file_data.get("filename", None):
        if not pathlib.Path(name).exists():
            raise IOError(
                f"local file: `{name}` does not exist. You can upload files "
                'to Colab using the file manager ("📁 Files" in the left '
                "toolbar)"
            )
        file_info = genai.upload_file(path, display_name=name, mime_type=mime_type)
        file_data["file_uri"] = file_info.uri
        return

    if "inline_data" in file_data:
        return

    raise ValueError("Either `drive_id`, `url` or `inline_data` must be provided.")


contents = copy.deepcopy(gais_contents)

index = 0
for content in contents:
    for n, part in enumerate(content["parts"]):
        if file_data := part.get("file_data", None):
            upload_file_data(file_data, index)
            index += 1

import json
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

## [optional] Show the conversation

This section displays the conversation received from Google AI Studio.


```
# @title Show the conversation, in colab.
import mimetypes

def show_file(file_data):
    mime_type = file_data["mime_type"]

    if drive_id := file_data.get("drive_id", None):
        path = next(
            pathlib.Path(f"/gdrive/.shortcut-targets-by-id/{drive_id}").glob("*")
        )
        name = path
        # data = path.read_bytes()
        kwargs = {"filename": path}
    elif url := file_data.get("url", None):
        name = url
        kwargs = {"url": url}
        # response = requests.get(url)
        # data = response.content
    elif data := file_data.get("inline_data", None):
        name = None
        kwargs = {"data": data}
    elif name := file_data.get("filename", None):
        if not pathlib.Path(name).exists():
            raise IOError(
                f"local file: `{name}` does not exist. You can upload files to "
                'Colab using the file manager ("📁 Files"in the left toolbar)'
            )
    else:
        raise ValueError("Either `drive_id`, `url` or `inline_data` must be provided.")

        print(f"File:\n    name: {name}\n    mime_type: {mime_type}\n")
        return

    format = mimetypes.guess_extension(mime_type).strip(".")
    if mime_type.startswith("image/"):
        image = IPython.display.Image(**kwargs, width=256)
        IPython.display.display(image)
        print()
        return

    if mime_type.startswith("audio/"):
        if len(data) < 2**12:
            audio = IPython.display.Audio(**kwargs)
            IPython.display.display(audio)
            print()
            return

    if mime_type.startswith("video/"):
        if len(data) < 2**12:
            audio = IPython.display.Video(**kwargs, mimetype=mime_type)
            IPython.display.display(audio)
            print()
            return

    print(f"File:\n    name: {name}\n    mime_type: {mime_type}\n")


for content in gais_contents:
    if role := content.get("role", None):
        print("Role:", role, "\n")

    for n, part in enumerate(content["parts"]):
        if text := part.get("text", None):
            print(text, "\n")

        elif file_data := part.get("file_data", None):
            show_file(file_data)

    print("-" * 80, "\n")
```
