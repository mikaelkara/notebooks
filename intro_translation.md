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

# Getting Started with Translation

<table align="left">
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Flanguage%2Ftranslation%2Fintro_translation.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/translation/intro_translation.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/translation/intro_translation.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


---

* Author: Holt Skinner
* Created: Jan 2024

---

## Overview

This notebook demonstrates how to use the [Google Cloud Translation API](https://cloud.google.com/translate) to translate text in [130+ languages](https://cloud.google.com/translate/docs/languages).

### Objective

This tutorial uses the following Google Cloud AI services and resources:

- [Cloud Translation API](https://cloud.google.com/translate/docs/overview)
- Cloud Storage


### Costs

This tutorial uses billable components of Google Cloud:

* Cloud Translation
* Cloud Storage

Learn about [Translate pricing](https://cloud.google.com/translate/pricing),
and [Cloud Storage pricing](https://cloud.google.com/storage/pricing),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Getting Started


### Install Vertex AI SDK, other packages and their dependencies

Install the following packages required to execute this notebook.


```
# Install the packages
%pip install --user --upgrade -q google-cloud-translate
```

### Run the following cell to restart the kernel.


```
# Automatically restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>

Set the project and region.

* Please note the **available regions** for Translation, see [documentation](https://cloud.google.com/translate/docs/advanced/endpoints)


```
PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
```

### Authenticating your notebook environment

* If you are using **Colab** to run this notebook, run the cell below and continue.
* If you are using **Vertex AI Workbench**, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()

    ! gcloud config set project {PROJECT_ID}
    ! gcloud auth application-default login -q
```

Initialize the [Vertex AI SDK](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk)


```
import vertexai

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=VERTEXAI_LOCATION)
```

### Import libraries


```
from google.cloud import translate
```

### Create client


```
client = translate.TranslationServiceClient(
    # Optional: https://cloud.google.com/translate/docs/advanced/endpoints
    # client_options=ClientOptions(
    #     api_endpoint=f"translate-{TRANSLATE_LOCATION}.googleapis.com"
    # )
)
```

### Create helper functions


```
def translate_text(
    text: str,
    project_id: str = PROJECT_ID,
    location: str = LOCATION,
    glossary: str | None = None,
) -> translate.TranslateTextResponse:
    """Translating Text."""
    # Translate text from English to Spanish
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request=translate.TranslateTextRequest(
            parent=client.common_location_path(project_id, location),
            contents=[text],
            # Supported language codes: https://cloud.google.com/translate/docs/languages
            source_language_code="en",
            target_language_code="es",
            glossary_config=(
                translate.TranslateTextGlossaryConfig(glossary=glossary)
                if glossary
                else None
            ),
        )
    )

    return response


def create_glossary(
    input_uri: str,
    glossary_id: str,
    project_id: str = PROJECT_ID,
    location: str = LOCATION,
    timeout: int = 180,
) -> translate.Glossary:
    """
    Create a unidirectional glossary. Glossary can be words or
    short phrases (usually fewer than five words).
    https://cloud.google.com/translate/docs/advanced/glossary#format-glossary
    """
    glossary = translate.Glossary(
        name=client.glossary_path(project_id, location, glossary_id),
        # Supported language codes: https://cloud.google.com/translate/docs/languages
        language_pair=translate.Glossary.LanguageCodePair(
            source_language_code="en", target_language_code="es"
        ),
        input_config=translate.GlossaryInputConfig(
            gcs_source=translate.GcsSource(input_uri=input_uri)
        ),
    )

    # glossary is a custom dictionary Translation API uses
    # to translate the domain-specific terminology.
    operation = client.create_glossary(
        parent=client.common_location_path(project_id, location), glossary=glossary
    )

    result = operation.result(timeout)
    return result
```

Now let's try to translate a simple phrase from English to Spanish.


```
response = translate_text("Hi there!")

# Display the translation for each input text provided
for translation in response.translations:
    print(f"Translated text: {translation.translated_text}")
```

    Translated text: ¡Hola!
    

## Glossaries

That looks great! However, let's look at what happens if we try to translate a technical word, such as the Google Cloud product [Compute Engine](https://cloud.google.com/compute?hl=en).


```
response = translate_text("Compute Engine")

# Display the translation for each input text provided
for translation in response.translations:
    print(f"Translated text: {translation.translated_text}")
```

    Translated text: Motor de Computación
    

### Create a Glossary

Notice that the Translation API translated the name literally.

Suppose we want this name to be the same in all languages, we can create a [Glossary](https://cloud.google.com/translate/docs/advanced/glossary) to consistently translate domain-specific words and phrases.

Next, we'll create a glossary for lots of Google Cloud product names to indicate how they should be translated into Spanish.

We've already created an input TSV file and uploaded it to a publicly-accessible Cloud Storage bucket.


```
glossary = create_glossary(
    input_uri="gs://github-repo/translation/GoogleCloudGlossary.tsv",
    glossary_id="google_cloud_english_to_spanish",
)
print(glossary)
```

Now, let's try translating the text again using the glossary.


```
response = translate_text("Compute Engine", glossary=glossary.name)

# Display the translation for each input text provided
for translation in response.translations:
    print(f"Default Translated text: {translation.translated_text}")

for translation in response.glossary_translations:
    print(f"Glossary Translated text: {translation.translated_text}")
```

    Default Translated text: Motor de Computación
    Glossary Translated text: Compute Engine
    
