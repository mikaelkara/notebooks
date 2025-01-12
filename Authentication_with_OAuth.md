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

# Gemini API: OAuth Quickstart

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication_with_OAuth.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

Some parts of the Gemini API like model tuning and semantic retrieval use OAuth for authentication.

If you are a beginner, you should start by using [API keys](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb), and come back to this OAuth guide only when you need it for these features.

To help you get started with OAuth, this notebook shows a simplified approach that is appropriate
for a testing environment.

For a production environment, learn
about [authentication and authorization](https://developers.google.com/workspace/guides/auth-overview) before [choosing the access credentials](https://developers.google.com/workspace/guides/create-credentials#choose_the_access_credential_that_is_right_for_you) that are appropriate for your app.

## Prerequisites

To run this quickstart, you need:

*   The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk) installed on your local machine.
*   [A Google Cloud project](https://developers.google.com/workspace/guides/create-project).

If you created an API key in Google AI Studio, a Google Cloud project was made for you. Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and note the Google Cloud project name to use that project.

## Set up your Cloud project

To complete this quickstart, you first need to setup your Cloud project.

### 1. Enable the API

Before using Google APIs, you need to turn them on in a Google Cloud project.

*   In the Google Cloud console, [enable](https://console.cloud.google.com/flows/enableapi?apiid=generativelanguage.googleapis.com) the Google Generative Language API. If you created an API Key in AI Studio, this was done for you.<br>

### 2. Configure the OAuth consent screen

Next configure the project's OAuth consent screen and add yourself as a test user. If you've already completed this step for your Cloud project, skip to the next section.

1. In the Google Cloud console, go to the [OAuth consent screen](https://console.cloud.google.com/apis/credentials/consent), this can be found under **Menu** > **APIs & Services** > **OAuth
  consent screen**.

2. Select the user type **External** for your app, then click **Create**.

3. Complete the app registration form (you can leave most fields blank), then click **Save and Continue**.

4. For now, you can skip adding scopes and click **Save and Continue**. In the
   future, when you create an app for use outside of your Google Workspace
   organization, you must add and verify the authorization scopes that your
   app requires.

5. Add test users:
    1. Under **Test users**, click **Add users**.
    2. Enter your email address and any other authorized test users, then
       click **Save and Continue**.

6. Review your app registration summary. To make changes, click **Edit**. If
  the app registration looks OK, click **Back to Dashboard**.

### 3. Authorize credentials for a desktop application

To authenticate as an end user and access user data in your app, you need to
create one or more OAuth 2.0 Client IDs. A client ID is used to identify a
single app to Google's OAuth servers. If your app runs on multiple platforms,
you must create a separate client ID for each platform.

1. In the Google Cloud console, go to [Credentials](https://console.cloud.google.com/apis/credentials/consent), this can be found under **Menu** > **APIs & Services** >
   **Credentials**.

2. Click **Create Credentials** > **OAuth client ID**.
3. Click **Application type** > **Desktop app**.
4. In the **Name** field, type a name for the credential. This name is only
  shown in the Google Cloud console.
5. Click **Create**. The OAuth client created screen appears, showing your new
  Client ID and Client secret.
6. Click **OK**. The newly created credential appears under **OAuth 2.0 Client
  IDs.**
7. Click the download button to save the JSON file. It will be saved as
  `client_secret_<identifier>.json`.


## Set up application default credentials

In this quickstart you will use [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) to authenticate.

### Add client secret to Colab secrets

If you need to use OAuth with the Gemini API in Google Colab frequently, it is easiest to add the contents of your `client_secret.json` file into Colab's Secrets manager.

1. Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the left panel.
2. Create a new secret with the name `CLIENT_SECRET`.
3. Open your `client_secret.json` file in a text editor and copy/paste the content into the `Value` input box of `CLIENT_SECRET`.
4. Toggle the button on the left to allow notebook access to the secret.

Now you can programmatically create the file instead of uploading it every time. The client secret is also available in all your Google Colab notebooks after you allow access.


```
from google.colab import userdata
import pathlib
pathlib.Path('client_secret.json').write_text(userdata.get('CLIENT_SECRET'))
```




    413



### Set the application default credentials

To convert the `client_secret.json` file into usable credentials, pass its location the `gcloud auth application-default login` command's `--client-id-file` argument.

The simplified project setup in this tutorial triggers a **Google hasn't verified this app** dialog. This is normal, choose **Continue**.

You will need to do this step once for every new Google Colab notebook or runtime.

**Note**: Carefully follow the instructions the following command prints (don't just click the link). Also make sure your local `gcloud --version` is the [latest](https://cloud.google.com/sdk/docs/release-notes) to match the version pre-installed in Google Colab.



```
!gcloud auth application-default login \
  --no-browser --client-id-file client_secret.json \
  --scopes https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning,https://www.googleapis.com/auth/generative-language.retriever

```

The specific `scopes` you need depends on the API you are using. For example, looking at the API reference for [`tunedModels.create`](https://ai.google.dev/api/rest/v1beta/tunedModels/create#authorization-scopes), you will see:

> Requires one of the following OAuth scopes:
>
> *   `https://www.googleapis.com/auth/generative-language.tuning`

This sample asks for all the scopes for tuning and semantic retrieval, but best practice is to use the smallest set of scopes for security and user confidence.

## Using the Python SDK with OAuth

The Python SDK will automatically find and use application default credentials.


```
!pip install -U -q "google-generativeai>=0.7.2"
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m137.4/137.4 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25h

Let's do a quick test. Note that you did not set an API key using `genai.configure()`!


```
import google.generativeai as genai

print('Available base models:', [m.name for m in genai.list_models()])
```

# Appendix

## Making authenticated REST calls from Colab

In general, you should use the Python SDK to interact with the Gemini API when possible. This example shows how to make OAuth authenticated REST calls from Python for debugging or testing purposes. It assumes you have already set application default credentials from the Quickstart.


```
import requests

access_token = !gcloud auth application-default print-access-token

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {access_token[0]}',
}

response = requests.get('https://generativelanguage.googleapis.com/v1/models', headers=headers)
response_json = response.json()

# All the model names
for model in response_json['models']:
    print(model['name'])
```

### Share a tuned model

Some beta API features may not be supported by the Python SDK yet. This example shows how to make a REST call to add a permission to a tuned model from Python.


```
import requests

model_name = ''   # @param {type:"string"}
emailAddress = '' # @param {type:"string"}


access_token = !gcloud auth application-default print-access-token

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {access_token[0]}',
}

body = {
  'granteeType': 'USER',        # Or 'GROUP' or 'EVERYONE' https://ai.google.dev/api/rest/v1beta/tunedModels.permissions
  'emailAddress': emailAddress, # Optional if 'granteeType': 'EVERYONE'
  'role': 'READER'
}

response = requests.post(f'https://generativelanguage.googleapis.com/v1beta/tunedModels/{model_name}/permissions', json=body, headers=headers)
print(response.json())

```

## Use a service account to authenticate

Google Cloud [service accounts](https://cloud.google.com/iam/docs/service-account-overview) are accounts that do not represent a human user. They provide a way to manage authentication and authorization when a human is not directly involved, such as your application calling the Gemini API to fulfill a user request, but not authenticated as the user. A simple way to use service accounts to authenticate with the Gemini API is to use a [service account key](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key).

This guide briefly covers how to use service account keys in Google Colab.

**Important:** Service account keys can be a security risk! For more information, see [best practices for managing service account keys](https://cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys).

### 1. Create a service account

Follow the instructions to [create a service account](https://cloud.google.com/iam/docs/service-accounts-create#creating). The **Console** instructions are easiest if you are doing this manually.

### 2. Create a service account key

Follow the instructions to [create a service account key]( https://cloud.google.com/iam/docs/keys-create-delete#creating). Note the name of the downloaded key.

### 3. Add the service account key to Colab

1. Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the left panel.
2. Create a new secret with the name `SERVICE_ACCOUNT_KEY`.
3. Open your service account key file in a text editor and copy/paste the content into the `Value` input box of `SERVICE_ACCOUNT_KEY`.
4. Toggle the button on the left to allow notebook access to the secret.

### 4. Authenticate with the Python SDK by service account key


```
import google.generativeai as genai
import pathlib
from google.colab import userdata
from google.oauth2 import service_account

pathlib.Path('service_account_key.json').write_text(userdata.get('SERVICE_ACCOUNT_KEY'))

credentials = service_account.Credentials.from_service_account_file('service_account_key.json')

# Adjust scopes as needed
scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/generative-language.retriever'])

genai.configure(credentials=scoped_credentials)

print('Available base models:', [m.name for m in genai.list_models()])
```
