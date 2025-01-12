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

# Analyze a codebase with the Vertex AI Gemini 1.5 Pro


<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/code/analyze_codebase_with_gemini_1_5_pro.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fuse-cases%2Fcode%2Fanalyze_codebase_with_gemini_1_5_pro.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/use-cases/code/analyze_codebase_with_gemini_1_5_pro.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/code/analyze_codebase_with_gemini_1_5_pro.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong), [Aakash Gouda](https://github.com/aksstar)|

## Overview

Gemini 1.5 Pro introduces a breakthrough long context window of up to 1 million tokens that can help seamlessly analyze, classify and summarize large amounts of content within a given prompt. With its long-context reasoning, Gemini 1.5 Pro can analyze an entire codebase for deeper insights.

In this tutorial, you learn how to analyze an entire codebase with Gemini 1.5 Pro and prompt the model to:

- **Analyze**: Summarize codebases effortlessly.
- **Guide**: Generate clear developer getting-started documentation.
- **Debug**: Uncover critical bugs and provide fixes.
- **Enhance**: Implement new features and improve reliability and security.


## Getting Started

### Install Vertex AI SDK for Python



```
%pip install --upgrade --user --quiet google-cloud-aiplatform \
                                        gitpython \
                                        magika
```

### Restart runtime (Colab only)

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


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

If you are running this notebook on Google Colab, run the following cell to authenticate your environment.



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

### Import libraries


```
from IPython.core.interactiveshell import InteractiveShell
import IPython.display

InteractiveShell.ast_node_interactivity = "all"

from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)
```

## Cloning a codebase

You will use repo [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) as an example in this notebook. Online Boutique is a cloud-first microservices demo application. The application is a web-based e-commerce app where users can browse items, add them to the cart, and purchase them. This application consists of 11 microservices across multiple languages.


```
# The GitHub repository URL
repo_url = "https://github.com/GoogleCloudPlatform/microservices-demo"  # @param {type:"string"}

# The location to clone the repo
repo_dir = "./repo"
```

#### Define helper functions for processing GitHub repository


```
import os
from pathlib import Path
import shutil

import git
import magika
import requests

m = magika.Magika()


def clone_repo(repo_url, repo_dir):
    """Clone a GitHub repository."""

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    os.makedirs(repo_dir)
    git.Repo.clone_from(repo_url, repo_dir)


def extract_code(repo_dir):
    """Create an index, extract content of code/text files."""

    code_index = []
    code_text = ""
    for root, _, files in os.walk(repo_dir):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_dir)
            code_index.append(relative_path)

            file_type = m.identify_path(Path(file_path))
            if file_type.output.group in ("text", "code"):
                try:
                    with open(file_path) as f:
                        code_text += f"----- File: {relative_path} -----\n"
                        code_text += f.read()
                        code_text += "\n-------------------------\n"
                except Exception:
                    pass

    return code_index, code_text


def get_github_issue(owner: str, repo: str, issue_number: str) -> str:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }  # Set headers for GitHub API

    # Construct API URL
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

    try:
        response_git = requests.get(url, headers=headers)
        response_git.raise_for_status()  # Check for HTTP errors
    except requests.exceptions.RequestException as error:
        print(f"Error fetching issue: {error}")  # Handle potential errors

    issue_data = response_git.json()
    if issue_data:
        return issue_data["body"]
    return ""
```

#### Create an index and extract content of a codebase

Clone the repo and create an index and extract content of code/text files.


```
clone_repo(repo_url, repo_dir)

code_index, code_text = extract_code(repo_dir)
```

## Analyzing the codebase with Gemini 1.5 Pro

With its long-context reasoning, Gemini 1.5 Pro can process the codebase and answer questions about the codebase.

#### Load the Gemini 1.5 Pro model

Learn more about the [Gemini API models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models).



```
MODEL_ID = "gemini-1.5-pro"  # @param {type:"string"}

model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a coding expert.",
        "Your mission is to answer all code related questions with given context and instructions.",
    ],
)
```

#### Define a helper function to generate a prompt to a code related question


```
def get_code_prompt(question):
    """Generates a prompt to a code related question."""

    prompt = f"""
    Questions: {question}

    Context:
    - The entire codebase is provided below.
    - Here is an index of all of the files in the codebase:
      \n\n{code_index}\n\n.
    - Then each of the files is concatenated together. You will find all of the code you need:
      \n\n{code_text}\n\n

    Answer:
  """

    return prompt
```

### 1. Summarizing the codebase


Generate a summary of the codebase.


```
question = """
  Give me a summary of this codebase, and tell me the top 3 things that I can learn from it.
"""

prompt = get_code_prompt(question)
contents = [prompt]

# Generate text using non-streaming method
response = model.generate_content(contents)

# Print generated text and usage metadata
print(f"\nAnswer:\n{response.text}")
print(f'\nUsage metadata:\n{response.to_dict().get("usage_metadata")}')
print(f"\nFinish reason:\n{response.candidates[0].finish_reason}")
print(f"\nSafety settings:\n{response.candidates[0].safety_ratings}")
```

### 2. Creating a developer getting started guide

Generate a getting started guide for developers. This sample uses the streaming option to generate the content.


```
question = """
  Provide a getting started guide to onboard new developers to the codebase.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 3. Finding bugs

Find the top 3 most severe issues in the codebase.


```
question = """
  Find the top 3 most severe issues in the codebase.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 4. Fixing bug

Find the most severe issue in the codebase that can be fixed and provide a code fix for it.



```
question = """
  Find the most severe bug in the codebase that you can provide a code fix for.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 5. Implementing a feature request using Function Calling

Generate code to implement a feature request.

Get feature request text from GitHub Issue


```
# Function declaration with detailed docstring
extract_details_from_url_func = FunctionDeclaration(
    name="extract_details_from_url",
    description="Extracts owner, repository name, and issue number details from a GitHub issue URL",
    parameters={
        "type": "object",
        "properties": {
            "owner": {
                "type": "string",
                "description": "The owner of the GitHub repository.",
            },
            "repo": {
                "type": "string",
                "description": "The name of the GitHub repository.",
            },
            "issue_number": {
                "type": "string",
                "description": "The issue number to fetch the body of.",
            },
        },
    },
)

# Tool definition
extraction_tool = Tool(function_declarations=[extract_details_from_url_func])

FEATURE_REQUEST_URL = (
    "https://github.com/GoogleCloudPlatform/microservices-demo/issues/2205"
)

# Prompt content
prompt_content = f"What is the feature request of the following {FEATURE_REQUEST_URL}"

# Model generation with tool usage
response = model.generate_content(
    [prompt_content],
    generation_config=GenerationConfig(temperature=0),
    tools=[extraction_tool],
)
# Extract parameters from model response
function_call = response.candidates[0].function_calls[0]

# Fetch issue details from GitHub API if function call matches
if function_call.name == "extract_details_from_url":
    issue_body = get_github_issue(
        function_call.args["owner"],
        function_call.args["repo"],
        function_call.args["issue_number"],
    )

IPython.display.Markdown(f"Feature Request:\n{issue_body}")
```

Use the GitHub Issue text to implement the feature request


```
# Combine feature request with URL and get code prompt
question = (
    "Implement the following feature request" + FEATURE_REQUEST_URL + "\n" + issue_body
)

prompt = get_code_prompt(question)

# Generate code response
response = model.generate_content([prompt])
IPython.display.Markdown(response.text)  # Display in Markdown format
```

### 6. Creating a troubleshooting guide

Create a troubleshooting guide to help resolve common issues.


```
question = """
    Provide a troubleshooting guide to help resolve common issues.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 7. Making the app more reliable

Recommend best practices to make the application more reliable.



```
question = """
  How can I make this application more reliable? Consider best practices from https://www.r9y.dev/
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 8. Making the app more secure

Recommend best practices to make the application more secure.


```
question = """
  How can you secure the application?
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 9. Learning the codebase

Create a quiz about the concepts used in the codebase.


```
question = """
  Create a quiz about the concepts used in my codebase to help me solidify my understanding.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 10. Creating a quickstart tutorial

Create an end-to-end quickstart tutorial for a specific component.



```
question = """
  Please write an end-to-end quickstart tutorial that introduces AlloyDB,
  shows how to configure it with the CartService,
  and highlights key capabilities of AlloyDB in context of the Online Boutique application.
"""

prompt = get_code_prompt(question)
contents = [prompt]

responses = model.generate_content(contents, stream=True)
for response in responses:
    IPython.display.Markdown(response.text)
```

### 11. Creating a Git Changelog Generator

Understanding changes made between Git commits and highlighting the most important aspects of the changes.


```
### Fetches commit IDs from a local Git repository on a specified branch.

repo = git.Repo(repo_dir)
branch_name = "main"
commit_ids = [
    commit.hexsha for commit in repo.iter_commits(branch_name)
]  # A list of commit IDs (SHA-1 hashes) in reverse chronological order (newest first)

if len(commit_ids) >= 2:
    diff_text = repo.git.diff(commit_ids[0], commit_ids[1])

    question = """
      Given the above git diff output, Summarize the important changes made.
    """

    prompt = diff_text + question + code_text
    contents = [prompt]

    responses = model.generate_content(contents, stream=True)
    for response in responses:
        IPython.display.Markdown(response.text)
```

## Conclusion

In this tutorial, you've learned how to use the Gemini 1.5 Pro to analyze a codebase and prompt the model to:

- Summarize codebases effortlessly.
- Generate clear developer getting-started documentation.
- Uncover critical bugs and provide fixes.
- Implement new features and improve reliability and security.
- Understanding changes made between Git commits
