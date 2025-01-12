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

# Forced Function Calling with Tool Configurations in Gemini

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/forced_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fforced_function_calling.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/function-calling/forced_function_calling.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/forced_function_calling.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Kristopher Overholt](https://github.com/koverholt) |

## Overview

This notebook demonstrates the use of forced Function Calling in the Gemini model.

### Gemini

Gemini is a family of generative AI models developed by Google DeepMind that is designed for multimodal use cases.

### Function Calling in Gemini

[Function Calling in Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with.

### Forced Function Calling

[Forced Function Calling in Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#tool-config) allows you to place constraints on how the model should use the function declarations that you provide it with. Using tool configurations, you can force the Gemini model to only predict function calls. You can also choose to provide the model with a full set of function declarations, but restrict its responses to a subset of these functions.

## Objectives

In this tutorial, you will learn how to use the Vertex AI SDK for Python to use different function calling modes, including forced function calling, via the Gemini model.

You will complete the following tasks:

- Read through an overview of forced function calling and when to use it
- Use the default function calling behavior in `AUTO` mode
- Enable forced function calling using the `ANY` mode
- Disable function calling using the `NONE` mode

## Getting Started

### Install Vertex AI SDK and other required packages


```
%pip install --upgrade --user --quiet google-cloud-aiplatform arxiv
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
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.


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

## Import libraries


```
from IPython.display import Markdown, display
import arxiv
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    Tool,
)
from vertexai.preview.generative_models import ToolConfig
```

## Initialize model

Initialize the Gemini model. Refer to the [Gemini Function Calling documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) for more information on which models and model versions support forced function calling and tool configurations.


```
model = GenerativeModel(
    "gemini-1.5-pro",
    generation_config=GenerationConfig(temperature=0),
)
```

## Define a function to search for scientific papers in arXiv

Since this notebook focuses on using different tool configurations and modes in Gemini Function Calling, you'll define a function declaration to use throughout the examples. The purpose of this function is to extract a parameter to send as a query to search for relevant papers in [arXiv](https://arxiv.org/). arXiv is an open-access repository of electronic preprints and postprints that consists of scientific papers in various fields.


```
search_arxiv = FunctionDeclaration(
    name="search_arxiv",
    description="Search for articles and publications in arXiv",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query to search for in arXiv"}
        },
    },
)
```

Define a tool that wraps the above function:


```
search_tool = Tool(
    function_declarations=[
        search_arxiv,
    ],
)
```

You'll use this function declaration and tool throughout the next few sections of the notebook.

## Overview of Forced Function Calling in Gemini

The default behavior for Function Calling allows the Gemini model to decide whether to predict a function call or a natural language response. This is because the default Function Calling mode in Gemini is set to `AUTO`.

In most cases this is the desired behavior when you want the Gemini model to use information from the prompt to determine if it should call a function, and which function it should call. However, you might have specific use cases where you want to **force** the Gemini model to call a function (or a set of functions) in a given model generation request.

Tool configurations in the Gemini API allow you to specify different Function Calling modes in Gemini. Refer to the [Gemini Function Calling documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) for more information on forced function calling and tool configurations.

The following code example for `tool_config` shows various modes that you can set and pass to the Gemini model either globally when you initialize the model or for a given model generation request:


```
# tool_config = ToolConfig(
#     function_calling_config =
#         ToolConfig.FunctionCallingConfig(
#             mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,  # The default model behavior. The model decides whether to predict a function call or a natural language response.
#             mode=ToolConfig.FunctionCallingConfig.Mode.ANY,  # ANY mode forces the model to predict a function call from a subset of function names.
#             mode=ToolConfig.FunctionCallingConfig.Mode.NONE,  # NONE mode instructs the model to not predict function calls. Equivalent to a model request without any function declarations.
#             allowed_function_names = ["function_to_call"]  # Allowed functions to call when mode is ANY, if empty any one of the provided functions will be called.
#         )
# )
```

Using these Function Calling modes, you can configure the model to behave in one of the following ways:

- Allow the model to choose whether to predict a function call or natural language response (`AUTO` mode)
- Force the model to predict a function call on one function or a set of functions (`ANY` mode)
- Disable function calling and return a natural language response as if no functions or tools were defined (`NONE` mode)

In the following sections, you'll walk through examples and sample code for each Function Calling mode.

## Example: Default Function Calling mode (`AUTO`)

In this example, you'll specify the function calling mode as `AUTO`. Note that `AUTO` mode is the default model behavior, therefore the Gemini model will also use this mode when there is no `tool_config` specified:


```
tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,  # The default model behavior. The model decides whether to predict a function call or a natural language response.
    )
)
```

Ask a question about a topic related to publications in arXiv and include the `tool_config` kwarg. Note that you can also set the `tool_config` kwarg globally in the model rather than with every request to generate content:


```
prompt = "Explain the Schrodinger equation in a few sentences and give me papers from arXiv to learn more"
response = model.generate_content(prompt, tools=[search_tool], tool_config=tool_config)

display(Markdown(response.candidates[0].content.parts[0].text))
```


The Schrödinger equation is a fundamental equation in quantum mechanics that describes how the quantum state of a physical system changes over time. It is a linear partial differential equation that governs the wave function of a quantum-mechanical system. The equation is named after Erwin Schrödinger, who derived it in 1925 and published it in 1926.




The response includes a natural language summary to the prompt. However, you were probably hoping to make a function call along the way to search for actual papers in arXiv and return them to the end user!

We'll make that happen in the next section by using the forced function calling mode.

## Example: Using Forced Function Calling mode (`ANY`)

In this example, you'll set the tool configuration to `ANY`, and (optionally) specify one or more `allowed_function_names` that will force Gemini to make a function call against a function or subset of functions:


```
tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,  # ANY mode forces the model to predict a function call from a subset of function names.
        allowed_function_names=[
            "search_arxiv"
        ],  # Allowed functions to call when mode is ANY, if empty any one of the provided functions will be called.
    )
)
```

Now you can ask the same question publications in arXiv with our newly defined `tool_config` that is set to `ANY` function calling mode, which will force the Gemini model to call our search function.


```
prompt = "Explain the Schrodinger equation in a few sentences and give me papers from arXiv to learn more"
response = model.generate_content(prompt, tools=[search_tool], tool_config=tool_config)

response_function_call_content = response.candidates[0].content
response.candidates[0].content.parts[0].function_call
```




    name: "search_arxiv"
    args {
      fields {
        key: "query"
        value {
          string_value: "Schrödinger equation"
        }
      }
    }



You can extract the parameters from the model response so that we can use them to make an API call to search papers in arXiv:


```
params = {}
for key, value in response.candidates[0].content.parts[0].function_call.args.items():
    params[key] = value
params
```




    {'query': 'Schrödinger equation'}




```
if response.candidates[0].content.parts[0].function_call.name == "search_arxiv":
    client = arxiv.Client()

    search = arxiv.Search(
        query=params["query"], max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)
    results = str([r for r in results])
```

Print a sample of the API response from arXiv:


```
results[:1000]
```




    '[arxiv.Result(entry_id=\'http://arxiv.org/abs/2404.15250v1\', updated=datetime.datetime(2024, 4, 23, 17, 36, 59, tzinfo=datetime.timezone.utc), published=datetime.datetime(2024, 4, 23, 17, 36, 59, tzinfo=datetime.timezone.utc), title=\'Unifying the Temperature Dependent Dynamics of Glasses\', authors=[arxiv.Result.Author(\'Joseph B. Schlenoff\'), arxiv.Result.Author(\'Khalil Akkaoui\')], summary=\'Strong changes in bulk properties, such as modulus and viscosity, are\\nobserved near the glass transition temperature, T_{g}, of amorphous materials.\\nFor more than a century, intense efforts have been made to define a microscopic\\norigin for these macroscopic changes in properties. Using transition state\\ntheory, we delve into the atomic/molecular level picture of how microscopic\\nlocalized relaxations, or "cage rattles," translate to macroscopic structural\\nrelaxations above T_{g}. Unit motion is broken down into two populations: (1)\\nsimultaneous rearrangement occurs among a critical number of unit'




```
response = model.generate_content(
    [
        Content(
            role="user",
            parts=[
                Part.from_text(prompt),
            ],
        ),
        response_function_call_content,  # Function call response
        Content(
            parts=[
                Part.from_function_response(
                    name="search_arxiv",
                    response={
                        "content": results,  # Return the API response to the Gemini model
                    },
                )
            ],
        ),
    ],
    tools=[search_tool],
)

display(Markdown(response.text))
```


The Schrödinger equation is a fundamental equation in quantum mechanics that describes how the quantum state of a physical system changes over time. It is a partial differential equation that involves the wavefunction of the system, which contains all the information about the system's state. The equation is named after Erwin Schrödinger, who first proposed it in 1926.

Here are a few papers from arXiv that you can read to learn more about the Schrödinger equation:

*   **"Unifying the Temperature Dependent Dynamics of Glasses"** (arXiv:2404.15250v1) This paper discusses the use of transition state theory to understand the atomic/molecular level picture of how microscopic localized relaxations translate to macroscopic structural relaxations above the glass transition temperature.
*   **"A GPU-accelerated Cartesian grid method for PDEs on irregular domain"** (arXiv:2404.15249v1) This paper presents a GPU-accelerated Cartesian grid method for solving partial differential equations (PDEs) on irregular domains. 
*   **"A Hybrid Kernel-Free Boundary Integral Method with Operator Learning for Solving Parametric Partial Differential Equations In Complex Domains"** (arXiv:2404.15242v1) This paper proposes a hybrid kernel-free boundary integral method that integrates the foundational principles of the KFBI method with the capabilities of deep learning. 



In this case, the natural language response contains information about relevant papers based on our function call to the arXiv API.

## Example: Disabling Function Calling (`NONE`)

In this example, you'll set the tool configuration to `NONE`, which will instruct the Gemini model to behave as if no tools or functions were defined.


```
tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.NONE,  # NONE mode instructs the model to not predict function calls. Equivalent to a model request without any function declarations.
    )
)
```


```
prompt = "Explain the Schrodinger equation in a few sentences and give me papers from arXiv to learn more"
response = model.generate_content(
    prompt,
    tool_config=tool_config,
)

display(Markdown(response.candidates[0].content.parts[0].text))
```


## The Schrödinger Equation Explained

The Schrödinger equation is a fundamental equation in quantum mechanics that describes how the quantum state of a physical system changes over time. It is a linear partial differential equation that relates the wavefunction of a system to its energy and potential. The wavefunction, typically denoted by the Greek letter psi (ψ), contains all the information about a system, and its evolution in time governed by the Schrödinger equation determines the system's behavior. 

Essentially, the Schrödinger equation acts as the quantum counterpart to Newton's second law in classical mechanics, dictating how quantum systems evolve.

### Delving Deeper: arXiv Papers

arXiv is a fantastic resource for exploring scientific papers, including those related to the Schrödinger equation. Here are a few papers you might find helpful:

*   **"Derivation of the Schrödinger equation from classical stochastic dynamics" (arXiv:1011.0674)**: This paper explores the connection between classical stochastic dynamics and the Schrödinger equation, offering a unique perspective on its foundations.
*   **"The Schrödinger equation as a diffusion equation" (arXiv:quant-ph/0608221)**: This paper presents an interpretation of the Schrödinger equation as a diffusion equation, providing insights into the probabilistic nature of quantum mechanics.
*   **"Numerical solution of the time-dependent Schrödinger equation for a multielectron atom" (arXiv:physics/0607082)**: This paper delves into the computational aspects of solving the Schrödinger equation for complex systems like multi-electron atoms.

**Exploring arXiv with Keywords:**

To discover more relevant papers, you can use keywords like "Schrödinger equation," "quantum mechanics," "wavefunction," and "quantum dynamics" on the arXiv website. You can also filter your search by specific categories like "quant-ph" (quantum physics) or "math-ph" (mathematical physics).



Note that the natural language response only contains content that was generated by the Gemini model and within the scope of its training data rather than real-time information from the arXiv API.
