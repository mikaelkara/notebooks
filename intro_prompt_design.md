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

# Prompt Design - Best Practices

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/prompts/intro_prompt_design.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fprompts%2Fintro_prompt_design.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/prompts/intro_prompt_design.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/prompts/intro_prompt_design.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | | |
|-|-|-|
|Author(s) | [Polong Lin](https://github.com/polong-lin) | [Karl Weinmeister](https://github.com/kweinmeister)|

## Overview

This notebook covers the essentials of prompt engineering, including some best practices.

Learn more about prompt design in the [official documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/text/text-overview).

In this notebook, you learn best practices around prompt engineering -- how to design prompts to improve the quality of your responses.

This notebook covers the following best practices for prompt engineering:

- Be concise
- Be specific and well-defined
- Ask one task at a time
- Turn generative tasks into classification tasks
- Improve response quality by including examples

## Getting Started

### Install Vertex AI SDK and other required packages



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

Authenticate your environment on Google Colab.



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
PROJECT_ID = "your-project-id"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}


import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```


```
from vertexai.generative_models import GenerationConfig, GenerativeModel
```

### Load model


```
model = GenerativeModel("gemini-1.5-flash")
```

## Prompt engineering best practices

Prompt engineering is all about how to design your prompts so that the response is what you were indeed hoping to see.

The idea of using "unfancy" prompts is to minimize the noise in your prompt to reduce the possibility of the LLM misinterpreting the intent of the prompt. Below are a few guidelines on how to engineer "unfancy" prompts.

In this section, you'll cover the following best practices when engineering prompts:

* Be concise
* Be specific, and well-defined
* Ask one task at a time
* Improve response quality by including examples
* Turn generative tasks to classification tasks to improve safety

### Be concise

🛑 Not recommended. The prompt below is unnecessarily verbose.


```
prompt = "What do you think could be a good name for a flower shop that specializes in selling bouquets of dried flowers more than fresh flowers?"

print(model.generate_content(prompt).text)
```

✅ Recommended. The prompt below is to the point and concise.


```
prompt = "Suggest a name for a flower shop that sells bouquets of dried flowers"

print(model.generate_content(prompt).text)
```

### Be specific, and well-defined

Suppose that you want to brainstorm creative ways to describe Earth.

🛑 The prompt below might be a bit too generic (which is certainly OK if you'd like to ask a generic question!)


```
prompt = "Tell me about Earth"

print(model.generate_content(prompt).text)
```

✅ Recommended. The prompt below is specific and well-defined.


```
prompt = "Generate a list of ways that makes Earth unique compared to other planets"

print(model.generate_content(prompt).text)
```

### Ask one task at a time

🛑 Not recommended. The prompt below has two parts to the question that could be asked separately.


```
prompt = "What's the best method of boiling water and why is the sky blue?"

print(model.generate_content(prompt).text)
```

✅ Recommended. The prompts below asks one task a time.


```
prompt = "What's the best method of boiling water?"

print(model.generate_content(prompt).text)
```


```
prompt = "Why is the sky blue?"

print(model.generate_content(prompt).text)
```

### Watch out for hallucinations

Although LLMs have been trained on a large amount of data, they can generate text containing statements not grounded in truth or reality; these responses from the LLM are often referred to as "hallucinations" due to their limited memorization capabilities. Note that simply prompting the LLM to provide a citation isn't a fix to this problem, as there are instances of LLMs providing false or inaccurate citations. Dealing with hallucinations is a fundamental challenge of LLMs and an ongoing research area, so it is important to be cognizant that LLMs may seem to give you confident, correct-sounding statements that are in fact incorrect.

Note that if you intend to use LLMs for the creative use cases, hallucinating could actually be quite useful.

Try the prompt like the one below repeatedly. We set the temperature to 1.0 so that it takes more risks in its choices. It's possible that it may provide an inaccurate, but confident answer.


```
generation_config = GenerationConfig(temperature=1.0)

prompt = "What day is it today?"

print(model.generate_content(prompt, generation_config=generation_config).text)
```

Since LLMs do not have access to real-time information without further integrations, you may have noticed it hallucinates what day it is today in some of the outputs.

### Using system instructions to guardrail the model from irrelevant responses

How can we attempt to reduce the chances of irrelevant responses and hallucinations?

One way is to provide the LLM with [system instructions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini#system-instructions).

Let's see how system instructions works and how you can use them to reduce hallucinations or irrelevant questions for a travel chatbot.

Suppose we ask a simple question about one of Italy's most famous tourist spots.


```
model_travel = GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=[
        "Hello! You are an AI chatbot for a travel web site.",
        "Your mission is to provide helpful queries for travelers.",
        "Remember that before you answer a question, you must check to see if it complies with your mission.",
        "If not, you can say, Sorry I can't answer that question.",
    ],
)

chat = model_travel.start_chat()

prompt = "What is the best place for sightseeing in Milan, Italy?"
print(chat.send_message(prompt).text)
```

Now let us pretend to be a user asks the chatbot a question that is unrelated to travel.


```
prompt = "What's for dinner?"
print(chat.send_message(prompt).text)
```

You can see that this way, a guardrail in the prompt prevented the chatbot from veering off course.

### Turn generative tasks into classification tasks to reduce output variability

#### Generative tasks lead to higher output variability

The prompt below results in an open-ended response, useful for brainstorming, but response is highly variable.


```
prompt = "I'm a high school student. Recommend me a programming activity to improve my skills."

print(model.generate_content(prompt).text)
```

#### Classification tasks reduces output variability

The prompt below results in a choice and may be useful if you want the output to be easier to control.


```
prompt = """I'm a high school student. Which of these activities do you suggest and why:
a) learn Python
b) learn JavaScript
c) learn Fortran
"""

print(model.generate_content(prompt).text)
```

### Improve response quality by including examples

Another way to improve response quality is to add examples in your prompt. The LLM learns in-context from the examples on how to respond. Typically, one to five examples (shots) are enough to improve the quality of responses. Including too many examples can cause the model to over-fit the data and reduce the quality of responses.

Similar to classical model training, the quality and distribution of the examples is very important. Pick examples that are representative of the scenarios that you need the model to learn, and keep the distribution of the examples (e.g. number of examples per class in the case of classification) aligned with your actual distribution.

#### Zero-shot prompt

Below is an example of zero-shot prompting, where you don't provide any examples to the LLM within the prompt itself.


```
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment:
"""

print(model.generate_content(prompt).text)
```

#### One-shot prompt

Below is an example of one-shot prompting, where you provide one example to the LLM within the prompt to give some guidance on what type of response you want.


```
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment:
"""

print(model.generate_content(prompt).text)
```

#### Few-shot prompt

Below is an example of few-shot prompting, where you provide a few examples to the LLM within the prompt to give some guidance on what type of response you want.


```
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment: negative

Tweet: Something surprised me about this video - it was actually original. It was not the same old recycled stuff that I always see. Watch it - you will not regret it.
Sentiment:
"""

print(model.generate_content(prompt).text)
```

#### Choosing between zero-shot, one-shot, few-shot prompting methods

Which prompt technique to use will solely depends on your goal. The zero-shot prompts are more open-ended and can give you creative answers, while one-shot and few-shot prompts teach the model how to behave so you can get more predictable answers that are consistent with the examples provided.
