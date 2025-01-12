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

# Getting Started with Chat with the Gemini Pro model

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_chat.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Fgetting-started%2Fintro_gemini_chat.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Run in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_chat.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/getting-started/intro_gemini_chat.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Eric Dong](https://github.com/gericdong) |

## Overview

This notebook demonstrates how to send chat prompts to the Gemini 1.5 Pro model (`gemini-1.5-pro`) by using the Vertex AI SDK for Python and LangChain. Gemini 1.5 Pro supports prompts with text-only input, including natural language tasks, multi-turn text and code chat, and code generation. It can output text and code.

Learn more about [Sending chat prompt requests (Gemini)](https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/send-chat-prompts-gemini).

### Objectives

In this tutorial, you learn how to send chat prompts to the Gemini 1.5 Pro model (`gemini-1.5-pro`) using the Vertex AI SDK for Python and LangChain.

You will complete the following tasks:

- Sending chat prompts using Vertex AI SDK for Python
- Sending chat prompts using LangChain

### Costs
This tutorial uses billable components of Google Cloud:

- Vertex AI

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate a cost estimate based on your projected usage.

## Getting Started

### Install libraries



```
%pip install --upgrade --quiet google-cloud-aiplatform \
                                 langchain-google-vertexai \
                                 langchain
```

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After its restarted, continue to the next step.


```
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ Wait for the kernel to finish restarting before you continue. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the cell below to authenticate your environment.

This step is not required if you are using [Vertex AI Workbench](https://cloud.google.com/vertex-ai-workbench).


```
import sys

# Additional authentication is required for Google Colab
if "google.colab" in sys.modules:
    # Authenticate user to Google Cloud
    from google.colab import auth

    auth.authenticate_user()
```

### Define Google Cloud project information and initialize Vertex AI

To get started using Vertex AI, you must have an existing Google Cloud project and [enable the Vertex AI API](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = "[your-project-id]"  # @param {type: "string", placeholder: "[your-project-id]" isTemplate: true}
if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries


```
from IPython.display import Markdown
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from vertexai.generative_models import Content, GenerativeModel, Part
```

## Sending chat prompts using Vertex AI SDK for Python

### Load the Gemini 1.5 Pro model

Gemini 1.5 Pro supports text and code generation from a text prompt.


```
model = GenerativeModel("gemini-1.5-pro")
```

### Start a chat session

You start a stateful chat session and then send chat prompts with configuration parameters including generation configurations and safety settings.


```
chat = model.start_chat()

response = chat.send_message(
    """You are an astronomer, knowledgeable about the solar system.
How many moons does Mars have? Tell me some fun facts about them.
"""
)

print(response.text)
```

    Mars has two moons: Phobos and Deimos.
    
    **Phobos**
    
    * **Closest moon to Mars:** It orbits just 6,000 kilometers (3,700 miles) above the surface.
    * **Unusual shape:** Phobos is not spherical but rather irregularly shaped, resembling a potato.
    * **Astronomical sync:** Phobos orbits Mars three times a day, completing its orbit faster than any other known moon in the solar system.
    * **Unique groove:** Phobos has a prominent groove or canyon called Stickney, which is thought to be the result of an impact.
    * **Tidal acceleration:** Phobos is gradually spiraling inward towards Mars due to tidal forces. It is estimated that in about 100 million years, Phobos will either crash into Mars or break up into a ring.
    
    **Deimos**
    
    * **Outer moon of Mars:** Deimos orbits about 23,500 kilometers (14,600 miles) from the planet's surface.
    * **Small and irregularly shaped:** Deimos is even smaller and more irregularly shaped than Phobos.
    * **Slow orbit:** Deimos takes about 30 hours to complete one orbit around Mars.
    * **Captured asteroid:** Deimos is believed to be a captured asteroid rather than a moon that formed alongside Mars.
    * **Speculation of being artificial:** In 1958, Soviet astronomer Iosif Shklovsky suggested that Deimos might be an artificial satellite placed in orbit around Mars by an advanced extraterrestrial civilization. However, this theory has not gained scientific acceptance.
    

You can use `Markdown` to display the generated text.


```
Markdown(response.text)
```




Mars has two moons: Phobos and Deimos.

**Phobos**

* **Closest moon to Mars:** It orbits just 6,000 kilometers (3,700 miles) above the surface.
* **Unusual shape:** Phobos is not spherical but rather irregularly shaped, resembling a potato.
* **Astronomical sync:** Phobos orbits Mars three times a day, completing its orbit faster than any other known moon in the solar system.
* **Unique groove:** Phobos has a prominent groove or canyon called Stickney, which is thought to be the result of an impact.
* **Tidal acceleration:** Phobos is gradually spiraling inward towards Mars due to tidal forces. It is estimated that in about 100 million years, Phobos will either crash into Mars or break up into a ring.

**Deimos**

* **Outer moon of Mars:** Deimos orbits about 23,500 kilometers (14,600 miles) from the planet's surface.
* **Small and irregularly shaped:** Deimos is even smaller and more irregularly shaped than Phobos.
* **Slow orbit:** Deimos takes about 30 hours to complete one orbit around Mars.
* **Captured asteroid:** Deimos is believed to be a captured asteroid rather than a moon that formed alongside Mars.
* **Speculation of being artificial:** In 1958, Soviet astronomer Iosif Shklovsky suggested that Deimos might be an artificial satellite placed in orbit around Mars by an advanced extraterrestrial civilization. However, this theory has not gained scientific acceptance.



You can check out the metadata of the response including the `safety_ratings` and `usage_metadata`.


```
print(response)
```

    candidates {
      content {
        role: "model"
        parts {
          text: "Mars has two moons: Phobos and Deimos.\n\n**Phobos**\n\n* **Closest moon to Mars:** It orbits just 6,000 kilometers (3,700 miles) above the surface.\n* **Unusual shape:** Phobos is not spherical but rather irregularly shaped, resembling a potato.\n* **Astronomical sync:** Phobos orbits Mars three times a day, completing its orbit faster than any other known moon in the solar system.\n* **Unique groove:** Phobos has a prominent groove or canyon called Stickney, which is thought to be the result of an impact.\n* **Tidal acceleration:** Phobos is gradually spiraling inward towards Mars due to tidal forces. It is estimated that in about 100 million years, Phobos will either crash into Mars or break up into a ring.\n\n**Deimos**\n\n* **Outer moon of Mars:** Deimos orbits about 23,500 kilometers (14,600 miles) from the planet\'s surface.\n* **Small and irregularly shaped:** Deimos is even smaller and more irregularly shaped than Phobos.\n* **Slow orbit:** Deimos takes about 30 hours to complete one orbit around Mars.\n* **Captured asteroid:** Deimos is believed to be a captured asteroid rather than a moon that formed alongside Mars.\n* **Speculation of being artificial:** In 1958, Soviet astronomer Iosif Shklovsky suggested that Deimos might be an artificial satellite placed in orbit around Mars by an advanced extraterrestrial civilization. However, this theory has not gained scientific acceptance."
        }
      }
      finish_reason: STOP
      safety_ratings {
        category: HARM_CATEGORY_HATE_SPEECH
        probability: NEGLIGIBLE
      }
      safety_ratings {
        category: HARM_CATEGORY_DANGEROUS_CONTENT
        probability: NEGLIGIBLE
      }
      safety_ratings {
        category: HARM_CATEGORY_HARASSMENT
        probability: NEGLIGIBLE
      }
      safety_ratings {
        category: HARM_CATEGORY_SEXUALLY_EXPLICIT
        probability: NEGLIGIBLE
      }
    }
    usage_metadata {
      prompt_token_count: 28
      candidates_token_count: 334
      total_token_count: 362
    }
    
    

You can retrieve the history of the chat session.


```
print(chat.history)
```

    [role: "user"
    parts {
      text: "You are an astronomer, knowledgeable about the solar system.\nHow many moons does Mars have? Tell me some fun facts about them.\n"
    }
    , role: "model"
    parts {
      text: "Mars has two moons: Phobos and Deimos.\n\n**Phobos**\n\n* **Closest moon to Mars:** It orbits just 6,000 kilometers (3,700 miles) above the surface.\n* **Unusual shape:** Phobos is not spherical but rather irregularly shaped, resembling a potato.\n* **Astronomical sync:** Phobos orbits Mars three times a day, completing its orbit faster than any other known moon in the solar system.\n* **Unique groove:** Phobos has a prominent groove or canyon called Stickney, which is thought to be the result of an impact.\n* **Tidal acceleration:** Phobos is gradually spiraling inward towards Mars due to tidal forces. It is estimated that in about 100 million years, Phobos will either crash into Mars or break up into a ring.\n\n**Deimos**\n\n* **Outer moon of Mars:** Deimos orbits about 23,500 kilometers (14,600 miles) from the planet\'s surface.\n* **Small and irregularly shaped:** Deimos is even smaller and more irregularly shaped than Phobos.\n* **Slow orbit:** Deimos takes about 30 hours to complete one orbit around Mars.\n* **Captured asteroid:** Deimos is believed to be a captured asteroid rather than a moon that formed alongside Mars.\n* **Speculation of being artificial:** In 1958, Soviet astronomer Iosif Shklovsky suggested that Deimos might be an artificial satellite placed in orbit around Mars by an advanced extraterrestrial civilization. However, this theory has not gained scientific acceptance."
    }
    ]
    

### Code chat

Gemini 1.5 Pro also supports code generation from a text prompt.


```
code_chat = model.start_chat()

response = code_chat.send_message(
    "Write a function that checks if a year is a leap year"
)

print(response.text)
```

    ```python
    def is_leap_year(year):
      """
      Checks if a year is a leap year.
    
      Args:
        year: The year to check.
    
      Returns:
        True if the year is a leap year, False otherwise.
      """
    
      if year % 4 != 0:
        return False
    
      if year % 100 == 0 and year % 400 != 0:
        return False
    
      return True
    ```
    

You can generate unit tests to test the function in this multi-turn chat.


```
response = code_chat.send_message("Write a unit test of the generated function")

print(response.text)
```

    ```python
    import unittest
    
    class TestIsLeapYear(unittest.TestCase):
    
        def test_is_leap_year(self):
            self.assertTrue(is_leap_year(2000))
            self.assertTrue(is_leap_year(2004))
            self.assertTrue(is_leap_year(2012))
    
        def test_is_not_leap_year(self):
            self.assertFalse(is_leap_year(1900))
            self.assertFalse(is_leap_year(1901))
            self.assertFalse(is_leap_year(2018))
    ```
    

### Add chat history

You can add chat history to a chat by adding messages from role `user` and `model` alternately. System messages can be set in the first part for the first message.


```
chat2 = model.start_chat(
    history=[
        Content(
            role="user",
            parts=[
                Part.from_text(
                    """
    My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.
    Who do you work for?
    """
                )
            ],
        ),
        Content(role="model", parts=[Part.from_text("I work for Ned.")]),
        Content(role="user", parts=[Part.from_text("What do I like?")]),
        Content(role="model", parts=[Part.from_text("Ned likes watching movies.")]),
    ]
)

response = chat2.send_message("Are my favorite movies based on a book series?")
Markdown(response.text)
```




Yes, Lord of the Rings and Hobbit are based on the book series of the same names by J.R.R. Tolkien.




```
response = chat2.send_message("When were these books published?")
Markdown(response.text)
```




The Hobbit was published in 1937 and The Lord of the Rings was published in 1954.



## Sending chat prompts using LangChain

The Gemini API in Vertex AI is integrated with the LangChain Python SDK, making it convenient to build applications on top of Gemini models.

### Start a chat session

You can start a chat by sending chat prompts to the Gemini 1.5 Pro model directly. Gemini 1.5 Pro doesn't support `SystemMessage` at the moment, but `SystemMessage` can be added to the first human message by setting the `convert_system_message_to_human` to `True`.


```
system_message = "You are a helpful assistant who translates English to French."
human_message = "Translate this sentence from English to French. I love programming."

messages = [SystemMessage(content=system_message), HumanMessage(content=human_message)]

chat = ChatVertexAI(
    model_name="gemini-1.5-pro",
    convert_system_message_to_human=True,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
)

result = chat.generate([messages])
print(result.generations[0][0].text)
```

    J'aime la programmation.
    

You can check out the metadata of the generated content.


```
print(result.generations[0][0].generation_info)
```

    {'is_blocked': False, 'safety_ratings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability_label': 'NEGLIGIBLE', 'blocked': False}], 'citation_metadata': None}
    

### Use a chat chain with chat prompt template


```
system_message = "You are a helpful assistant who translates English to French."
human_message = "Translate this sentence from English to French. I love programming."

messages = [SystemMessage(content=system_message), HumanMessage(content=human_message)]
prompt = ChatPromptTemplate.from_messages(messages)

chat = ChatVertexAI(
    model_name="gemini-1.5-pro",
    convert_system_message_to_human=True,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
)

chain = prompt | chat
chain.invoke({})
```




    AIMessage(content="J'adore la programmation.")



### Use a conversation chain

You also can wrap up a chat in `ConversationChain`, which has built-in memory for remembering past user inputs and model outputs.


```
model = ChatVertexAI(
    model_name="gemini-1.5-pro",
    convert_system_message_to_human=True,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant who is good at language translation."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=model, prompt=prompt, verbose=True, memory=memory)

conversation.invoke(
    input="Translate this sentence from English to French. I love programming."
)
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mSystem: You are a helpful assistant who is good at language translation.
    Human: Translate this sentence from English to French. I love programming.[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'Translate this sentence from English to French. I love programming.',
     'history': [HumanMessage(content='Translate this sentence from English to French. I love programming.'),
      AIMessage(content=" J'adore la programmation.")],
     'response': " J'adore la programmation."}




```
conversation.invoke("Translate it to Spanish")
```

    
    
    [1m> Entering new ConversationChain chain...[0m
    Prompt after formatting:
    [32;1m[1;3mSystem: You are a helpful assistant who is good at language translation.
    Human: Translate this sentence from English to French. I love programming.
    AI:  J'adore la programmation.
    Human: Translate it to Spanish[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'Translate it to Spanish',
     'history': [HumanMessage(content='Translate this sentence from English to French. I love programming.'),
      AIMessage(content=" J'adore la programmation."),
      HumanMessage(content='Translate it to Spanish'),
      AIMessage(content='Me encanta programar.')],
     'response': 'Me encanta programar.'}


