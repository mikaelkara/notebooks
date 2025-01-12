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

# Introduction to Long Context Window with Gemini on Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/long-context/intro_long_context.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Flong-context%2Fintro_long_context.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/long-context/intro_long_context.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/long-context/intro_long_context.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Holt Skinner](https://github.com/holtskinner) |

## Overview

Gemini 1.5 Flash comes standard with a 1 million token context window, and Gemini 1.5 Pro comes with a 2 million token context window. Historically, large language models (LLMs) were significantly limited by the amount of text (or tokens) that could be passed to the model at one time. The Gemini 1.5 long context window, with [near-perfect retrieval (>99%)](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf), unlocks many new use cases and developer paradigms.

In practice, 1 million tokens would look like:

-   50,000 lines of code (with the standard 80 characters per line)
-   All the text messages you have sent in the last 5 years
-   8 average length English novels
-   Transcripts of over 200 average length podcast episodes
-   1 hour of video
-   ~45 minutes of video with audio
-   9.5 hours of audio

While the standard use case for most generative models is still text input, the Gemini 1.5 model family enables a new paradigm of multimodal use cases. These models can natively understand text, video, audio, and images.

In this notebook, we'll explore multimodal use cases of the long context window.

For more information, refer to the [Gemini documentation about long context](https://ai.google.dev/gemini-api/docs/long-context).

## Tokens

Tokens can be single characters like `z` or whole words like `cat`. Long words
are broken up into several tokens. The set of all tokens used by the model is
called the vocabulary, and the process of splitting text into tokens is called
_tokenization_.

> **Important:** For Gemini models, a token is equivalent to about 4 characters. 100 tokens is equal to about 60-80 English words.

For multimodal input, this is how tokens are calculated regardless of display or file size:

* Images: `258` tokens
* Video: `263` tokens per second
* Audio: `32` tokens per second

## Why is the long context window useful?

The basic way you use the Gemini models is by passing information (context)
to the model, which will subsequently generate a response. An analogy for the
context window is short term memory. There is a limited amount of information
that can be stored in someone's short term memory, and the same is true for
generative models.

You can read more about how models work under the hood in our [generative models guide](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview).

Even though the models can take in more and more context, much of the
conventional wisdom about using large language models assumes this inherent
limitation on the model, which as of 2024, is no longer the case.

Some common strategies to handle the limitation of small context windows
included:

-   Arbitrarily dropping old messages / text from the context window as new text
    comes in
-   Summarizing previous content and replacing it with the summary when the
    context window gets close to being full
-   Using RAG with semantic search to move data out of the context window and
    into a vector database
-   Using deterministic or generative filters to remove certain text /
    characters from prompts to save tokens

While many of these are still relevant in certain cases, the default place to start is now just putting all of the tokens into the context window. Because Gemini 1.5 models were purpose-built with a long context window, they are much more capable of in-context learning. This means that instructional materials provided in context can be highly effective for handling inputs that are not covered by the model's training data.

## Getting Started

### Install Vertex AI SDK for Python



```
%pip install --upgrade --user --quiet google-cloud-aiplatform
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.


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
PROJECT_ID = "YOUR_PROJECT_ID"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION)
```

### Import libraries



```
from IPython.display import Markdown, display
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part
```

### Load the Gemini 1.5 Flash model

To learn more about all [Gemini API models on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models).



```
MODEL_ID = "gemini-1.5-flash"  # @param {type:"string"}

model = GenerativeModel(
    MODEL_ID, generation_config=GenerationConfig(max_output_tokens=8192)
)
```

## Long-form text

Text has proved to be the layer of intelligence underpinning much of the momentum around LLMs. As mentioned earlier, much of the practical limitation of LLMs was because of not having a large enough context window to do certain tasks. This led to the rapid adoption of retrieval augmented generation (RAG) and other techniques which dynamically provide the model with relevant
contextual information.

Some emerging and standard use cases for text based long context include:

-   Summarizing large corpuses of text
    -   Previous summarization options with smaller context models would require
        a sliding window or another technique to keep state of previous sections
        as new tokens are passed to the model
-   Question and answering
    -   Historically this was only possible with RAG given the limited amount of
        context and models' factual recall being low
-   Agentic workflows
    -   Text is the underpinning of how agents keep state of what they have done
        and what they need to do; not having enough information about the world
        and the agent's goal is a limitation on the reliability of agents

[War and Peace by Leo Tolstoy](https://en.wikipedia.org/wiki/War_and_Peace) is considered one of the greatest literary works of all time; however, it is over 1,225 pages and the average reader will spend 37 hours and 48 minutes reading this book at 250 WPM (words per minute). 😵‍💫 The text alone takes up 3.4 MB of storage space. However, the entire novel consists of less than 900,000 tokens, so it will fit within the Gemini context window.

We are going to pass in the entire text into Gemini 1.5 Flash and get a detailed summary of the plot. For this example, we have the text of the novel from [Project Gutenberg](https://www.gutenberg.org/ebooks/2600) stored in a public Google Cloud Storage bucket.

First, we will use the `count_tokens()` method to examine the token count of the full prompt, then send the prompt to Gemini.


```
# Set contents to send to the model
contents = [
    "Provide a detailed summary of the following novel.",
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/WarAndPeace.txt",
        mime_type="text/plain",
    ),
]

# Counts tokens
print(model.count_tokens(contents))

# Prompt the model to generate content
response = model.generate_content(
    contents,
)

# Print the model response
print(f"\nUsage metadata:\n{response.usage_metadata}")

display(Markdown(response.text))
```

## Long-form video

Video content has been difficult to process due to constraints of the format itself.
It was hard to skim the content, transcripts often failed to capture the nuance of a video, and most tools don't process images, text, and audio together.
The Gemini 1.5 long context window allows the ability to reason and answer questions about multimodal inputs with
sustained performance.

When tested on the needle in a video haystack problem with 1M tokens, Gemini 1.5 Flash obtained >99.8% recall of the video in the context window, and Gemini 1.5 Pro reached state of the art performance on the [Video-MME benchmark](https://video-mme.github.io/home_page.html).

Some emerging and standard use cases for video long context include:

-   Video question and answering
-   Video memory, as shown with [Google's Project Astra](https://deepmind.google/technologies/gemini/project-astra/)
-   Video captioning
-   Video recommendation systems, by enriching existing metadata with new
    multimodal understanding
-   Video customization, by looking at a corpus of data and associated video
    metadata and then removing parts of videos that are not relevant to the
    viewer
-   Video content moderation
-   Real-time video processing

[Google I/O](https://io.google/) is one of the major events when Google's developer tools are announced. Workshop sessions and are filled with a lot of material, so it can be difficult to keep track all that is discussed.

We are going to use a video of a session from Google I/O 2024 focused on [Grounding for Gemini](https://www.youtube.com/watch?v=v4s5eU2tfd4) to calculate tokens and process the information presented. We will ask a specific question about a point in the video and ask for a general summary.


```
# Set contents to send to the model
video = Part.from_uri(
    "gs://github-repo/generative-ai/gemini/long-context/GoogleIOGroundingRAG.mp4",
    mime_type="video/mp4",
)

contents = ["At what time in the following video is the Cymbal Starlight demo?", video]

# Counts tokens
print(model.count_tokens(contents))

# Prompt the model to generate content
response = model.generate_content(
    contents,
)

# Print the model response
print(f"\nUsage metadata:\n{response.usage_metadata}")

display(Markdown(response.text))
```


```
contents = [
    "Provide an enthusiastic summary of the video, tailored for software developers.",
    video,
]

# Counts tokens
print(model.count_tokens(contents))

# Prompt the model to generate content
response = model.generate_content(contents)

# Print the model response
print(f"\nUsage metadata:\n{response.usage_metadata}")

display(Markdown(response.text))
```

## Long-form audio

In order to process audio, developers have typically needed to string together multiple models, like a speech-to-text model and a text-to-text model, in order to process audio. This led to additional latency due to multiple round-trip requests, and the context of the audio itself could be lost.

The Gemini 1.5 models were the first natively multimodal large language models that could understand audio.

On standard audio-haystack evaluations, Gemini 1.5 Pro is able to find the hidden audio in 100% of the tests and Gemini 1.5 Flash is able to find it in 98.7% [of the tests](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf). Further, on a test set of 15-minute audio clips, Gemini 1.5 Pro archives a word error rate (WER) of ~5.5%, much lower than even specialized speech-to-text models, without the added complexity of extra input segmentation and pre-processing.

The long context window accepts up to 9.5 hours of audio in a single request.

Some emerging and standard use cases for audio context include:

-   Real-time transcription and translation
-   Podcast / video question and answering
-   Meeting transcription and summarization
-   Voice assistants

Podcasts are a great way to learn about the latest news in technology, but there are so many out there that it can be difficult to follow them all. It's also challenging to find a specific episode with a given topic or a quote.

In this example, we will process 9 episodes of the [Google Kubernetes Podcast](https://cloud.google.com/podcasts/kubernetespodcast) and ask specific questions about the content.


```
# Set contents to send to the model
contents = [
    "According to the following podcasts, what can you tell me about AI/ML workloads on Kubernetes?",
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240417-kpod223.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240430-kpod224.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240515-kpod225.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240529-kpod226.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240606-kpod227.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240611-kpod228.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240625-kpod229.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240709-kpod230.mp3",
        mime_type="audio/mpeg",
    ),
    Part.from_uri(
        "gs://github-repo/generative-ai/gemini/long-context/20240723-kpod231.mp3",
        mime_type="audio/mpeg",
    ),
]

# Counts tokens
print(model.count_tokens(contents))

# Prompt the model to generate content
response = model.generate_content(
    contents,
)

# Print the model response
print(f"\nUsage metadata:\n{response.usage_metadata}")

display(Markdown(response.text))
```

## Code

For a long context window use case involving ingesting an entire GitHub repository, check out [Analyze a codebase with Vertex AI Gemini 1.5 Pro](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/code/analyze_codebase_with_gemini_1_5_pro.ipynb)

## Context caching

[Context caching](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview) allows developers to reduce the time and cost of repeated requests using the large context window.
For examples on how to use Context Caching with Gemini on Vertex AI, refer to [Intro to Context Caching with Gemini on Vertex AI](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/context-caching/intro_context_caching.ipynb)
