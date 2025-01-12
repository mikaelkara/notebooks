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

# Get started with Chirp 2 - Advanced features

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Faudio%2Fspeech%2Fgetting-started%2Fget_started_with_chirp_2_sdk_features.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Ivan Nardini](https://github.com/inardini) |

## Overview

In this tutorial, you learn about how to use [Chirp 2](https://cloud.google.com/speech-to-text/v2/docs/chirp_2-model), the latest generation of Google's multilingual ASR-specific models, and its new features, including word-level timestamps, model adaptation, and speech translation.

## Get started

### Install Speech-to-Text SDK and other required packages



```
! apt update -y -qq
! apt install ffmpeg -y -qq
```


```
%pip install --quiet 'google-cloud-speech' 'protobuf<4.21' 'google-auth==2.27.0' 'pydub' 'etils' 'jiwer' 'ffmpeg-python' 'plotly' 'gradio'
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
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>


### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Set Google Cloud project information and initialize Speech-to-Text V2 SDK

To get started using the Speech-to-Text API, you must have an existing Google Cloud project and [enable the Speech-to-Text API](https://console.cloud.google.com/flows/enableapi?apiid=speech.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).


```
import os

PROJECT_ID = "[your-project-id]"  # @param {type:"string", isTemplate: true}

if PROJECT_ID == "[your-project-id]":
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
```


```
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient

API_ENDPOINT = f"{LOCATION}-speech.googleapis.com"

client = SpeechClient(
    client_options=ClientOptions(
        api_endpoint=API_ENDPOINT,
    )
)
```

### Import libraries


```
from google.cloud.speech_v2.types import cloud_speech
import gradio as gr
```


```
import io
import os

import IPython.display as ipd
from etils import epath as ep
from pydub import AudioSegment
```

### Set constants


```
INPUT_AUDIO_SAMPLE_FILE_URI = (
    "gs://github-repo/audio_ai/speech_recognition/attention_is_all_you_need_podcast.wav"
)

RECOGNIZER = client.recognizer_path(PROJECT_ID, LOCATION, "_")

MAX_CHUNK_SIZE = 25600
```

### Helpers


```
def read_audio_file(audio_file_path: str) -> bytes:
    """
    Read audio file as bytes.
    """
    if audio_file_path.startswith("gs://"):
        with ep.Path(audio_file_path).open("rb") as f:
            audio_bytes = f.read()
    else:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
    return audio_bytes


def save_audio_sample(audio_bytes: bytes, output_file_uri: str) -> None:
    """
    Save audio sample as a file in Google Cloud Storage.
    """

    output_file_path = ep.Path(output_file_uri)
    if not output_file_path.parent.exists():
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with output_file_path.open("wb") as f:
        f.write(audio_bytes)


def extract_audio_sample(audio_bytes: bytes, duration: int) -> bytes:
    """
    Extracts a random audio sample of a given duration from an audio file.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    start_time = 0
    audio_sample = audio[start_time : start_time + duration * 1000]

    audio_bytes = io.BytesIO()
    audio_sample.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    return audio_bytes.read()


def play_audio_sample(audio_bytes: bytes) -> None:
    """
    Plays the audio sample in a notebook.
    """
    audio_file = io.BytesIO(audio_bytes)
    ipd.display(ipd.Audio(audio_file.read(), rate=44100))


def parse_real_time_recognize_response(response) -> list[tuple[str, int]]:
    """Parse real-time responses from the Speech-to-Text API"""
    real_time_recognize_results = []
    for result in response.results:
        real_time_recognize_results.append(
            (result.alternatives[0].transcript, result.result_end_offset)
        )
    return real_time_recognize_results


def parse_words_real_time_recognize_response(response):
    """
    Parse the word-level results from a real-time speech recognition response.
    """
    real_time_recognize_results = []
    for result in response.results:
        for word_info in result.alternatives[0].words:
            word = word_info.word
            start_time = word_info.start_offset.seconds
            end_time = word_info.end_offset.seconds
            real_time_recognize_results.append(
                {"word": word, "start": start_time, "end": end_time}
            )
    return real_time_recognize_results


def print_transcription(
    audio_sample_bytes: bytes, transcriptions: str, play_audio=True
) -> None:
    """Prettify the play of the audio and the associated print of the transcription text in a notebook"""

    if play_audio:
        # Play the audio sample
        display(ipd.HTML("<b>Audio:</b>"))
        play_audio_sample(audio_sample_bytes)
        display(ipd.HTML("<br>"))

    # Display the transcription text
    display(ipd.HTML("<b>Transcription:</b>"))
    for transcription, _ in transcriptions:
        formatted_text = f"<pre style='font-family: monospace; white-space: pre-wrap;'>{transcription}</pre>"
        display(ipd.HTML(formatted_text))
```

### Prepare audio samples

The podcast audio is ~ 8 mins. Depending on the audio length, you can use different transcribe API methods. To learn more, check out the official documentation.  

#### Read the audio file

Let's start reading the input audio sample you want to transcribe.

In this case, it is a podcast generated with NotebookLM about the "Attention is all you need" [paper](https://arxiv.org/abs/1706.03762).


```
input_audio_bytes = read_audio_file(INPUT_AUDIO_SAMPLE_FILE_URI)
```

#### Prepare a short audio sample (< 1 min)

Extract a short audio sample from the original one for streaming and real-time audio processing.


```
short_audio_sample_bytes = extract_audio_sample(input_audio_bytes, 30)
```


```
play_audio_sample(short_audio_sample_bytes)
```

## Improve transcription using Chirp 2's word-timing and speech adaptation features

Chirp 2 supports word-level timestamps for each transcribed word and speech adaptation to help the model improving recognition accuracy for specific terms or proper nouns.

### Perform real-time speech recognition with word-timing

#### Define real-time recognition configuration with `enable_word_time_offsets` parameter.

You define the real-time recognition configuration which allows you to set the model to use, language code of the audio and more.

In this case, you enable word timing feature. When True, the top result includes a list of words and the start and end time offsets (timestamps) for those words.


```
wt_real_time_config = cloud_speech.RecognitionConfig(
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    model="chirp_2",
    features=cloud_speech.RecognitionFeatures(
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    ),
)
```

#### Define the real-time request configuration

Next, you define the real-time request passing the configuration and the audio sample you want to transcribe.



```
wt_real_time_request = cloud_speech.RecognizeRequest(
    config=wt_real_time_config, content=short_audio_sample_bytes, recognizer=RECOGNIZER
)
```

#### Run the real-time recognition request

Finally you submit the real-time recognition request.


```
wt_response = client.recognize(request=wt_real_time_request)
wt_real_time_recognize_results = parse_real_time_recognize_response(wt_response)
```

And you use a helper function to visualize transcriptions and the associated streams.


```
for transcription, _ in wt_real_time_recognize_results:
    print_transcription(short_audio_sample_bytes, transcription)
```

#### Visualize word timings


```
n = 10
word_timings = parse_words_real_time_recognize_response(wt_response)
for word_info in word_timings[:n]:
    print(
        f"Word: {word_info['word']} - Start: {word_info['start']} sec - End: {word_info['end']} sec"
    )
```

### Improve real-time speech recognition accuracy with model adaptation

So far, Chirp 2 transcribes the podcast correctly. That's in part because podcasts are recorded in ideal enviroments like a recording studio. But that's not always the case. For example, suppose that your audio data is recorded in noisy environment or the recording has strong accents or someone speaks quickly.

To handle this and many other scenarios and improve real-time speech recognition accuracy, you can use model adaptation. To enable model adaptation with Chirp 2, you use the `adaptation` parameter.

With `adaptation` parameter, you provide "hints" to the speech recognizer to favor specific words and phrases (`AdaptationPhraseSet` class) in the results. And for each hint you can define a hint boost which is the probability that a specific word or phrase will be recognized over other similar sounding phrases. Be careful to use higher boost. Higher the boost, higher is the chance of false positive recognition as well. We recommend using a binary search approach to finding the optimal value for your use case as well as adding phrases both with and without boost to your requests.


#### Define real-time recognition configuration with `adaptation` parameter

You define a new real-time recognition configuration which includes the `adaptation` configuration.



```
adaptation_real_time_config = cloud_speech.RecognitionConfig(
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    model="chirp_2",
    features=cloud_speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
    ),
    adaptation=cloud_speech.SpeechAdaptation(
        phrase_sets=[
            cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                inline_phrase_set=cloud_speech.PhraseSet(
                    phrases=[
                        {
                            "value": "you know",  # often mumbled or spoken quickly
                            "boost": 10.0,
                        },
                        {
                            "value": "what are they called again?"  # hesitations and changes in intonation
                        },
                        {
                            "value": "Yeah, it's wild."  # short interjections have brevity and the emotional inflection
                        },
                    ]
                )
            )
        ]
    ),
)
```

#### Define the real-time request configuration


```
adaptation_real_time_request = cloud_speech.RecognizeRequest(
    config=adaptation_real_time_config,
    content=short_audio_sample_bytes,
    recognizer=RECOGNIZER,
)
```

#### Run the real-time recognition request


```
adapted_response = client.recognize(request=adaptation_real_time_request)
adapted_real_time_recognize_results = parse_real_time_recognize_response(
    adapted_response
)
```

And you use a helper function to visualize transcriptions and the associated streams.


```
for transcription, _ in adapted_real_time_recognize_results:
    print_transcription(short_audio_sample_bytes, transcription)
```

## Transcript and translate using language-agnostic transcription and language translation

Chirp 2 supports language-agnostic audio transcription and language translation. This means that Chirp 2 is capable of recognizing the language of the input audio and, at the same time, translate the outcome transcription in many different language.


#### Define real-time recognition configuration with `language_code` and `translationConfig` parameters.

You define a real-time recognition configuration by setting language codes in both `language_codes` and `translationConfig` parameters :

*   When `language_codes=["auto"]`, you enable language-agnostic transcription to auto to detect language.

*  When `target_language=language_code` where `language_code` is one of the language in this list but different from the original language, you enable language translation.


```
target_language_code = "ca-ES"  # @param {type:"string", isTemplate: true}
```


```
ts_real_time_config = cloud_speech.RecognitionConfig(
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    translation_config=cloud_speech.TranslationConfig(
        target_language=target_language_code
    ),
    model="chirp_2",
    features=cloud_speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
    ),
)
```

#### Define the real-time request configuration


```
ts_real_time_request = cloud_speech.RecognizeRequest(
    config=ts_real_time_config, content=short_audio_sample_bytes, recognizer=RECOGNIZER
)
```

#### Run the real-time recognition request


```
ts_response = client.recognize(request=ts_real_time_request)
ts_real_time_recognize_results = parse_real_time_recognize_response(ts_response)
```

And you use a helper function to visualize transcriptions and the associated streams.


```
print_transcription(short_audio_sample_bytes, transcription, play_audio=False)
```

## Chirp 2 playground

To play with Chirp 2, you can create a simple Gradio application where you enable several Chirp 2 features.

Below you have an example for language-agnostic transcription and language translation with Chirp 2.

To know more, check out the official documentation [here](https://cloud.google.com/speech-to-text/v2/docs/chirp_2-model).



```
def transcribe_audio(audio, enable_translation, target_language_code):
    """Transcribe the given audio file with optional features."""

    # Set variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
    location = os.environ.get("GOOGLE_CLOUD_REGION", LOCATION)
    api_endpoint = f"{location}-speech.googleapis.com"

    # initiate client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=api_endpoint,
        )
    )

    # read the audio
    with open(audio, "rb") as audio_file:
        content = audio_file.read()

    # define language agnostic real time recognition configuration
    real_time_config = cloud_speech.RecognitionConfig(
        model="chirp_2",
        language_codes=["auto"],
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    )

    if enable_translation:
        real_time_config.language_codes = ["en-US"]
        real_time_config.translation_config = cloud_speech.TranslationConfig(
            target_language=target_language_code
        )

    # define real-time recognition request
    recognizer = client.recognizer_path(project_id, location, "_")

    real_time_request = cloud_speech.RecognizeRequest(
        config=real_time_config,
        content=content,
        recognizer=recognizer,
    )

    response = client.recognize(request=real_time_request)

    full_transcript = ""
    for result in response.results:
        full_transcript += result.alternatives[0].transcript + " "
    return full_transcript.strip()


def speech_to_text(audio, enable_translation=False, target_language_code=None):
    if audio is None:
        return ""

    text = transcribe_audio(audio, enable_translation, target_language_code)
    return text
```


```
# Create Gradio interface
demo = gr.Interface(
    fn=speech_to_text,
    inputs=[
        gr.Audio(type="filepath", label="Audio input"),
        gr.Checkbox(label="🧠 Enable Translation"),
        gr.Dropdown(
            label="Select language to translate",
            choices=["ca-ES", "cy-GB", "de-DE", "ja-JP", "zh-Hans-CN"],
            interactive=True,
            multiselect=False,
        ),
    ],
    outputs=[gr.Textbox(label="📄 Transcription")],
    title="Chirp 2 Playground",
    description="<p style='text-align: center'> Speak or pass an audio and get the transcription!</p>",
)

# Launch the app
demo.launch()
```


```
demo.close()
```
