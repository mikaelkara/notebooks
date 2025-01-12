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

# Get started with Chirp 2 using Speech-to-Text V2 SDK

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_2_sdk.ipynb">
      <img width="32px" src="https://www.gstatic.com/pantheon/images/bigquery/welcome_page/colab-logo.svg" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Faudio%2Fspeech%2Fgetting-started%2Fget_started_with_chirp_2_sdk.ipynb">
      <img width="32px" src="https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/audio/speech/getting-started/get_started_with_chirp_2_sdk.ipynb">
      <img src="https://www.gstatic.com/images/branding/gcpiconscolors/vertexai/v1/32px.svg" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_2_sdk.ipynb">
      <img width="32px" src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

| | |
|-|-|
| Author(s) | [Ivan Nardini](https://github.com/inardini) |

## Overview

In this tutorial, you learn about how to use Chirp 2, the latest generation of Google's multilingual ASR-specific models.

Chirp 2 improves upon the original Chirp model in accuracy and speed, as well as expanding into key new features like word-level timestamps, model adaptation, and speech translation.

## Get started

### Install Speech-to-Text SDK and other required packages



```
! apt update -y -qq
! apt install ffmpeg -y -qq
```


```
%pip install --quiet 'google-cloud-speech' 'protobuf<4.21' 'google-auth==2.27.0' 'pydub' 'etils' 'jiwer' 'ffmpeg-python' 'plotly'
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

### Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.


```
BUCKET_NAME = "your-bucket-name-unique"  # @param {type:"string", isTemplate: true}

BUCKET_URI = f"gs://{BUCKET_NAME}"  # @param {type:"string"}
```

**Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.


```
! gsutil mb -l $LOCATION -p $PROJECT_ID $BUCKET_URI
```

### Import libraries


```
from google.cloud.speech_v2.types import cloud_speech
```


```
import io
import os
import subprocess
import time

import IPython.display as ipd
from etils import epath as ep
import jiwer
import pandas as pd
import plotly.graph_objs as go
from pydub import AudioSegment
```

### Set constants


```
INPUT_AUDIO_SAMPLE_FILE_URI = (
    "gs://github-repo/audio_ai/speech_recognition/attention_is_all_you_need_podcast.wav"
)
INPUT_LONG_AUDIO_SAMPLE_FILE_URI = (
    f"{BUCKET_URI}/speech_recognition/data/long_audio_sample.wav"
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


def audio_sample_chunk_n(audio_bytes: bytes, num_chunks: int) -> list[bytes]:
    """
    Chunks an audio sample into a specified number of chunks and returns a list of bytes for each chunk.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    total_duration = len(audio)
    chunk_duration = total_duration // num_chunks

    chunks = []
    start_time = 0

    for _ in range(num_chunks):
        end_time = min(start_time + chunk_duration, total_duration)
        chunk = audio[start_time:end_time]

        audio_bytes_chunk = io.BytesIO()
        chunk.export(audio_bytes_chunk, format="wav")
        audio_bytes_chunk.seek(0)
        chunks.append(audio_bytes_chunk.read())

        start_time = end_time

    return chunks


def audio_sample_merge(audio_chunks: list[bytes]) -> bytes:
    """
    Merges a list of audio chunks into a single audio sample.
    """
    audio = AudioSegment.empty()
    for chunk in audio_chunks:
        audio += AudioSegment.from_file(io.BytesIO(chunk))

    audio_bytes = io.BytesIO()
    audio.export(audio_bytes, format="wav")
    audio_bytes.seek(0)

    return audio_bytes.read()


def compress_for_streaming(audio_bytes: bytes) -> bytes:
    """
    Compresses audio bytes for streaming using ffmpeg, ensuring the output size is under 25600 bytes.
    """

    # Temporary file to store original audio
    with open("temp_original.wav", "wb") as f:
        f.write(audio_bytes)

    # Initial compression attempt with moderate bitrate
    bitrate = "32k"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            "temp_original.wav",
            "-b:a",
            bitrate,
            "-y",
            "temp_compressed.mp3",
        ]
    )

    # Check if compressed size is within limit
    compressed_size = os.path.getsize("temp_compressed.mp3")
    if compressed_size <= 25600:
        with open("temp_compressed.mp3", "rb") as f:
            compressed_audio_bytes = f.read()
    else:
        # If too large, reduce bitrate and retry
        while compressed_size > 25600:
            bitrate = str(int(bitrate[:-1]) - 8) + "k"  # Reduce bitrate by 8kbps
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    "temp_original.wav",
                    "-b:a",
                    bitrate,
                    "-y",
                    "temp_compressed.mp3",
                ]
            )
            compressed_size = os.path.getsize("temp_compressed.mp3")

        with open("temp_compressed.mp3", "rb") as f:
            compressed_audio_bytes = f.read()

    # Clean up temporary files
    os.remove("temp_original.wav")
    os.remove("temp_compressed.mp3")

    return compressed_audio_bytes


def parse_streaming_recognize_response(response) -> list[tuple[str, int]]:
    """Parse streaming responses from the Speech-to-Text API"""
    streaming_recognize_results = []
    for r in response:
        for result in r.results:
            streaming_recognize_results.append(
                (result.alternatives[0].transcript, result.result_end_offset)
            )
    return streaming_recognize_results


def parse_real_time_recognize_response(response) -> list[tuple[str, int]]:
    """Parse real-time responses from the Speech-to-Text API"""
    real_time_recognize_results = []
    for result in response.results:
        real_time_recognize_results.append(
            (result.alternatives[0].transcript, result.result_end_offset)
        )
    return real_time_recognize_results


def parse_batch_recognize_response(
    response, audio_sample_file_uri: str = INPUT_LONG_AUDIO_SAMPLE_FILE_URI
) -> list[tuple[str, int]]:
    """Parse batch responses from the Speech-to-Text API"""
    batch_recognize_results = []
    for result in response.results[
        audio_sample_file_uri
    ].inline_result.transcript.results:
        batch_recognize_results.append(
            (result.alternatives[0].transcript, result.result_end_offset)
        )
    return batch_recognize_results


def get_recognize_output(
    audio_bytes: bytes, recognize_results: list[tuple[str, int]]
) -> list[tuple[bytes, str]]:
    """
    Get the output of recognize results, handling 0 timedelta and ensuring no overlaps or gaps.
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    recognize_output = []
    start_time = 0

    initial_end_time = recognize_results[0][1].total_seconds() * 1000

    # This loop handles the streaming case where result timestamps might be zero.
    if initial_end_time == 0:
        for i, (transcript, timedelta) in enumerate(recognize_results):
            if i < len(recognize_results) - 1:
                # Use the next timedelta if available
                next_end_time = recognize_results[i + 1][1].total_seconds() * 1000
                end_time = next_end_time
            else:
                next_end_time = len(audio)
                end_time = next_end_time

            # Ensure no gaps between chunks
            chunk = audio[start_time:end_time]
            chunk_bytes = io.BytesIO()
            chunk.export(chunk_bytes, format="wav")
            chunk_bytes.seek(0)
            recognize_output.append((chunk_bytes.read(), transcript))

            # Set start_time for the next iteration
            start_time = end_time
    else:
        for i, (transcript, timedelta) in enumerate(recognize_results):
            # Calculate end_time in milliseconds
            end_time = timedelta.total_seconds() * 1000

            # Ensure no gaps between chunks
            chunk = audio[start_time:end_time]
            chunk_bytes = io.BytesIO()
            chunk.export(chunk_bytes, format="wav")
            chunk_bytes.seek(0)
            recognize_output.append((chunk_bytes.read(), transcript))

            # Set start_time for the next iteration
            start_time = end_time

    return recognize_output


def print_transcription(audio_sample_bytes: bytes, transcription: str) -> None:
    """Prettify the play of the audio and the associated print of the transcription text in a notebook"""

    # Play the audio sample
    display(ipd.HTML("<b>Audio:</b>"))
    play_audio_sample(audio_sample_bytes)
    display(ipd.HTML("<br>"))

    # Display the transcription text
    display(ipd.HTML("<b>Transcription:</b>"))
    formatted_text = f"<pre style='font-family: monospace; white-space: pre-wrap;'>{transcription}</pre>"
    display(ipd.HTML(formatted_text))


def evaluate_stt(
    actual_transcriptions: list[str],
    reference_transcriptions: list[str],
    audio_sample_file_uri: str = INPUT_LONG_AUDIO_SAMPLE_FILE_URI,
) -> pd.DataFrame:
    """
    Evaluate speech-to-text (STT) transcriptions against reference transcriptions.
    """
    audio_uris = [audio_sample_file_uri] * len(actual_transcriptions)
    evaluations = []
    for audio_uri, actual_transcription, reference_transcription in zip(
        audio_uris, actual_transcriptions, reference_transcriptions
    ):
        evaluation = {
            "audio_uri": audio_uri,
            "actual_transcription": actual_transcription,
            "reference_transcription": reference_transcription,
            "wer": jiwer.wer(reference_transcription, actual_transcription),
            "cer": jiwer.cer(reference_transcription, actual_transcription),
        }
        evaluations.append(evaluation)

    evaluations_df = pd.DataFrame(evaluations)
    evaluations_df.reset_index(inplace=True, drop=True)
    return evaluations_df


def plot_evaluation_results(
    evaluations_df: pd.DataFrame,
) -> go.Figure:
    """
    Plot the mean Word Error Rate (WER) and Character Error Rate (CER) from the evaluation results.
    """
    mean_wer = evaluations_df["wer"].mean()
    mean_cer = evaluations_df["cer"].mean()

    trace_means = go.Bar(
        x=["WER", "CER"], y=[mean_wer, mean_cer], name="Mean Error Rate"
    )

    trace_baseline = go.Scatter(
        x=["WER", "CER"], y=[0.5, 0.5], mode="lines", name="Baseline (0.5)"
    )

    layout = go.Layout(
        title="Speech-to-Text Evaluation Results",
        xaxis=dict(title="Metric"),
        yaxis=dict(title="Error Rate", range=[0, 1]),
        barmode="group",
    )

    fig = go.Figure(data=[trace_means, trace_baseline], layout=layout)
    return fig
```

## Transcribe using Chirp 2

You can use Chirp 2 to transcribe audio in Streaming, Online and Batch modes:

* Streaming mode is good for streaming and real-time audio.  
* Online mode is good for short audio < 1 min.
* Batch mode is good for long audio 1 min to 8 hrs. 

In the following sections, you explore how to use the API to transcribe audio in these three different scenarios.

### Read the audio file

Let's start reading the input audio sample you want to transcribe.

In this case, it is a podcast generated with NotebookLM about the "Attention is all you need" [paper](https://arxiv.org/abs/1706.03762).


```
input_audio_bytes = read_audio_file(INPUT_AUDIO_SAMPLE_FILE_URI)
```

### Prepare audio samples

The podcast audio is ~ 8 mins. Depending on the audio length, you can use different transcribe API methods. To learn more, check out the official documentation.  

#### Prepare a short audio sample (< 1 min)

Extract a short audio sample from the original one for streaming and real-time audio processing.


```
short_audio_sample_bytes = extract_audio_sample(input_audio_bytes, 30)
```


```
play_audio_sample(short_audio_sample_bytes)
```

#### Prepare a long audio sample (from 1 min up to 8 hrs)

Extract a longer audio sample from the original one for batch audio processing.


```
long_audio_sample_bytes = extract_audio_sample(input_audio_bytes, 120)
```


```
play_audio_sample(long_audio_sample_bytes)
```


```
save_audio_sample(long_audio_sample_bytes, INPUT_LONG_AUDIO_SAMPLE_FILE_URI)
```

### Perform streaming speech recognition

Let's start performing streaming speech recognition.

#### Prepare the audio stream

To simulate an audio stream, you can create a generator yielding chunks of audio data.


```
stream = [
    compress_for_streaming(audio_chuck)
    for audio_chuck in audio_sample_chunk_n(short_audio_sample_bytes, num_chunks=5)
]
```


```
for s in stream:
    play_audio_sample(s)
```

#### Prepare the stream request

Once you have your audio stream, you can use the `StreamingRecognizeRequest`class to convert each stream component into a API message.


```
audio_requests = (cloud_speech.StreamingRecognizeRequest(audio=s) for s in stream)
```

#### Define streaming recognition configuration

Next, you define the streaming recognition configuration which allows you to set the model to use, language code of the audio and more.


```
streaming_config = cloud_speech.StreamingRecognitionConfig(
    config=cloud_speech.RecognitionConfig(
        language_codes=["en-US"],
        model="chirp_2",
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
    )
)
```

#### Define the streaming request configuration

Then, you use the streaming configuration to define the streaming request. 


```
stream_request_config = cloud_speech.StreamingRecognizeRequest(
    streaming_config=streaming_config, recognizer=RECOGNIZER
)
```

#### Run the streaming recognition request

Finally, you are able to run the streaming recognition request.


```
def requests(request_config: cloud_speech.RecognitionConfig, s: list) -> list:
    yield request_config
    yield from s


response = client.streaming_recognize(
    requests=requests(stream_request_config, audio_requests)
)
```

Here you use a helper function to visualize transcriptions and the associated streams.


```
streaming_recognize_results = parse_streaming_recognize_response(response)
streaming_recognize_output = get_recognize_output(
    short_audio_sample_bytes, streaming_recognize_results
)
```


```
for audio_sample_bytes, transcription in streaming_recognize_output:
    print_transcription(audio_sample_bytes, transcription)
```

### Perform real-time speech recognition

#### Define real-time recognition configuration

As for the streaming transcription, you define the real-time recognition configuration which allows you to set the model to use, language code of the audio and more.


```
real_time_config = cloud_speech.RecognitionConfig(
    language_codes=["en-US"],
    model="chirp_2",
    features=cloud_speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
    ),
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
)
```

#### Define the real-time request configuration

Next, you define the real-time request passing the configuration and the audio sample you want to transcribe. Again, you don't need to define a recognizer.


```
real_time_request = cloud_speech.RecognizeRequest(
    config=real_time_config,
    content=short_audio_sample_bytes,
    recognizer=RECOGNIZER,
)
```

#### Run the real-time recognition request

Finally you submit the real-time recognition request.


```
response = client.recognize(request=real_time_request)

real_time_recognize_results = parse_real_time_recognize_response(response)
```

And you use a helper function to visualize transcriptions and the associated streams.


```
for transcription, _ in real_time_recognize_results:
    print_transcription(short_audio_sample_bytes, transcription)
```

### Perform batch speech recognition

#### Define batch recognition configuration

You start defining the batch recognition configuration.


```
batch_recognition_config = cloud_speech.RecognitionConfig(
    language_codes=["en-US"],
    model="chirp_2",
    features=cloud_speech.RecognitionFeatures(
        enable_automatic_punctuation=True,
    ),
    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
)
```

#### Set the audio file you want to transcribe

For the batch transcription, you need the audio be staged in a Cloud Storage bucket. Then you set the associated metadata to pass in the batch recognition request.


```
audio_metadata = cloud_speech.BatchRecognizeFileMetadata(
    uri=INPUT_LONG_AUDIO_SAMPLE_FILE_URI
)
```

#### Define batch recognition request

Next, you define the batch recognition request. Notice how you define a recognition output configuration which allows you to determine how would you retrieve the resulting transcription outcome.


```
batch_recognition_request = cloud_speech.BatchRecognizeRequest(
    config=batch_recognition_config,
    files=[audio_metadata],
    recognition_output_config=cloud_speech.RecognitionOutputConfig(
        inline_response_config=cloud_speech.InlineOutputConfig(),
    ),
    recognizer=RECOGNIZER,
)
```

#### Run the batch recognition request

Finally you submit the batch recognition request which is a [long-running operation](https://google.aip.dev/151) as you see below.


```
operation = client.batch_recognize(request=batch_recognition_request)
```


```
while True:
    if not operation.done():
        print("Waiting for operation to complete...")
        time.sleep(5)
    else:
        print("Operation completed.")
        break
```

After the operation finishes, you can retrieve the result as shown below.


```
response = operation.result()
```

And visualize transcriptions using a helper function.


```
batch_recognize_results = parse_batch_recognize_response(
    response, audio_sample_file_uri=INPUT_LONG_AUDIO_SAMPLE_FILE_URI
)
batch_recognize_output = get_recognize_output(
    long_audio_sample_bytes, batch_recognize_results
)
for audio_sample_bytes, transcription in batch_recognize_output:
    print_transcription(audio_sample_bytes, transcription)
```

### Evaluate transcriptions

Finally, you may want to evaluate Chirp transcriptions. To do so, you can use [JiWER](https://github.com/jitsi/jiwer), a simple and fast Python package which supports several metrics. In this tutorial, you use:

- **WER (Word Error Rate)** which is the most common metric. WER is the number of word edits (insertions, deletions, substitutions) needed to change the recognized text to match the reference text, divided by the total number of words in the reference text.
- **CER (Character Error Rate)** which is the number of character edits (insertions, deletions, substitutions) needed to change the recognized text to match the reference text, divided by the total number of characters in the reference text.


```
actual_transcriptions = [t for _, t in batch_recognize_output]
reference_transcriptions = [
    """Okay, so, you know, everyone's been talking about AI lately, right? Writing poems, like nailing those tricky emails, even building websites and all you need is a few, what do they call it again? Prompts? Yeah, it's wild. These AI tools are suddenly everywhere. It's hard to keep up. Seriously. But here's the thing, a lot of this AI stuff we're seeing, it all goes back to this one research paper from way back in 2017. Attention is all you need. So, today we're doing a deep dive into the core of it. The engine that's kind of driving""",
    """all this change. The Transformer. It's funny, right? This super technical paper, I mean, it really did change how we think about AI and how it uses language. Totally. It's like it, I don't know, cracked a code or something. So, before we get into the transformer, we need to like paint that before picture. Can you take us back to how AI used to deal with language before this whole transformer thing came along? Okay. So, imagine this. You're trying to understand a story, but you can only read like one word at a time. Ouch. Right. And not only that, but you also""",
    """have to like remember every single word you read before just to understand the word you're on right now. That sounds so frustrating, like trying to get a movie by looking at one pixel at a time. Exactly. And that's basically how old AI models used to work. RNNs, recurrent neural networks, they processed language one word after the other, which, you can imagine, was super slow and not that great at handling how, you know, language actually works. So, like remembering how the start of a sentence connects""",
    """to the end or how something that happens at the beginning of a book affects what happens later on. That was really tough for older AI. Totally. It's like trying to get a joke by only remembering the punch line. You miss all the important stuff, all that context. Okay, yeah. I'm starting to see why this paper was such a big deal. So how did "Attention Is All You Need" change everything? What's so special about this Transformer thing? Well, I mean, even the title is a good hint, right? It's all about attention. This paper introduced self-attention. Basically, it's how the""",
]

evaluation_df = evaluate_stt(actual_transcriptions, reference_transcriptions)
plot_evaluation_results(evaluation_df)
```

## Cleaning up


```
delete_bucket = False

if delete_bucket:
    ! gsutil rm -r $BUCKET_URI
```
