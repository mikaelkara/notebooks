```
# Copyright 2023 Google LLC
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

# Get started with Chirp on Google Cloud

<table align="left">

  <td>
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_sdk.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Colab logo"> Run in Colab
    </a>
  </td>
  <td>
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_sdk.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo">
      View on GitHub
    </a>
  </td>
  <td>
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/audio/speech/getting-started/get_started_with_chirp_sdk.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo">
      Open in Vertex AI Workbench
    </a>
  </td>
</table>

| | |
|-|-|
|Author(s) | [Ivan Nardini](https://github.com/inardini) |

## Overview

This notebook demonstrates how to use Chirp for converting voice to text in several languages.

Currently, there are some main challenges in Automatic Speech Recognition, ASR in short. Supervised learning approaches in ASR are not scalable. They are required large amounts of high quality labeled data which has a reasonable language coverage. Also, they are computational expensive. 

As foundational model, Chirp is the next generation of Google's speech-to-text models. Representing the culmination of years of research, the first version of Chirp is now available for Speech-to-Text. With Chirp, you get access to the first 2B parameter speech model which achieved 98% accuracy on English and over 300% relative improvement in tail languages.

For details, check out the official documentation about [Chirp](https://cloud.google.com/speech-to-text/v2/docs/chirp-model).

### Objective

In this tutorial, you will use Chirp to transcribe short and long audio clips. You will also evaluate the model results using automatic speech recognition system metrics.

This tutorial uses the following Google Cloud ML services and resources:

- Cloud Storage
- [Cloud Speech-to-Text API (v2)](https://cloud.google.com/speech-to-text/v2/docs)

The steps performed include:

- Get audio file to process
- Transcribe short audio files
- Transcribe long audio files
- Evaluate transcriptions

### Costs

This tutorial uses billable components of Google Cloud:

* Speech-to-text
* Cloud Storage

Learn about [Speech-to-text pricing](https://cloud.google.com/speech-to-text/pricing),
and [Cloud Storage pricing](https://cloud.google.com/storage/pricing),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Getting Started


### Install Vertex AI SDK, other packages and their dependencies

Install the following packages required to execute this notebook.


```
# Install the packages
%pip install --user --upgrade google-cloud-speech librosa jiwer protobuf
```

### Colab only: Uncomment the following cell to restart the kernel.

***Colab only***: Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.


```
# Automatically restart kernel after installs so that your environment can access the new packages
# import IPython

# app = IPython.Application.instance()
# app.kernel.do_shutdown(True)
```

### Authenticating your notebook environment
* If you are using **Colab** to run this notebook, uncomment the cell below and continue.
* If you are using **Vertex AI Workbench**, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).


```
# from google.colab import auth

# auth.authenticate_user()
```

### Set Google Cloud project information

To get started using Speech-to-Text, Cloud Storage, and Vertex AI, you must have an existing Google Cloud project and [enable the APIs](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com,storage.googleapis.com,speech.googleapis.com).

Learn more about [setting up a project and a development environment](https://cloud.google.com/vertex-ai/docs/start/cloud-environment).

Please note the **available regions** for Chirp, see [documentation](https://cloud.google.com/speech-to-text/v2/docs/speech-to-text-supported-languages)


```
PROJECT_ID = "[your-project-id]"  # @param {type:"string"}

REGION = "us-central1"  # @param {type:"string"}
```

### Create a Cloud Storage bucket

Create a storage bucket to store intermediate artifacts such as datasets.


```
BUCKET_URI = "gs://your-bucket-name-unique"  # @param {type:"string"}
```

**Only if your bucket doesn't already exist**: Run the following cell to create your Cloud Storage bucket.


```
! gsutil mb -l $REGION -p $PROJECT_ID $BUCKET_URI
```

### Import libraries


```
import json
from pathlib import Path as p
from pprint import pprint

from IPython.display import Audio as play
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import jiwer
import librosa
import pandas as pd
```

### Helper functions


```
def get_stt_metric(truth, transcription, metric="wer"):
    """
    A function to calculate common automatic speech recognition system metrics using JiWER library.
    Default metrics is Word error rate (WER) which is a measure of the misrecognized words.
    Possible metrics are:
    - Word information lost (WIL) is a measure of the amount of information that is lost when the model transcribes a word.
    - Word information preserved (WIP) is a measure of the amount of information that is preserved.
    """
    chirp_evaluation = jiwer.compute_measures(truth, transcription)
    return chirp_evaluation.get(metric)
```

### Prepare data

You load some existing audio file from a public Google Cloud Storage bucket. In this tutorial, you will use `brooklyn.flac` and `vr.wav` audio files.


```
data_folder = p.cwd() / "data"
p(data_folder).mkdir(parents=True, exist_ok=True)
```

#### Short audio

As short audio, let's use `brooklyn.flac` which is less than 60 seconds.


```
short_audio_uri = (
    "gs://cloud-samples-tests/speech/brooklyn.flac"  # @param {type:"string"}
)
```

To verify if the audio length, you can use `librosa`, a python package for music and audio analysis. 


```
short_audio_path = str(data_folder / "short_audio.flac")
! gsutil cp {short_audio_uri} {short_audio_path}

short_audio_duration = librosa.get_duration(path=short_audio_path)
if short_audio_duration > 60:
    raise Exception(
        f"The audio is longer than 60 sec. Please use a GCS url for audio longer than 60 sec. Actual length: {short_audio_duration}"
    )
```


```
play(short_audio_path)
```

#### Long audio

For long audio, you can use `vr.wav` which is longer than 60 seconds.


```
long_audio_origin_uri = (
    "gs://cloud-samples-tests/speech/vr.wav"  # @param {type:"string"}
)
```

As before, let's verify the audio length.


```
long_audio_uri = f"{BUCKET_URI}/data/vr.wav"  # @param {type:"string"}
long_audio_path = str(data_folder / "long_audio.wav")
!gsutil cp {long_audio_origin_uri} {long_audio_uri}
!gsutil cp {long_audio_uri} {long_audio_path}

long_audio_duration = librosa.get_duration(path=long_audio_path)
if long_audio_duration < 60:
    raise Exception(
        f"The audio is less than 1 min. Actual length: {long_audio_duration}"
    )
```


```
play(long_audio_path)
```

## Transcribe short audio files (< 1 min)

To transcribe short audio files, you need to create a Recognizer which allows you to define the model used for recognition and a list of settings used for recognition. Then you use the Recognizer to run a recognizer request which generates the transcription.



```
client = SpeechClient(
    client_options=ClientOptions(api_endpoint=f"{REGION}-speech.googleapis.com")
)
```

### Create a recognizer

First, you need to initiate a Recognizer which uses the Chirp model and transcribe the audio in English.

See [the documentation](https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.CreateRecognizerRequest) to learn more about how to configure the `CreateRecognizerRequest` request.


```
language_code = "en-US"
recognizer_id = f"chirp-{language_code.lower()}-test"

recognizer_request = cloud_speech.CreateRecognizerRequest(
    parent=f"projects/{PROJECT_ID}/locations/{REGION}",
    recognizer_id=recognizer_id,
    recognizer=cloud_speech.Recognizer(
        language_codes=[language_code],
        model="chirp",
    ),
)
```

Then, you create a Speech-to-Text [Recognizer](https://cloud.google.com/speech-to-text/v2/docs/recognizers) that uses Chirp running a create operation.


```
create_operation = client.create_recognizer(request=recognizer_request)
recognizer = create_operation.result()
```


```
recognizer
```

### Transcribe a short audio

After you create a Speech-to-Text Recognizer that uses Chirp, you are ready to transcribe your audio.

You can create a recognition configuration and the associated recognition request.


```
with open(short_audio_path, "rb") as f:
    content = f.read()

short_audio_config = cloud_speech.RecognitionConfig(
    features=cloud_speech.RecognitionFeatures(
        enable_automatic_punctuation=True, enable_word_time_offsets=True
    ),
    auto_decoding_config={},
)

short_audio_request = cloud_speech.RecognizeRequest(
    recognizer=recognizer.name, config=short_audio_config, content=content
)

short_audio_response = client.recognize(request=short_audio_request)
```

Below you can see the transcribed audio.


```
short_audio_transcription = short_audio_response.results[0].alternatives[0].transcript
pprint(short_audio_transcription)
```

## Transcribe long audio files (> 1 min)

To transcribe long audio files, you use the same process described above but you can run a batch recognition request which uses some audio uploaded on a bucket.


```
transcriptions_folder = p.cwd() / "transcriptions"
p(transcriptions_folder).mkdir(parents=True, exist_ok=True)
```

### Transcribe a long audio

Unlike the short audio transcription, the recognition request requires information about the bucket location of the audio file to transcribe and the bucket destination of transcriptions.


```
long_audio_config = cloud_speech.RecognitionConfig(
    features=cloud_speech.RecognitionFeatures(
        enable_automatic_punctuation=True, enable_word_time_offsets=True
    ),
    auto_decoding_config={},
)

long_audio_request = cloud_speech.BatchRecognizeRequest(
    recognizer=recognizer.name,
    recognition_output_config={
        "gcs_output_config": {"uri": f"{BUCKET_URI}/transcriptions"}
    },
    files=[{"config": long_audio_config, "uri": long_audio_uri}],
)


long_audio_operation = client.batch_recognize(request=long_audio_request)
```

Below you can see the result of the transcription job.


```
long_audio_result = long_audio_operation.result()
print(long_audio_result)
```

### Get transcriptions

To see the result of transcription job, you get the generated transcription file.


```
transcriptions_uri = long_audio_result.results[long_audio_uri].uri
transcriptions_file_path = str(data_folder / "transcriptions.text")

! gsutil cp {transcriptions_uri} {transcriptions_file_path}
```


```
transcriptions = json.loads(open(transcriptions_file_path).read())
transcriptions = transcriptions["results"]
transcriptions = [
    transcription["alternatives"][0]["transcript"]
    for transcription in transcriptions
    if "alternatives" in transcription.keys()
]
long_audio_transcription = " ".join(transcriptions)
print(long_audio_transcription)
```


```
transcriptions
```

## Evaluate transcriptions

Finally, you may want to evaluate Chirp transcriptions. To do so, you can use `[JiWER](https://github.com/jitsi/jiwer)`, a simple and fast Python package which supports several metrics. In this tutorial, you use:

- **Word error rate (WER)** which is the most common metric. It calculates the number of words that are incorrectly recognized divided by the total number of words in the reference transcript.
- **Word information lost (WIL)** is a measure of the amount of information that is lost when the model transcribes a word. It is based on the uncorrected number of phonemes over the total number of phonemes in the word.
- **Word information preserved (WIP)** is a measure of the amount of information that is preserved and it is calculated as the number of phonemes that are correctly recognized divided by the total number of phonemes in the word.


```
audio_uris = [short_audio_uri, long_audio_uri]
actual_transcriptions = [
    """
    how old is the Brooklyn Bridge?
    """,
    """
    so okay, so what am I doing here? why am I here at GDC talking about VR video?
    um, it's because I believe um, my favorite games, I love games, I believe in games,
    my favorite games are the ones that are all about the stories, I love narrative game design,
    I love narrative-based games and I think that when it comes to telling stories in VR,
    bringing together capturing the world with narrative-based games and narrative-based game design,
    is going to unlock some of the killer apps and killer stories of the medium,
    so I'm really here looking for people who are interested in telling those sort of stories,
    that are planning projects around telling those types of stories,
    um and I would love to talk to you, so if this sounds like your project,
    if you're looking at blending VR video and interactivity to tell a story,
    I want to talk to you, um, I want to help you, so if this sounds like you,
    please get in touch, please come find me, I'll be here all week, I have pink
    I work for Google um and I would love to talk with you further about
    um VR video, interactivity and storytelling.
    """,
]
hypothesis_transcriptions = [short_audio_transcription, long_audio_transcription]

evaluations = []
for a, t, h in zip(audio_uris, actual_transcriptions, hypothesis_transcriptions):
    evaluation = {}
    evaluation["audio_uri"] = a
    evaluation["truth"] = t
    evaluation["hypothesis"] = h
    evaluation["wer"] = get_stt_metric(t, h, "wer")
    evaluation["wil"] = get_stt_metric(t, h, "wil")
    evaluation["wip"] = get_stt_metric(t, h, "wip")
    evaluations.append(evaluation)
```


```
evaluations_df = pd.DataFrame.from_dict(evaluations)
evaluations_df.reset_index(inplace=True, drop=True)
evaluations_df
```

## Conclusion

In this tutorial, you learned how to use Chirp for converting English audio to text.

Although Chirp does not currently support many of the Speech-to-Text features, including Speech adaptation and Diarization, Chirp is a powerful new speech-to-text model that can accurately transcribe audio in over 100 languages. It is different from previous speech models because it uses a universal encoder that is trained on data in many different languages. This allows Chirp to achieve state-of-the-art accuracy, even for languages with limited training data.

Chirp is ideal for developers who need to transcribe audio in multiple languages. It can be used for a variety of tasks, such as video captioning, content transcription, and speech recognition.
