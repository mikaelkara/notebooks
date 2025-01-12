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

# Gemini API: Analyze a Video - Historic Event Recognition

This notebook shows how you can use Gemini models' multimodal capabilities to recognize which historic event is happening in the video.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Analyze_a_Video_Historic_Event_Recognition.ipynb"><img src = "../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import time
import google.generativeai as genai
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
api_key = userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=api_key)
```

## Example

This example uses [video of President Ronald Reagan's Speech at the Berlin Wall](https://s3.amazonaws.com/NARAprodstorage/opastorage/live/16/147/6014716/content/presidential-libraries/reagan/5730544/6-12-1987-439.mp4) taken on June 12 1987.


```
# Download video
path = "berlin.mp4"
url = "https://s3.amazonaws.com/NARAprodstorage/opastorage/live/16/147/6014716/content/presidential-libraries/reagan/5730544/6-12-1987-439.mp4"
!wget $url -O $path
```

    --2024-08-15 17:25:44--  https://s3.amazonaws.com/NARAprodstorage/opastorage/live/16/147/6014716/content/presidential-libraries/reagan/5730544/6-12-1987-439.mp4
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.170.208, 52.217.124.192, 52.216.54.104, ...
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.170.208|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 628645171 (600M) [video/mp4]
    Saving to: ‘berlin.mp4’
    
    berlin.mp4          100%[===================>] 599.52M  25.5MB/s    in 18s     
    
    2024-08-15 17:26:02 (33.9 MB/s) - ‘berlin.mp4’ saved [628645171/628645171]
    
    


```
# Upload video
video_file = genai.upload_file(path=path)
```


```
# Wait until the uploaded video is available
while video_file.state.name == "PROCESSING":
  print('.', end='')
  time.sleep(5)
  video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)
```

    ..............

The uploaded video is ready for processing. This prompt instructs the model to provide basic information about the historical events portrayed in the video.


```
system_prompt = """
You are historian who specializes in events caught on film.
When you receive a video answer following questions:
When did it happen?
Who is the most important person in video?
How the event is called?
"""
```

Some historic events touch on controversial topics that may get flagged by Gemini API, which blocks the response for the query.

Because of this, it might be a good idea to turn off safety settings.


```
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
```


```
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash", safety_settings=safety_settings,
                              system_instruction=system_prompt)
response = model.generate_content([video_file])
print(response.text)
```

    This is Ronald Reagan's speech at the Brandenburg Gate in West Berlin, Germany. It happened on June 12, 1987.  The most important person in this video is Ronald Reagan, the 40th president of the United States,  as he gave a landmark speech demanding the fall of the Berlin Wall. The event is called "Tear Down This Wall" speech. 
    
    

As you can see, the model correctly provided information about the dates, Ronald Reagan, who was the main subject of the video, and the name of this event.

You can delete the video to prevent unnecessary data storage.


```
# Delete video
genai.delete_file(video_file.name)
```

## Summary

Now you know how you can prompt Gemini models with videos and use them to recognize historic events.

This notebook shows only one of many use cases. Try thinking of more yourself or see other notebooks utilizing Gemini API with videos.
