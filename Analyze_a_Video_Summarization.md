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

# Gemini API: Analyze a Video - Summarization

This notebook shows how you can use Gemini API's multimodal capabilities for video summarization.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Analyze_a_Video_Summarization.ipynb"><img src = "../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import time
import google.generativeai as genai
```


```
# this is only needed for demonstration purposes
import cv2
import matplotlib.pyplot as plt
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
api_key = userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=api_key)
```

## Example
This notebook will use [Wing It!](https://studio.blender.org/films/wing-it/) movie directed by Rik Schutte, wills falls under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/deed.en).

See the full [credits](https://studio.blender.org/films/wing-it/pages/credits/) for all of the other people involved in its creation.


```
# Download video
path = "wingit.webm"
url = "https://upload.wikimedia.org/wikipedia/commons/3/38/WING_IT%21_-_Blender_Open_Movie-full_movie.webm"
!wget $url -O $path
```

    --2024-08-15 19:23:47--  https://upload.wikimedia.org/wikipedia/commons/3/38/WING_IT%21_-_Blender_Open_Movie-full_movie.webm
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 208.80.154.240, 2620:0:861:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|208.80.154.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 36196718 (35M) [video/webm]
    Saving to: ‘wingit.webm’
    
    wingit.webm         100%[===================>]  34.52M  28.1MB/s    in 1.2s    
    
    2024-08-15 19:23:49 (28.1 MB/s) - ‘wingit.webm’ saved [36196718/36196718]
    
    


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

    ...


```
# Display some of the video content
cap = cv2.VideoCapture(path)
frame_number = 1000
for _ in range(frame_number):
    ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

plt.imshow(frame_rgb)
plt.axis('off')
plt.show()

cap.release()
```


    
![png](output_13_0.png)
    


The video is now ready to be summarized by the model.


```
system_prompt = "You should provide a quick 2 or 3 sentence summary of what is happening in the video."
```


```
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash", system_instruction=system_prompt)
response = model.generate_content([video_file])
print(response.text)
```

    A cat and a dog attempt to fly a rocket made out of random junk they found in a barn. They make it into the air but run into a few problems and crash back into the hay pile where they started. 
    
    

The model correctly describes the plot of the short movie.

Now, you can delete the no longer necessary uploaded file.


```
# delete video
genai.delete_file(video_file.name)
```

### Important Note

Gemini API takes only one frame per second of the video. It may cause models not to see everything that is happening, especially if something is visible only for a fraction of a second.

## Summary

Now you know how you can use Gemini models to summarize what is happening in videos.

This notebook shows only one of many use cases of Gemini API's multimodal capabilities. Try thinking of more yourself or see other notebooks utilizing Gemini API with videos.
