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

# Gemini API: Analyze a Video - Classification

This notebook uses multimodal capabilities of the Gemini model to classify the species of animals shown in a video.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Analyze_a_Video_Classification.ipynb"><img src = "../images/colab_logo_32px.png"/>Run in Google Colab</a>
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
# This is only needed to see the first frame of the video for demonstration purposes
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

This example uses a [video](https://commons.wikimedia.org/wiki/File:American_black_bears_(Ursus_americanus).webm) published by Bryon Evans containing an American black bear.

The video falls under the [Creative Commons Attribution 3.0 Unported license](https://creativecommons.org/licenses/by/3.0/deed.en).


```
# Download video
path = "black_bear.webm"
url = "https://upload.wikimedia.org/wikipedia/commons/8/81/American_black_bears_%28Ursus_americanus%29.webm"
!wget $url -O $path
```

    --2024-08-15 17:13:19--  https://upload.wikimedia.org/wikipedia/commons/8/81/American_black_bears_%28Ursus_americanus%29.webm
    Resolving upload.wikimedia.org (upload.wikimedia.org)... 208.80.153.240, 2620:0:860:ed1a::2:b
    Connecting to upload.wikimedia.org (upload.wikimedia.org)|208.80.153.240|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 45046409 (43M) [video/webm]
    Saving to: ‘black_bear.webm’
    
    black_bear.webm     100%[===================>]  42.96M  25.5MB/s    in 1.7s    
    
    2024-08-15 17:13:21 (25.5 MB/s) - ‘black_bear.webm’ saved [45046409/45046409]
    
    


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

    ..

To demonstrate the video content, display the first frame:


```
cap = cv2.VideoCapture(path)
_, frame = cap.read()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display using matplotlib
plt.imshow(frame_rgb)
plt.axis('off')
plt.show()

# close video file
cap.release()
```


    
![png](output_14_0.png)
    


The uploaded video is ready to be analyzed. The constructed prompt instructs the model to classify animals in the video. In addition to providing both their English and Latin names.


```
system_prompt = """
You are a zoologist whose job is to name animals in videos.
You should always provide an english and latin name.
"""
```


```
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash", system_instruction=system_prompt)
response = model.generate_content([video_file])
print(response.text)
```

    Black Bear, *Ursus americanus* 
    
    

As you can see, the model accurately named the animal and provided a correct Latin name. You can delete the video to prevent unnecessary data storage.


```
# Delete video
genai.delete_file(video_file.name)
```

## Summary

Now you know how you can prompt Gemini models with videos and use them to classify species of animals.

This notebook shows only one of many use cases. Try thinking of more yourself or see other notebooks utilizing Gemini API with videos.
