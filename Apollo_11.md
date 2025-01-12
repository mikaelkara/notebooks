##### Copyright 2024 Google LLC.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
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

# Prompting with an Apollo 11 transcript

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Apollo_11.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook provides a quick example of how to prompt Gemini 1.5 Pro using a text file. In this case, you'll use a 400 page transcript from [Apollo 11](https://www.nasa.gov/history/alsj/a11/a11trans.html).


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai
```

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

Download the transcript.


```
!wget https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

    --2024-05-31 19:45:08--  https://storage.googleapis.com/generativeai-downloads/data/a11.txt
    Resolving storage.googleapis.com (storage.googleapis.com)... 142.250.99.207, 142.250.107.207, 173.194.202.207, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|142.250.99.207|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 847790 (828K) [text/plain]
    Saving to: ‘a11.txt’
    
    a11.txt             100%[===================>] 827.92K  --.-KB/s    in 0.01s   
    
    2024-05-31 19:45:08 (58.7 MB/s) - ‘a11.txt’ saved [847790/847790]
    
    

Prepare it for use in a prompt.


```
text_file_name = "a11.txt"
print(f"Uploading file...")
text_file = genai.upload_file(path=text_file_name)
print(f"Completed upload: {text_file.uri}")
```

    Uploading file...
    Completed upload: https://generativelanguage.googleapis.com/v1beta/files/75sa0bj4c0zn
    

## Generate Content

After the file has been uploaded, you can make `GenerateContent` requests that reference the File API URI. Then you will ask the model to find a few lighthearted moments.


```
prompt = "Find four lighthearted moments in this text file."

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

response = model.generate_content([prompt, text_file],
                                  request_options={"timeout": 600})
print(response.text)
```

    Here are four lighthearted moments from the text:
    
    1. **00 00 05 35 CDR:** "You sure sound clear down there, Bruce. Sounds like you're sitting in your living room." This playful comment shows Armstrong's sense of humor and the camaraderie between the crew and CAP COMM.
    
    2. **00 00 54 13 CMP:** "And tell Glenn Parker down at the Cape that he lucked out." This is a funny jab at Glenn Parker, indicating a playful rivalry between the crew and someone on the ground.
    
    3. **00 01 29 27 LMP:** "Cecil B. deAldrin is standing by for instructions." Aldrin's self-deprecating humor makes a lighthearted moment out of a technical procedure.
    
    4. **00 02 53 03 CDR:** "Hey, Houston, Apollo 11. That Saturn gave us a magnificent ride." Armstrong's enthusiasm and appreciation for the smooth launch are conveyed in a lighthearted way. 
    
    

## Delete File

Files are automatically deleted after 2 days or you can manually delete them using `files.delete()`.


```
genai.delete_file(text_file.name)
```

## Learning more

The File API accepts files under 2GB in size and can store up to 20GB of files per project. Learn more about the [File API](https://github.com/google-gemini/cookbook/blob/main/quickstarts/File_API.ipynb) here.
