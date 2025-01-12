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

# Gemini API: Context Caching Quickstart

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Caching.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook introduces context caching with the Gemini API and provides examples of interacting with the Apollo 11 transcript using the Python SDK. For a more comprehensive look, check out [the caching guide](https://ai.google.dev/gemini-api/docs/caching?lang=python).

### Install dependencies


```
!pip install -qU 'google-generativeai>=0.7.0'
```


```
import google.generativeai as genai
from google.generativeai import caching
```

### Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

genai.configure(api_key=userdata.get("GOOGLE_API_KEY"))
```

## Upload a file

A common pattern with the Gemini API is to ask a number of questions of the same document. Context caching is designed to assist with this case, and can be more efficient by avoiding the need to pass the same tokens through the model for each new request.

This example will be based on the transcript from the Apollo 11 mission.

Start by downloading that transcript.


```
!wget -q https://storage.googleapis.com/generativeai-downloads/data/a11.txt
!head a11.txt
```

    INTRODUCTION
    
    This is the transcription of the Technical Air-to-Ground Voice Transmission (GOSS NET 1) from the Apollo 11 mission.
    
    Communicators in the text may be identified according to the following list.
    
    Spacecraft:
    CDR	Commander	Neil A. Armstrong
    CMP	Command module pilot   	Michael Collins
    LMP	Lunar module pilot	Edwin E. ALdrin, Jr.
    

Now upload the transcript using the [File API](../quickstarts/File_API.ipynb).


```
document = genai.upload_file(path="a11.txt")
```

## Cache the prompt

Next create a [`CachedContent`](https://ai.google.dev/api/python/google/generativeai/protos/CachedContent) object specifying the prompt you want to use, including the file and other fields you wish to cache. In this example the [`system_instruction`](../quickstarts/System_instructions.ipynb) has been set, and the document was provided in the prompt.


```
# Note that caching requires a frozen model, e.g. one with a `-001` version suffix.
model_name = "gemini-1.5-flash-001"

apollo_cache = caching.CachedContent.create(
    model=model_name,
    system_instruction="You are an expert at analyzing transcripts.",
    contents=[document],
)

apollo_cache
```




    CachedContent(
        name='cachedContents/40k37vcojf2o',
        model='models/gemini-1.5-flash-001',
        display_name='',
        usage_metadata={
            'total_token_count': 323383,
        },
        create_time=2024-06-18 16:15:48.903792+00:00,
        update_time=2024-06-18 16:15:48.903792+00:00,
        expire_time=2024-06-18 17:15:48.514878+00:00
    )



## Manage the cache expiry

Once you have a `CachedContent` object, you can update the expiry time to keep it alive while you need it.


```
import datetime

apollo_cache.update(ttl=datetime.timedelta(hours=2))
apollo_cache
```




    CachedContent(
        name='cachedContents/40k37vcojf2o',
        model='models/gemini-1.5-flash-001',
        display_name='',
        usage_metadata={
            'total_token_count': 323383,
        },
        create_time=2024-06-18 16:15:48.903792+00:00,
        update_time=2024-06-18 16:15:49.983070+00:00,
        expire_time=2024-06-18 18:15:49.822943+00:00
    )



## Use the cache for generation

As the `CachedContent` object refers to a specific model and parameters, you must create a [`GenerativeModel`](https://ai.google.dev/api/python/google/generativeai/GenerativeModel) using [`from_cached_content`](https://ai.google.dev/api/python/google/generativeai/GenerativeModel#from_cached_content). Then, generate content as you would with a directly instantiated model object.


```
apollo_model = genai.GenerativeModel.from_cached_content(cached_content=apollo_cache)

response = apollo_model.generate_content("Find a lighthearted moment from this transcript")
print(response.text)
```

    A lighthearted moment occurs on page 31 of the transcript. The astronauts are discussing their view of Earth and Mission Control comments on how good the view must be from up there. 
    
    **CMP:**  I didn't know what I was looking at, but I sure did like it.
    **CC:** Okay. I guess the view must be pretty good from up there. We show you just roughly somewhere around 19,000 miles out now.
    **CMP:** I didn't have much outside my window.
    **CC:** We'll get you into the PTC one of these days, and you take turns looking. 
    
    This exchange is lighthearted because the astronauts are joking about the limited view from their individual windows and the Mission Control is playfully suggesting they share the view by switching positions.  This lighthearted moment shows the astronauts' sense of humor and camaraderie even during a stressful mission. 
    
    

You can inspect token usage through `usage_metadata`. Note that the cached prompt tokens are included in `prompt_token_count`, but excluded from the `total_token_count`.


```
response.usage_metadata
```




    prompt_token_count: 323392
    candidates_token_count: 189
    total_token_count: 198
    cached_content_token_count: 323383



You can ask new questions of the model, and the cache is reused.


```
chat = apollo_model.start_chat()
response = chat.send_message("Give me a quote from the most important part of the transcript.")
print(response.text)
```

    The most important part of the transcript is when Neil Armstrong steps onto the lunar surface.  Here's his famous quote: 
    
    "**That's one small step for (a) man, one giant leap for mankind.**" 
    
    


```
response = chat.send_message("What was recounted after that?")
print(response.text)
```

    After Neil Armstrong's famous quote, the transcript details his initial observations of the lunar surface:
    
    * **The surface is fine and powdery.**  He describes being able to pick up the dust with his toe, and how it sticks to his boots.
    * **There seems to be no difficulty in moving around.** He found it easier than the simulations on Earth, and that walking is comfortable.
    * **The descent engine did not leave a crater of any size.** He notes that the LM had about a foot of clearance from the surface.
    * **There is some evidence of rays emanating from the descent engine.**  This is an indication of the engine's impact on the lunar surface. 
    
    These initial observations provide the first detailed description of the lunar surface, marking a historic moment in human exploration. 
    
    


```
response.usage_metadata
```




    prompt_token_count: 323455
    candidates_token_count: 164
    total_token_count: 236
    cached_content_token_count: 323383



As you can see, among the 323455 tokens, 323383 were cached (and thus less expensive) and only 236 were from the prompt.

Since the cached tokens are cheaper than the normal ones, it means this prompt was 75% cheaper that if you had not used caching. Check the [pricing here](https://ai.google.dev/pricing) for the up-to-date discount on cached tokens.

## Counting tokens

The `GenerativeModel` object can be used to count the tokens of a request in the same manner as a direct, uncached, model.


```
apollo_model.count_tokens("How many people are involved in this transcript?")
```




    total_tokens: 9



## Delete the cache

The cache has a small recurring storage cost (cf. [pricing](https://ai.google.dev/pricing)) so by default it is only saved for an hour. In this case you even set it up for a shorter amont of time (using `"ttl"`) of 2h.

Still, if you don't need you cache anymore, it is good practice to delete it proactively.


```
print(apollo_cache.name)
apollo_cache.delete()
```

    cachedContents/40k37vcojf2o
    

## Next Steps
### Useful API references:

If you want to know more about the caching API, you can check the full [API specifications](https://ai.google.dev/api/rest/v1beta/cachedContents) and the [caching documentation](https://ai.google.dev/gemini-api/docs/caching).

### Continue your discovery of the Gemini API

Check the File API notebook to know more about that API. The [vision capabilities](../quickstarts/Video.ipynb) of the Gemini API are a good reason to use the File API and the caching. 
The Gemini API also has configurable [safety settings](../quickstarts/Safety.ipynb) that you might have to customize when dealing with big files.

