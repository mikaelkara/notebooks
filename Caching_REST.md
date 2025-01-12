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

# Gemini API: Caching Quickstart with REST

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/rest/Caching_REST.ipynb"><img src="../../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


This notebook introduces [context caching](https://ai.google.dev/gemini-api/docs/caching?lang=rest) with the Gemini API and provides examples of interacting with the Apollo 11 transcript using the Python SDK. Context caching is a way to save on requests costs when a substantial initial context is referenced repeatedly by shorter requests. It will use `curl` commands to call the methods in the REST API. 

For a more comprehensive look, check out [the caching guide](https://ai.google.dev/gemini-api/docs/caching?lang=rest).

This notebook contains `curl` commands you can run in Google Colab, or copy to your terminal. If you have never used the Gemini REST API, it is strongly recommended to start with the [Prompting quickstart](../../quickstarts/rest/Prompting_REST.ipynb) first.

### Authentication

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](../../quickstarts/Authentication.ipynb) to learn more.


```
import os
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Caching content

Let's start by getting the transcript from the Apollo 11 mission.


```
!wget https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

    --2024-07-11 17:55:31--  https://storage.googleapis.com/generativeai-downloads/data/a11.txt
    Resolving storage.googleapis.com (storage.googleapis.com)... 74.125.199.207, 74.125.20.207, 108.177.98.207, ...
    Connecting to storage.googleapis.com (storage.googleapis.com)|74.125.199.207|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 847790 (828K) [text/plain]
    Saving to: ‘a11.txt.2’
    
    a11.txt.2             0%[                    ]       0  --.-KB/s               a11.txt.2           100%[===================>] 827.92K  --.-KB/s    in 0.008s  
    
    2024-07-11 17:55:31 (103 MB/s) - ‘a11.txt.2’ saved [847790/847790]
    
    

Now you need to reencode it to base-64, so let's prepare the whole [cachedContent](https://ai.google.dev/api/rest/v1beta/cachedContents#resource:-cachedcontent) while you're at it.

Note that you have to use a stable model with fixed versions (`gemini-1.5-flash-001` in this case). That's why you had to add a version postfix ("`-001`" in this case). You'll get a `Model [...] not found or does not support caching` error if you forget to do so.



```bash
%%bash

echo '{
  "model": "models/gemini-1.5-flash-001",
  "contents":[
    {
      "parts":[
        {
          "inline_data": {
            "mime_type":"text/plain",
            "data": "'$(base64 -w0 a11.txt)'"
          }
        }
      ],
    "role": "user"
    }
  ],
  "systemInstruction": {
    "parts": [
      {
        "text": "You are an expert at analyzing transcripts."
      }
    ]
  },
  "ttl": "600s"
}' > request.json
```

We can now create the cached content.


```bash
%%bash

curl -X POST "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$GOOGLE_API_KEY" \
 -H 'Content-Type: application/json' \
 -d @request.json > cache.json

 cat cache.json
```

    {
      "name": "cachedContents/lf0nt062ulc1",
      "model": "models/gemini-1.5-flash-001",
      "createTime": "2024-07-11T18:02:48.257891Z",
      "updateTime": "2024-07-11T18:02:48.257891Z",
      "expireTime": "2024-07-11T18:07:47.635193373Z",
      "displayName": "",
      "usageMetadata": {
        "totalTokenCount": 323383
      }
    }
    

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100 1104k    0     0  100 1104k      0   902k  0:00:01  0:00:01 --:--:--  902k100 1104k    0   307  100 1104k    138   499k  0:00:02  0:00:02 --:--:--  500k100 1104k    0   307  100 1104k    138   499k  0:00:02  0:00:02 --:--:--  499k
    

You will need it for the next commands so save the name of the cache.

You're using a text file to save the name here beacuse of colab constrainsts but you could also simply use a variable.


```bash
%%bash

CACHE_NAME=$(cat cache.json | grep '"name":' | cut -d '"' -f 4 | head -n 1)

echo $CACHE_NAME > cache_name.txt

cat cache_name.txt
```

    cachedContents/qidqwuaxdqz4
    

## Listing caches
Since caches have a reccuring cost it's a good idea to keep an eye on them. It can also be useful if you need to find their name.


```
!curl "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=$GOOGLE_API_KEY"
```

    {
      "cachedContents": [
        {
          "name": "cachedContents/lf0nt062ulc1",
          "model": "models/gemini-1.5-flash-001",
          "createTime": "2024-07-11T18:02:48.257891Z",
          "updateTime": "2024-07-11T18:02:48.257891Z",
          "expireTime": "2024-07-11T18:07:47.635193373Z",
          "displayName": "",
          "usageMetadata": {
            "totalTokenCount": 323383
          }
        },
        {
          "name": "cachedContents/qidqwuaxdqz4",
          "model": "models/gemini-1.5-flash-001",
          "createTime": "2024-07-11T18:02:30.516233Z",
          "updateTime": "2024-07-11T18:02:30.516233Z",
          "expireTime": "2024-07-11T18:07:29.803448998Z",
          "displayName": "",
          "usageMetadata": {
            "totalTokenCount": 323383
          }
        }
      ]
    }
    

## Using cached content when prompting

Prompting using cached content is the same as what is illustrated in the [Prompting quickstart](../../quickstarts/rest/Prompting_REST.ipynb) except you're adding a `"cachedContent"` value which is the name of the cache you saved earlier.


```bash
%%bash

curl -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-001:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
      "contents": [
        {
          "parts":[{
            "text": "Please summarize this transcript"
          }],
          "role": "user"
        },
      ],
      "cachedContent": "'$(cat cache_name.txt)'"
    }'
```

    {
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "This is a transcript of the air-to-ground voice transmission between the Apollo 11 crew and Mission Control during their journey to the moon and back. The transcript covers the period from the launch of the Saturn V rocket until the splashdown of the command module in the Pacific Ocean. \n\nThe transcript documents a fascinating conversation between the astronauts and Mission Control, highlighting the various activities of the mission. It covers:\n\n* **Launch and Trans-Lunar Injection:** The launch sequence, staging of the rocket, and the critical Trans-Lunar Injection (TLI) maneuver that sent the Apollo 11 spacecraft towards the moon.\n* **Orbit and Docking:** The crew's actions in earth orbit, their successful docking with the Lunar Module (LM), and the transfer of the LM to the docking port of the command module. \n* **Lunar Surface Preparations:** Communications checks, pre-flight preparations for the Lunar Module, and the successful ejection of the LM from the command module. \n* **Lunar Orbit Insertion:** The burn that sent the spacecraft into lunar orbit and the subsequent activities in lunar orbit. \n* **Lunar Landing and EVA:** The descent of the LM to the surface of the moon, Neil Armstrong’s famous first steps, and the Lunar surface activities. \n* **Lunar Ascent:** The launch of the LM back into lunar orbit and the docking with the command module. \n* **Trans-Earth Injection:** The burn that sent the spacecraft on its return journey to Earth. \n* **Earth Entry and Splashdown:** The re-entry of the command module into the Earth’s atmosphere, the deployment of parachutes, and the splashdown in the Pacific Ocean.\n\nThe transcript provides valuable insight into the complexity and meticulous planning that went into the Apollo 11 mission. It showcases the close communication and coordination between the crew and Mission Control, and the dedication of the many individuals who made this historic mission possible. \n"
              }
            ],
            "role": "model"
          },
          "finishReason": "STOP",
          "index": 0,
          "safetyRatings": [
            {
              "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HATE_SPEECH",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_HARASSMENT",
              "probability": "NEGLIGIBLE"
            },
            {
              "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
              "probability": "NEGLIGIBLE"
            }
          ]
        }
      ],
      "usageMetadata": {
        "promptTokenCount": 323388,
        "candidatesTokenCount": 397,
        "totalTokenCount": 323785,
        "cachedContentTokenCount": 323383
      }
    }
    

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100   225    0     0  100   225      0   1107 --:--:-- --:--:-- --:--:--  1102100   225    0     0  100   225      0    186  0:00:01  0:00:01 --:--:--   186100   225    0     0  100   225      0    102  0:00:02  0:00:02 --:--:--   101100   225    0     0  100   225      0     70  0:00:03  0:00:03 --:--:--    70100   225    0     0  100   225      0     53  0:00:04  0:00:04 --:--:--    53100   225    0     0  100   225      0     43  0:00:05  0:00:05 --:--:--     0100   225    0     0  100   225      0     36  0:00:06  0:00:06 --:--:--     0100   225    0     0  100   225      0     31  0:00:07  0:00:07 --:--:--     0100   225    0     0  100   225      0     27  0:00:08  0:00:08 --:--:--     0100   225    0     0  100   225      0     24  0:00:09  0:00:09 --:--:--     0100   225    0     0  100   225      0     22  0:00:10  0:00:10 --:--:--     0100   225    0     0  100   225      0     20  0:00:11  0:00:11 --:--:--     0100   225    0     0  100   225      0     18  0:00:12  0:00:12 --:--:--     0100   225    0     0  100   225      0     17  0:00:13  0:00:13 --:--:--     0100   225    0     0  100   225      0     15  0:00:15  0:00:14  0:00:01     0100   225    0     0  100   225      0     14  0:00:16  0:00:15  0:00:01     0100   225    0     0  100   225      0     13  0:00:17  0:00:16  0:00:01     0100   225    0     0  100   225      0     13  0:00:17  0:00:17 --:--:--     0100   225    0     0  100   225      0     12  0:00:18  0:00:18 --:--:--     0100   225    0     0  100   225      0     11  0:00:20  0:00:19  0:00:01     0100   225    0     0  100   225      0     11  0:00:20  0:00:20 --:--:--     0100   225    0     0  100   225      0     10  0:00:22  0:00:21  0:00:01     0100   225    0     0  100   225      0     10  0:00:22  0:00:22 --:--:--     0100   225    0     0  100   225      0      9  0:00:25  0:00:23  0:00:02     0100  3042    0  2817  100   225    117      9  0:00:25  0:00:24  0:00:01   580
    

As you can see, among the 323699 tokens, 323383 were cached (and thus less expensive) and only 311 were from the prompt.

Since the cached tokens are cheaper than the normal ones, it means this prompt was 75% cheaper that if you had not used caching. Check the [pricing here](https://ai.google.dev/pricing) for the up-to-date discount on cached tokens.

## Optional: Updating a cache
If you need to update a cache, to chance its content, or just extend its longevity, just use `PATCH`:


```bash
%%bash

curl -X PATCH "https://generativelanguage.googleapis.com/v1beta/$(cat cache_name.txt)?key=$GOOGLE_API_KEY" \
 -H 'Content-Type: application/json' \
 -d '{"ttl": "300s"}'
```

    {
      "name": "cachedContents/qidqwuaxdqz4",
      "model": "models/gemini-1.5-flash-001",
      "createTime": "2024-07-11T18:02:30.516233Z",
      "updateTime": "2024-07-11T18:05:38.781423Z",
      "expireTime": "2024-07-11T18:10:38.759996261Z",
      "displayName": "",
      "usageMetadata": {
        "totalTokenCount": 323383
      }
    }
    

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0100   322    0   307  100    15    822     40 --:--:-- --:--:-- --:--:--   863
    

## Deleting cached content

The cache has a small recurring storage cost (cf. [pricing](https://ai.google.dev/pricing)) so by default it is only saved for an hour. In this case you even set it up for a shorter amont of time (using `"ttl"`) of 10mn.

Still, if you don't need you cache anymore, it is good practice to delete it proactively.


```
!curl -X DELETE "https://generativelanguage.googleapis.com/v1beta/$(cat cache_name.txt)?key=$GOOGLE_API_KEY"
```

    {}
    

## Next Steps
### Useful API references:

If you want to know more about the caching REST APIs, you can check the full [API specifications](https://ai.google.dev/api/rest/v1beta/cachedContents) and the [caching documentation](https://ai.google.dev/gemini-api/docs/caching).

### Continue your discovery of the Gemini API

Check the File API notebook to know more about that API. The [vision capabilities](../../quickstarts/rest/Video_REST.ipynb) of the Gemini API are a good reason to use the File API and the caching. 
The Gemini API also has configurable [safety settings](../../quickstarts/rest/Safety_REST.ipynb) that you might have to customize when dealing with big files.

