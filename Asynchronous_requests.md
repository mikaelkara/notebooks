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

# Gemini API: Asynchronous Python requests

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Asynchronous_requests.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>


This notebook will show you how to make asynchronous and parallel requests using the Gemini API's Python SDK and Python 3's [`asyncio`](https://docs.python.org/3/library/asyncio.html) standard library.

The examples here run in Google Colab and use the implicit event loop supplied in Colab. You can also run these commands interactively using the `asyncio` REPL (invoked with `python -m asyncio`), or you can manage the [event loop](https://docs.python.org/3/library/asyncio-eventloop.html) yourself.


```
!pip install -qU 'google-generativeai>=0.8.3' aiohttp
```


```
import aiohttp
import asyncio
import io
import PIL
import google.generativeai as genai
```


```
# This notebook should work fine in a normal Python environment, but due to https://github.com/google-gemini/generative-ai-python/issues/499
# this workaround is needed in Colab, effectively un-monkey-patching a Colab patch.
genai.configure = getattr(genai.configure, "func", genai.configure)
```

## Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
```

## Using local files

This simple example shows how can you use local files (presumed to load quickly) with the SDK's `async` API.


```
model = genai.GenerativeModel("gemini-1.5-flash-latest")

prompt = "Describe this image in just 3 words."

img_filenames = ["firefighter.jpg", "elephants.jpeg", "jetpack.jpg"]
img_dir = "https://storage.googleapis.com/generativeai-downloads/images/"
```

Start by downloading the files locally.


```
!wget -nv {img_dir}{{{','.join(img_filenames)}}}
```

    2024-10-18 01:00:43 URL:https://storage.googleapis.com/generativeai-downloads/images/firefighter.jpg [547369/547369] -> "firefighter.jpg.2" [1]
    2024-10-18 01:00:43 URL:https://storage.googleapis.com/generativeai-downloads/images/elephants.jpeg [224007/224007] -> "elephants.jpeg.2" [1]
    2024-10-18 01:00:43 URL:https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg [357568/357568] -> "jetpack.jpg.2" [1]
    FINISHED --2024-10-18 01:00:43--
    Total wall clock time: 0.1s
    Downloaded: 3 files, 1.1M in 0.01s (83.7 MB/s)
    

The async code uses the `generate_content_async` method to invoke the API. Most API methods have an `_async` variant that provides this functionality.

Note that this code is not run in parallel. The async call indicates that the event loop *can* yield to other tasks, but there are no other tasks scheduled in this code. This may be sufficient, e.g. if you are running this in a web server request handler as it will allow the handler to yield to other tasks while waiting for the API response.


```
async def describe_local_images():

  for img_filename in img_filenames:

    img = PIL.Image.open(img_filename)
    r = await model.generate_content_async([prompt, img])
    print(r.text)


await describe_local_images()
```

    Cat in a tree. 
    
    Elephants in grass. 
    
    Jetpack Backpack 
    
    

## Downloading images asynchronously and in parallel

This example shows a more real-world case where an image is downloaded from an external source using the async HTTP library [`aiohttp`](https://pypi.org/project/aiohttp), and each image is processed in parallel.


```
async def download_image(session: aiohttp.ClientSession, img_url: str) -> PIL.Image:
  """Returns a PIL.Image object from the provided URL."""
  async with session.get(img_url) as img_resp:
    buffer = io.BytesIO()
    buffer.write(await img_resp.read())
    return PIL.Image.open(buffer)


async def process_image(img_future: asyncio.Future[PIL.Image]) -> str:
  """Summarise the image using the Gemini API."""
  # This code uses a future so that it defers work as late as possible. Using a
  # concrete Image object would require awaiting the download task before *queueing*
  # this content generation task - this approach chains the futures together
  # so that the download only starts when the generation is scheduled.
  r = await model.generate_content_async([prompt, await img_future])
  return r.text
```


```
async def download_and_describe():

  async with aiohttp.ClientSession() as sesh:
    response_futures = []
    for img_filename in img_filenames:

      # Create the image download tasks (this does not schedule them yet).
      img_future = download_image(sesh, img_dir + img_filename)

      # Kick off the Gemini API request using the pending image download tasks.
      text_future = process_image(img_future)

      # Save the reference so they can be processed as they complete.
      response_futures.append(text_future)

    print(f"Download and content generation queued for {len(response_futures)} images.")

    # Process responses as they complete (may be a different order). The tasks are started here.
    for response in asyncio.as_completed(response_futures):
      print()
      print(await response)


await download_and_describe()
```

    Download and content generation queued for 3 images.
    
    Cat in tree. 
    
    
    Elephant Family Grass
    
    Jetpack Backpack 
    
    

In the above example, a coroutine is created for each image that both downloads and then summarizes the image. The coroutines are executed in the final step, in the `as_completed` loop. To start them as early as possible without blocking the other work, you could wrap `download_image` in [`asyncio.ensure_future`](https://docs.python.org/3/library/asyncio-future.html#asyncio.ensure_future), but for this example the execution has been deferred to keep the creation and execution concerns separate.

## Next Steps

* Check out the `*_async` methods on the [`GenerativeModel`](https://github.com/google-gemini/generative-ai-python/blob/main/docs/api/google/generativeai/GenerativeModel.md) class in the Python SDK reference.
* Read more on Python's [`asyncio`](https://docs.python.org/3/library/asyncio.html) library
