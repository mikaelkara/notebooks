##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Error handling

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Error_handling.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This Colab notebook demonstrates strategies for handling common errors you might encounter when working with the Gemini API:

*   **Transient Errors:** Temporary failures due to network issues, server overload, etc.
*   **Rate Limits:** Restrictions on the number of requests you can make within a certain timeframe.
*   **Timeouts:** When an API call takes too long to complete.

You have two main approaches to explore:

1.  **Automatic retries:** A simple way to retry requests when they fail due to transient errors.
2.  **Manual backoff and retry:** A more customizable approach that provides finer control over retry behavior.


**Gemini Rate Limits**

The default rate limits for different Gemini models are outlined in the [Gemini API model documentation](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations). If your application requires a higher quota, consider [requesting a rate limit increase](https://ai.google.dev/gemini-api/docs/quota).


```
!pip install -U "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai
```

### Setup your API key

To run the following cells, store your API key in a Colab Secret named `GOOGLE_API_KEY`. If you don't have an API key or need help creating a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) guide.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
```

### Automatic retries

The Gemini API's client library offers built-in retry mechanisms for handling transient errors. You can enable this feature by using the `request_options` argument with API calls like `generate_content`, `generate_answer`, `embed_content`, and `generate_content_async`.

**Advantages:**

* **Simplicity:** Requires minimal code changes for significant reliability gains.
* **Robust:** Effectively addresses most transient errors without additional logic.

**Customize retry behavior:**

Use these settings in [`retry`](https://googleapis.dev/python/google-api-core/latest/retry.html) to customize retry behavior:

* `predicate`:  (callable) Determines if an exception is retryable. Default: [`if_transient_error`](https://github.com/googleapis/python-api-core/blob/main/google/api_core/retry/retry_base.py#L75C4-L75C13)
* `initial`: (float) Initial delay in seconds before the first retry. Default: `1.0`
* `maximum`: (float) Maximum delay in seconds between retries. Default: `60.0`
* `multiplier`: (float) Factor by which the delay increases after each retry. Default: `2.0`
* `timeout`: (float) Total retry duration in seconds. Default: `120.0`


```
from google.api_core import retry

model = genai.GenerativeModel("gemini-1.5-flash-latest")
prompt = "Write a story about a magic backpack."

model.generate_content(
    prompt, request_options={"retry": retry.Retry(predicate=retry.if_transient_error)}
)
```

### Manually increase timeout when responses take time

If you encounter `ReadTimeout` or `DeadlineExceeded` errors, meaning an API call exceeds the default timeout (600 seconds), you can manually adjust it by defining `timeout` in the `request_options` argument.


```
model = genai.GenerativeModel("gemini-1.5-flash-latest")
prompt = "Write a story about a magic backpack."

model.generate_content(
    prompt, request_options={"timeout": 900}
)  # Increase timeout to 15 minutes
```

**Caution:**  While increasing timeouts can be helpful, be mindful of setting them too high, as this can delay error detection and potentially waste resources.

### Manually implement backoff and retry with error handling

For finer control over retry behavior and error handling, you can use the [`retry`](https://googleapis.dev/python/google-api-core/latest/retry.html) library (or similar libraries like [`backoff`](https://pypi.org/project/backoff/) and [`tenacity`](https://tenacity.readthedocs.io/en/latest/)). This gives you precise control over retry strategies and allows you to handle specific types of errors differently.


```
from google.api_core import retry, exceptions

model = genai.GenerativeModel("gemini-1.5-flash-latest")


@retry.Retry(
    predicate=retry.if_transient_error,
    initial=2.0,
    maximum=64.0,
    multiplier=2.0,
    timeout=600,
)
def generate_with_retry(model, prompt):
    response = model.generate_content(prompt)
    return response


prompt = "Write a one-liner advertisement for magic backpack."

generate_with_retry(model=model, prompt=prompt)
```

### Test the error handling with retry mechanism

To validate that your error handling and retry mechanism work as intended, define a `generate_content` function that deliberately raises a `ServiceUnavailable` error on the first call. This setup will help you ensure that the retry decorator successfully handles the transient error and retries the operation.


```
from google.api_core import retry, exceptions


@retry.Retry(
    predicate=retry.if_transient_error,
    initial=2.0,
    maximum=64.0,
    multiplier=2.0,
    timeout=600,
)
def generate_content_first_fail(model, prompt):
    if not hasattr(generate_content_first_fail, "call_counter"):
        generate_content_first_fail.call_counter = 0

    generate_content_first_fail.call_counter += 1

    try:
        if generate_content_first_fail.call_counter == 1:
            raise exceptions.ServiceUnavailable("Service Unavailable")

        response = model.generate_content(prompt)
        return response.text
    except exceptions.ServiceUnavailable as e:
        print(f"Error: {e}")
        raise


model = genai.GenerativeModel("gemini-1.5-flash-latest")
prompt = "Write a one-liner advertisement for magic backpack."

generate_content_first_fail(model=model, prompt=prompt)
```
