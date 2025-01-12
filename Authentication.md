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

# Gemini API: Authentication Quickstart

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

The Gemini API uses API keys for authentication. This notebook walks you through creating an API key, and using it with the Python SDK or a command line tool like `curl`.

## Create an API key

You can [create](https://aistudio.google.com/app/apikey) your API key using Google AI Studio with a single click.  

Remember to treat your API key like a password. Do not accidentally save it in a notebook or source file you later commit to GitHub. This notebook shows you two ways you can securely store your API key.

* If you are using Google Colab, it is recommended to store your key in Colab Secrets.

* If you are using a different development environment (or calling the Gemini API through `cURL` in your terminal), it is recommended to store your key in an environment variable.

Let's start with Colab Secrets.

## Add your key to Colab Secrets

Add your API key to the Colab Secrets manager to securely store it.

1. Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the left panel.
   
   <img src="https://storage.googleapis.com/generativeai-downloads/images/secrets.jpg" alt="The Secrets tab is found on the left panel." width=50%>

2. Create a new secret with the name `GOOGLE_API_KEY`.
3. Copy/paste your API key into the `Value` input box of `GOOGLE_API_KEY`.
4. Toggle the button on the left to allow notebook access to the secret.


## Install the Python SDK


```
!pip install -U -q "google-generativeai>=0.7.2"
```

## Configure the SDK with your API key

You'll call `genai.configure` with your API key, but instead of pasting your key into the notebook, you'll read it from Colab Secrets.


```
import google.generativeai as genai
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

And that's it! Now you're ready to call the Gemini API.


```
model = genai.GenerativeModel('models/gemini-1.5-flash')
response = model.generate_content("Please give me python code to sort a list.")
print(response.text)
```

    ```python
    def sort_list(list_to_sort):
      """
      Sorts a list using the built-in sort method.
    
      Args:
        list_to_sort: The list to be sorted.
    
      Returns:
        The sorted list.
      """
    
      list_to_sort.sort()
      return list_to_sort
    
    # Example usage:
    my_list = [5, 2, 8, 1, 9]
    
    sorted_list = sort_list(my_list)
    
    print(f"Original list: {my_list}")
    print(f"Sorted list: {sorted_list}")
    ```
    
    **Explanation:**
    
    * **`sort_list(list_to_sort)` function:**
        * Takes a list as input (`list_to_sort`).
        * Uses the built-in `sort()` method to sort the list in ascending order.
        * Returns the sorted list.
    
    * **Example usage:**
        * Creates a sample list `my_list`.
        * Calls `sort_list()` to sort the list and store the result in `sorted_list`.
        * Prints both the original and sorted lists.
    
    **Output:**
    
    ```
    Original list: [5, 2, 8, 1, 9]
    Sorted list: [1, 2, 5, 8, 9]
    ```
    
    **Other sorting methods:**
    
    * **`sorted()` function:** This function creates a new sorted list without modifying the original list.
    
    ```python
    sorted_list = sorted(my_list)
    ```
    
    * **Custom sorting:** You can use the `sort()` method with a `key` function to specify a custom sorting criteria.
    
    ```python
    my_list.sort(key=lambda x: -x) # Sort in descending order
    ```
    
    Choose the method that best suits your needs.
    
    

## Store your key in an environment variable

If you are using a different development environment (or calling the Gemini API through `cURL` in your terminal), it is recommended to store your key in an environment variable.

To store your key in an environment variable, open your terminal and run:

```export GOOGLE_API_KEY="YOUR_API_KEY"```

If you are using Python, add these two lines to your notebook to read the key:

```
import os
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
```

Or, if you're calling the API through your terminal using `cURL`, you can copy and paste this code to read your key from the environment variable.

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{
          "text": "Please give me Python code to sort a list."}]}]}'
```


## Learning more

The Gemini API uses API keys for most types of authentication, and that’s all you need to get started. You can use OAuth for more advanced authentication when tuning models. You can learn more about that in the [OAuth quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication_with_OAuth.ipynb).
