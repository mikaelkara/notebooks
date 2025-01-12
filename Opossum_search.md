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

# Opossum search

<table align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Opossum_search.ipynb"><img src="../images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
</table>

This notebook contains a simple example of generating code with the Gemini API and Gemini 1.5 Pro. Just for fun, you'll prompt the model to create a web app called "Opossum Search" that searches Google with "opossum" appended to the query.

<img src="https://storage.googleapis.com/generativeai-downloads/images/opossum_search.jpg" alt="An image of the opossum search web app running in a browser" width="500"/>

> The opossum image above is from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Opossum_2.jpg), and shared under a CC BY-SA 2.5 license.


```
!pip install -q -U "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai
```

## Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/gemini-api-cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

Prompt the model to generate the web app.


```
instruction = """You are a coding expert that specializes in creating web pages based on a user request.
You create correct and simple code that is easy to understand.
You implement all the functionality requested by the user.
You ensure your code works properly, and you follow best practices for HTML programming."""
```


```
prompt = """Create a web app called Opossum Search:
1. Every time you make a search query, it should redirect you to a Google search
with the same query, but with the word opossum before it.
2. It should be visually similar to Google search.
3. Instead of the google logo, it should have a picture of this opossum: https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Opossum_2.jpg/292px-Opossum_2.jpg.
4. It should be a single HTML file, with no separate JS or CSS files.
5. It should say Powered by opossum search in the footer.
6. Do not use any unicode characters.
Thank you!"""
```


```
model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=instruction)
response = model.generate_content(prompt)
print(response.text)
```

    ```html
    <!DOCTYPE html>
    <html>
    <head>
      <title>Opossum Search</title>
      <style>
        body {
          font-family: sans-serif;
        }
        .container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
        }
        .logo {
          width: 292px;
          height: auto;
          margin-bottom: 20px;
        }
        .search-bar {
          display: flex;
          width: 600px;
          height: 40px;
          border: 1px solid #ccc;
          border-radius: 20px;
          padding: 5px;
        }
        .search-bar input {
          flex: 1;
          border: none;
          outline: none;
          padding: 5px;
          font-size: 16px;
        }
        .search-bar button {
          background-color: #4285f4;
          color: white;
          border: none;
          border-radius: 20px;
          padding: 10px 15px;
          cursor: pointer;
        }
        .footer {
          margin-top: 20px;
          text-align: center;
          font-size: 12px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Opossum_2.jpg/292px-Opossum_2.jpg" alt="Opossum" class="logo">
        <form action="https://www.google.com/search" method="get">
          <div class="search-bar">
            <input type="text" name="q" placeholder="Search the web" required>
            <button type="submit">Search</button>
          </div>
          <input type="hidden" name="q" value="opossum ">
        </form>
      </div>
      <div class="footer">Powered by opossum search</div>
    </body>
    </html>
    ```
    
    This HTML code creates a simple search page with the following features:
    
    *   **Opossum Logo:** Uses the provided image URL as the logo.
    *   **Search Bar:** Allows users to enter their search query.
    *   **Search Button:** Submits the form to Google with the "opossum" prefix.
    *   **Footer:**  Includes the "Powered by opossum search" text.
    
    **Explanation:**
    
    *   The code uses basic HTML tags for structuring the page.
    *   The `style` tag in the `<head>` contains inline CSS for basic styling.
    *   The form is set to target Google Search (`https://www.google.com/search`).
    *   The `<input type="hidden">` field automatically adds "opossum" before the user's search query.
    *   The footer is styled to be centered and small.
    
    This code will provide a basic web app that functions as an "Opossum Search" with the required features.
    
    

## Run the output locally

You can start a web server as follows.

* Save the HTML output to a file called `search.html`
* In your terminal run `python3 -m http.server 8000`
* Open your web browser, and point it to `http://localhost:8000/search.html`

## Display the output in IPython

Like all LLMs, the output may not always be correct. You can experiment by rerunning the prompt, or by writing an improved one (and/or better system instructions). Have fun!


```
import IPython
code = response.text.split('```')[1][len('html'):]
IPython.display.HTML(code)
```





<!DOCTYPE html>
<html>
<head>
  <title>Opossum Search</title>
  <style>
    body {
      font-family: sans-serif;
    }
    .container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
    }
    .logo {
      width: 292px;
      height: auto;
      margin-bottom: 20px;
    }
    .search-bar {
      display: flex;
      width: 600px;
      height: 40px;
      border: 1px solid #ccc;
      border-radius: 20px;
      padding: 5px;
    }
    .search-bar input {
      flex: 1;
      border: none;
      outline: none;
      padding: 5px;
      font-size: 16px;
    }
    .search-bar button {
      background-color: #4285f4;
      color: white;
      border: none;
      border-radius: 20px;
      padding: 10px 15px;
      cursor: pointer;
    }
    .footer {
      margin-top: 20px;
      text-align: center;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="container">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Opossum_2.jpg/292px-Opossum_2.jpg" alt="Opossum" class="logo">
    <form action="https://www.google.com/search" method="get">
      <div class="search-bar">
        <input type="text" name="q" placeholder="Search the web" required>
        <button type="submit">Search</button>
      </div>
      <input type="hidden" name="q" value="opossum ">
    </form>
  </div>
  <div class="footer">Powered by opossum search</div>
</body>
</html>



