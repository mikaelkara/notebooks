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

# Gemini API: Entity extraction

Use Gemini API to speed up some of your tasks, such as searching through text to extract needed information. Entity extraction with a Gemini model is a simple query, and you can ask it to retrieve its answer in the form that you prefer.

This notebook shows how to extract entities into a list.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Entity_Extraction.ipynb"><img src = "../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

## Setup


```
!pip install -U -q "google-generativeai>=0.7.2"
```


```
import google.generativeai as genai
from IPython.display import Markdown
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
```

# Examples

You will use Gemini 1.5 Flash model for fast responses.


```
model = genai.GenerativeModel('gemini-1.5-flash-latest')
```

### Extracting few entities at once

This block of text is about possible ways to travel from the airport to the Colosseum.  

Let's extract all street names and proposed forms of transportation from it.


```
directions = """To reach the Colosseum from Rome's Fiumicino Airport (FCO), your options are diverse.
Take the Leonardo Express train from FCO to Termini Station, then hop on metro line A towards Battistini and alight at Colosseo station.
Alternatively, hop on a direct bus, like the Terravision shuttle, from FCO to Termini, then walk a short distance to the Colosseum on Via dei Fori Imperiali.
If you prefer a taxi, simply hail one at the airport and ask to be taken to the Colosseum.
The taxi will likely take you through Via del Corso and Via dei Fori Imperiali.
A private transfer service offers a direct ride from FCO to the Colosseum, bypassing the hustle of public transport.
If you're feeling adventurous, consider taking the train from FCO to Ostiense station,
then walking through the charming Trastevere neighborhood, crossing Ponte Palatino to reach the Colosseum, passing by the Tiber River and Via della Lungara.
Remember to validate your tickets on the metro and buses, and be mindful of pickpockets, especially in crowded areas.
No matter which route you choose, you're sure to be awed by the grandeur of the Colosseum."""
```


```
directions_prompt = f"""
  From the given text, extract the following entities and return a list of them.
  Entities to extract: street name, form of transport.
  Text: {directions}
  Street = []
  Transport = [] """

Markdown(model.generate_content(directions_prompt).text)
```




```json
{
  "Street": [
    "Via dei Fori Imperiali",
    "Via del Corso",
    "Via della Lungara"
  ],
  "Transport": [
    "train",
    "metro",
    "bus",
    "Terravision shuttle",
    "taxi",
    "private transfer service"
  ]
}
```



You can modify the form of the answer for your extracted entities even more:


```
directions_list_prompt = f"""
  From the given text, extract the following entities and return a list of them.
  Entities to extract: street name, form of transport.
  Text: {directions}
  Return your answer as two lists:
  Street = [street names]
  Transport = [forms of transport]"""

Markdown(model.generate_content(directions_list_prompt).text)
```




Here are the extracted entities:

**Street:**
- Via dei Fori Imperiali
- Via del Corso
- Via della Lungara 

**Transport:**
- Train (Leonardo Express)
- Metro (line A)
- Bus (Terravision shuttle)
- Taxi
- Private transfer service 




### Numbers

Try entity extraction of phone numbers


```
customer_service_email = """
Hello,
Thank you for reaching out to our customer support team regarding your recent purchase of our premium subscription service.
We will send activation code to +87 668 098 344
Additionally, if you require immediate assistance, feel free to contact us directly at +1 (800) 555-1234.
Our team is available Monday through Friday from 9:00 AM to 5:00 PM PST.
For after-hours support, please call our dedicated emergency line at +87 455 555 678.
We appreciate your business and look forward to resolving any issues you may encounter promptly.
Thank you."""
```


```
phone_prompt = f"""
  From the given text, extract the following entities and return a list of them.
  Entities to extract: phone numbers.
  Text: {customer_service_email}
  Return your answer in a list:"""

Markdown(model.generate_content(phone_prompt).text)
```




The phone numbers are:
- +87 668 098 344
- +1 (800) 555-1234
- +87 455 555 678 




### URLs


Try entity extraction of URLs and get response as a clickable link.


```
url_text = """
Gemini API billing FAQs

This page provides answers to frequently asked questions about billing for the Gemini API. For pricing information, see the pricing page https://ai.google.dev/pricing.
For legal terms, see the terms of service https://ai.google.dev/gemini-api/terms#paid-services.

What am I billed for?
Gemini API pricing is based on total token count, with different prices for input tokens and output tokens. For pricing information, see the pricing page https://ai.google.dev/pricing.

Where can I view my quota?
You can view your quota and system limits in the Google Cloud console https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas.

Is GetTokens billed?
Requests to the GetTokens API are not billed, and they don't count against inference quota."""
```


```
url_prompt = f"""
  From the given text, extract the following entities and return a list of them.
  Entities to extract: URLs.
  Text: {url_text}
  Do not duplicate entities.
  Return your answer in a markdown format:"""

Markdown(model.generate_content(url_prompt).text)
```




- https://ai.google.dev/pricing
- https://ai.google.dev/gemini-api/terms#paid-services
- https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas 



