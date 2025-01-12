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

# Gemini API: Chain of thought prompting

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Chain_of_thought_prompting.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

Using chain of thought helps the LLM take a logical and arithmetic approach. Instead of outputting the answer immediately, the LLM uses smaller and easier steps to get to the answer.


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

## Example

Sometimes LLMs can return non-satisfactory answers. To simulate that behavior, you can implement a phrase like "Return the answer immediately" in your prompt.

Without this, the model sometimes uses chain of thought by itself, but it is inconsistent and does not always result in the correct answer.


```
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
```


```
prompt = """
5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
Return the answer immediately.
"""
Markdown(model.generate_content(prompt).text)
```




5 minutes 




To influence this you can implement chain of thought into your prompt and look at the difference in the response. Note the multiple steps within the prompt.


```
prompt = """
Question: 11 factories can make 22 cars per hour. How much time would it take 22 factories to make 88 cars?
Answer: A factory can make 22/11=2 cars per hour. 22 factories can make 22*2=44 cars per hour. Making 88 cars would take 88/44=2 hours. The answer is 2 hours.
Question: 5 people can create 5 donuts every 5 minutes. How much time would it take 25 people to make 100 donuts?
Answer:"""
Markdown(model.generate_content(prompt).text)
```




Here's how to solve the donut problem:

**1. Find the individual production rate:**

* If 5 people make 5 donuts in 5 minutes, that means one person makes one donut in 5 minutes.

**2.  Calculate the total production rate:**

* With 25 people, and each person making one donut every 5 minutes, they would make 25 donuts every 5 minutes.

**3. Determine the time to make 100 donuts:**

* Since they make 25 donuts every 5 minutes, it will take 4 sets of 5 minutes to make 100 donuts (100 donuts / 25 donuts per 5 minutes = 4 sets).

**Answer:** It would take 25 people **20 minutes** (4 sets x 5 minutes per set) to make 100 donuts. 




## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own data, or try some of the other prompting techniques such as few-shot prompting.
