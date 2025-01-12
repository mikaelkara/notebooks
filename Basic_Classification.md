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

# Gemini API: Basic classification

This notebook demonstrates how to use prompting to perform classification tasks using the Gemini API's Python SDK.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Basic_Classification.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

LLMs can be used in tasks that require classifying content into predefined categories. This business case shows how it categorizes user messages under the blog topic. It can classify replies in the following categories: spam, abusive comments, and offensive messages.


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

## Examples


```
classification_system_prompt = """

As a social media moderation system, your task is to categorize user comments under a post.
Analyze the comment related to the topic and classify it into one of the following categories:

Abusive
Spam
Offensive

If the comment does not fit any of the above categories, classify it as: Neutral.

Provide only the category as a response without explanations.
"""
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', generation_config={"temperature": 0},
                              system_instruction=classification_system_prompt)
```


```
# Define a template that you will reuse in examples below
classification_template = """
Topic: What can I do after highschool?
Comment: You should do a gap year!
Class: Neutral

Topic: Where can I buy a cheap phone?
Comment: You have just won an IPhone 15 Pro Max!!! Click the link to receive the prize!!!
Class: Spam

Topic: How long do you boil eggs?
Comment: Are you stupid?
Class: Offensive

Topic: {topic}
Comment: {comment}
Class: """
```


```
spam_topic = "I am looking for a vet in our neighbourhood. Can anyone reccomend someone good? Thanks."
spam_comment = "You can win 1000$ by just following me!"
spam_prompt = classification_template.format(topic=spam_topic, comment=spam_comment)
Markdown(model.generate_content(spam_prompt).text)
```




Spam 





```
neutral_topic = "My computer froze. What should I do?"
neutral_comment = "Try turning it off and on."
neutral_prompt = classification_template.format(topic=neutral_topic, comment=neutral_comment)
Markdown(model.generate_content(neutral_prompt).text)
```




Neutral 




## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own datasets.
