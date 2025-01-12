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

# Gemini API: Basic evaluation

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Basic_Evaluation.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

Gemini API's Python SDK can be used for various forms of evaluation, including:
- Providing feedback based on selected criteria
- Comparing multiple texts
- Assigning grades or confidence scores
- Identifying weak areas

Below is an example of using the LLM to enhance text quality through feedback and grading.


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

Start by defining some system instructions for this problem. For demonstration purposes, the use case involves prompting the model to write an essay with some mistakes. Remember that for generation tasks like writing an essay, you can change the temperature of the model to get more creative answers. Here, you can use `"temperature": 0` for predictability.


```
student_system_prompt = """You're a college student. Your job is to write an essay riddled with common mistakes and a few major ones.
The essay should have mistakes regarding clarity, grammar, argumentation, and vocabulary.
Ensure your essay includes a clear thesis statement. You should write only an essay, so do not include any notes."""

student_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', generation_config={"temperature": 0},
                              system_instruction=student_system_prompt)

essay = student_model.generate_content("Write an essay about benefits of reading.").text
Markdown(essay)
```




## The Power of Words: Why Reading is the Best Thing Ever

Reading is a super important thing. It's like, the best thing ever. Everyone should read more, because it's good for you. Like, really good. Reading can make you smarter, more knowledgeable, and even a better person. It's like a magic potion that can transform your brain.

Firstly, reading helps you learn new things. When you read, you're basically absorbing information like a sponge. You can learn about history, science, art, and even how to cook a mean lasagna. It's like having a personal tutor in your pocket, except it doesn't cost anything.

Secondly, reading can make you a better writer. By reading, you learn how to use language effectively. You can see how different authors use words to create different effects. It's like learning from the masters. Plus, reading can help you expand your vocabulary, which is super important for sounding smart.

Thirdly, reading can help you relax and de-stress. When you're reading a good book, you can escape from the real world and get lost in another one. It's like taking a vacation without leaving your couch. Reading can also help you sleep better, because it can calm your mind and help you unwind.

In conclusion, reading is awesome. It's like a superpower that can make you smarter, more knowledgeable, and a better person. So, next time you're feeling bored or stressed, pick up a book and give it a try. You won't regret it. 





```
teacher_system_prompt = f"""
As a teacher, you are tasked with grading students' essays.
Please follow these instructions for evaluation:

1. Evaluate the essay on a scale of 1-5 based on the following criteria:
- Thesis statement,
- Clarity and precision of language,
- Grammar and punctuation,
- Argumentation

2. Write a corrected version of the essay, addressing any identified issues
in the original submission. Point what changes were made.
"""
teacher_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', generation_config={"temperature": 0},
                                         system_instruction=teacher_system_prompt)

Markdown(teacher_model.generate_content(essay).text)
```




## Evaluation:

**Thesis Statement:** 2/5 - The essay states that reading is "the best thing ever" but doesn't offer a specific argument or claim. It needs a more focused thesis statement.

**Clarity and Precision of Language:** 2/5 - The language is informal and uses slang ("super important," "like," "really good," "super important"). It lacks the precision and formality expected in academic writing.

**Grammar and Punctuation:** 2/5 - There are several grammatical errors, including comma splices and run-on sentences. The essay also lacks proper punctuation, particularly with the use of commas.

**Argumentation:** 3/5 - The essay presents three benefits of reading, but the arguments are not fully developed. They rely on generalizations and lack specific examples or evidence.

## Corrected Version:

**The Power of Words: The Transformative Benefits of Reading**

Reading is a fundamental human activity with profound benefits for individuals and society. It is not merely a pastime but a powerful tool that can enhance cognitive abilities, broaden perspectives, and foster empathy. This essay will explore three key ways in which reading transforms individuals: by expanding knowledge, improving communication skills, and promoting emotional well-being.

Firstly, reading serves as a gateway to knowledge and understanding. Through books, we can explore diverse cultures, historical events, scientific discoveries, and philosophical ideas. By immersing ourselves in different worlds and perspectives, we expand our intellectual horizons and develop a deeper understanding of the complexities of the world around us. For example, reading historical accounts can provide valuable insights into the past, while scientific texts can deepen our understanding of the natural world.

Secondly, reading enhances communication skills. By engaging with different writing styles and literary techniques, readers develop a greater appreciation for the nuances of language. They learn how to use words effectively to convey ideas, emotions, and arguments. Moreover, reading exposes individuals to a wider vocabulary, enriching their communication and enabling them to express themselves with greater clarity and precision.

Thirdly, reading promotes emotional well-being. Engaging with a good book can provide a much-needed escape from the stresses of daily life. It allows individuals to immerse themselves in fictional worlds, connect with relatable characters, and experience a range of emotions. This process can be therapeutic, fostering empathy, reducing stress, and promoting relaxation. Furthermore, reading can inspire creativity and imagination, enriching our inner lives and fostering a sense of wonder.

In conclusion, reading is not simply a leisure activity but a transformative experience that enriches our lives in countless ways. By expanding our knowledge, improving our communication skills, and promoting emotional well-being, reading empowers us to become more informed, articulate, and compassionate individuals. 

**Changes Made:**

* **Thesis Statement:** The thesis statement is now more specific and focused, outlining the three key benefits of reading that will be discussed in the essay.
* **Clarity and Precision of Language:** The language is more formal and precise, avoiding slang and informal expressions.
* **Grammar and Punctuation:** The essay has been corrected for grammatical errors, including comma splices and run-on sentences. Proper punctuation has been implemented throughout.
* **Argumentation:** The arguments are more developed, providing specific examples and evidence to support the claims. The essay now uses a more academic tone and structure. 




## Next steps

Be sure to explore other examples of prompting in the repository. Try writing your own prompts for evaluating texts.
