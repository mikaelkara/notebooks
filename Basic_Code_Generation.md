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

# Gemini API: Basic code generation

This notebook demonstrates how to use prompting to perform basic code generation using the Gemini API's Python SDK. Two use cases are explored: error handling and code generation.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/prompting/Basic_Code_Generation.ipynb"><img src = "../../images/colab_logo_32px.png"/>Run in Google Colab</a>
  </td>
</table>

The Gemini API can be a great tool to save you time during the development process. Tasks such as code generation, debugging, or optimization can be done with the assistance of the Gemini model.


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

### Error handling


```
error_handling_system_prompt =f"""
Your task is to explain exactly why this error occurred and how to fix it.
"""
error_handling_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', generation_config={"temperature": 0},
                                             system_instruction=error_handling_system_prompt)
```


```
error_message = """
   1 my_list = [1,2,3]
----> 2 print(my_list[3])

IndexError: list index out of range
"""

error_prompt = f"""
You've encountered the following error message:
Error Message: {error_message}"""

Markdown(error_handling_model.generate_content(error_prompt).text)
```




The error message "IndexError: list index out of range" means you're trying to access an element in a list using an index that doesn't exist.

**Explanation:**

* **List Indexing:** In Python, lists are zero-indexed. This means the first element has an index of 0, the second element has an index of 1, and so on.
* **Your Code:** In your code, `my_list = [1, 2, 3]` has three elements. The valid indices for this list are 0, 1, and 2.
* **The Error:** You're trying to access `my_list[3]`. Since the list only has three elements, there is no element at index 3. This causes the "IndexError: list index out of range" error.

**How to Fix It:**

1. **Check the Index:** Ensure the index you're using is within the valid range of the list. In this case, you should use an index between 0 and 2.
2. **Adjust the Code:**  To access the last element of the list, use `my_list[2]`.

**Corrected Code:**

```python
my_list = [1, 2, 3]
print(my_list[2])  # This will print 3
```

**Important Note:**  Always be mindful of the size of your lists and the indices you use to avoid this common error. 




### Code generation


```
code_generation_system_prompt = f"""
You are a coding assistant. Your task is to generate a code snippet that accomplishes a specific goal.
The code snippet must be concise, efficient, and well-commented for clarity.
Consider any constraints or requirements provided for the task.

If the task does not specify a programming language, default to Python.
"""
code_generation_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest', generation_config={"temperature": 0},
                                             system_instruction=code_generation_system_prompt)
```


```
code_generation_prompt = 'Create a countdown timer that ticks down every second and prints "Time is up!" after 20 seconds'

Markdown(code_generation_model.generate_content(code_generation_prompt).text)
```




```python
import time

# Set the countdown duration in seconds
countdown_duration = 20

# Start the countdown
for i in range(countdown_duration, 0, -1):
    print(i, end=" ")
    time.sleep(1)  # Wait for 1 second

# Print "Time is up!" after the countdown
print("Time is up!")
```



Let's check if generated code works.


```
import time

# Set the countdown duration in seconds
countdown_duration = 20

# Start the countdown
for i in range(countdown_duration, 0, -1):
    print(i, end=" ")
    time.sleep(1)  # Wait for 1 second

# Print "Time is up!" after the countdown
print("Time is up!")
```

    20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 Time is up!
    

## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts around your own code as well using the examples in this notebook.
