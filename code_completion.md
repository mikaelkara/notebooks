```
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Getting Started with the Vertex AI Codey APIs - Code Completion
> **NOTE:** This notebook uses the PaLM generative model, which will reach its [discontinuation date in October 2024](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text#model_versions). 

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_completion.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_completion.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/code/code_completion.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Lavi Nigam](https://github.com/lavinigam-gcp), [Polong Lin](https://github.com/polong-lin) |

## Overview

This notebook aims to provide a hands-on introduction to code completion using [Codey models](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview), specifically the `code-gecko` model. You will learn how to create prompts to interact with the `code-gecko` model and generate code suggestions based on the context of the code you're writing.


### Vertex AI PaLM API
The Vertex AI PaLM API, [released on May 10, 2023](https://cloud.google.com/vertex-ai/docs/generative-ai/release-notes#may_10_2023), is powered by [PaLM 2](https://ai.google/discover/palm2).

### Using Vertex AI PaLM API

You can interact with the Vertex AI PaLM API using the following methods:

* Use the [Generative AI Studio](https://cloud.google.com/generative-ai-studio) for quick testing and command generation.
* Use cURL commands in Cloud Shell.
* Use the Python SDK in a Jupyter notebook

This notebook focuses on using the Python SDK to call the Vertex AI PaLM API. For more information on using Generative AI Studio without writing code, you can explore [Getting Started with the UI instructions](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/intro_vertex_ai_studio.md)


For more information, check out the [documentation on generative AI support for Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview).

### Objectives

In this tutorial, you will learn various code completion examples for completing:
* Functions
* Classes
* Statements
* Expressions & Variables
* Imports
  

### Costs
This tutorial uses billable components of Google Cloud:

* Vertex AI Generative AI Studio

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK


```
%pip install google-cloud-aiplatform --upgrade --user
```

**Colab only:** Uncomment the following cell to restart the kernel or use the button to restart the kernel. For Vertex AI Workbench you can restart the terminal using the button on top.


```
# Automatically restart kernel after installs so that your environment can access the new packages
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

### Authenticating your notebook environment
* If you are using **Colab** to run this notebook, uncomment the cell below and continue.
* If you are using **Vertex AI Workbench**, check out the setup instructions [here](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/setup-env).


```
# from google.colab import auth
# auth.authenticate_user()
```

## Code Completion

What is Code Completion?

Code completion is a feature in many integrated development environments (IDEs) that suggests code to the programmer as they are typing. This can save time and help to prevent errors. Code completion suggestions are based on the context of the code being written, such as the programming language, the current line of code, and the variables that have been defined.



Benefits of using code completion?

There are several benefits to using code completion in general, including:

* **Increased productivity**: Code completion can save programmers a lot of time by suggesting code as they are typing. This can free them up to focus on other tasks, such as designing the architecture of their software or debugging their code.

* **Reduced errors**: Code completion can help to reduce errors by suggesting code that is syntactically correct and semantically meaningful. This can be especially helpful when programmers are working with new or unfamiliar programming languages or APIs.

* **Improved code quality**: Code completion can help to improve the quality of code by suggesting code that is consistent with the style guide of the project. This can make the code more readable and maintainable.


Code Completion and IDE Integration:

When code completion through Codey Model is integrated with an IDE, it can be even more powerful. The IDE can use its knowledge of the project's structure and codebase to provide more accurate and relevant suggestions. For example, if the programmer is typing code in a class, the IDE can suggest methods and fields from that class.

Here are some of the benefits of using code completion with integration with different IDEs:

* **Increased productivity**: Code completion can help programmers write code more quickly and accurately, which can lead to significant productivity gains.
* **Improved code quality**: Code completion can help programmers avoid errors and typos, and can also suggest more efficient and idiomatic code.
* **Better code readability**: Code completion can help programmers write more readable and maintainable code by suggesting consistent variable names and function signatures.
* **Reduced learning curve**: Code completion can help new programmers learn new languages and frameworks more quickly by suggesting the correct symbols and functions to use.


### Import libraries

**Colab only:** Uncomment the following cell to initialize the Vertex AI SDK. For Vertex AI Workbench, you don't need to run this.  


```
# import vertexai

# PROJECT_ID = ""  # @param {type:"string"}
# vertexai.init(project=PROJECT_ID, location="us-central1")
```


```
from vertexai.language_models import CodeGenerationModel
```

## Code completion with code-gecko@002

The Vertex AI Codey APIs include the code completion API, which supports code suggestions based on code that's recently written. Use the generative AI foundation model named `code-gecko` to interact with the code completion API.

To learn more about creating prompts for code completion, see [Create prompts for code completion](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/code-completion#:~:text=code%20completion%2C%20see-,Create%20prompts%20for%20code%20completion,-.).

Code completion API has few more parameters than code generation.

* prefix: required : For code models, prefix represents the beginning of a piece of meaningful programming code or a natural language prompt that describes code to be generated.

* suffix: optional : For code completion, suffix represents the end of a piece of meaningful programming code. The model attempts to fill in the code in between the prefix and suffix.

* temperature: required : Temperature controls the degree of randomness in token selection. Same as for other models. range: (0.0 - 1.0, default 0)

* maxOutputTokens: required : Maximum number of tokens that can be generated in the response. range: (1 - 64, default 64)

* stopSequences: optional : Specifies a list of strings that tells the model to stop generating text if one of the strings is encountered in the response. The strings are case-sensitive.

### Load model


```
code_completion_model = CodeGenerationModel.from_pretrained("code-gecko@latest")
```

### Hello Codey Completion

#### Python


```
prefix = """def find_x_in_string(string_s, x):

         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """
         def reverse_string(s):
            return s[::-1]
         def test_empty_input_string():
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### Java


```
prefix = """
        ArrayList<String> myList = new ArrayList<>();
        //add the `String` "Hello, world!" to the `ArrayList`:
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """
        public static List<String> getUniqueStrings(List<String> strings) {
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """
        String[] names = {"Alice", "Bob", "Carol"};
        for (String name : names) {
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### JavaScript


```
prefix = """
        #javaScript
        function factorial(n) {
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """
        function greet(name) {
            return "Hello, " + name + "!";
          }
        const greeting = greet(YOUR_NAME_HERE);
         """

response = code_completion_model.predict(prefix=prefix)

print(response.text)
```

#### C/C++


```
prefix = """
        int main() {
          char str[] = "Hello, world!";
         """

response = code_completion_model.predict(prefix=prefix)

print(response.text)
```


```
prefix = """
        LinkedList linkedList;

        linkedList.addNode(1);
        linkedList.addNode(2);
        linkedList.addNode(3);

        int value =
         """

response = code_completion_model.predict(prefix=prefix)

print(response.text)
```

### Code Completion Example:

#### Completing functions


```
prefix = """import math
            # Start typing the name of a function
            def sqrt(x):
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """def greet(name):
              print(f"Hello, {name}!")

            # Call the greet() function
         """

response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### Completing Class


```
prefix = """class Dog:
              def bark(self):
                print("Woof!")

            # Create a new Dog object
          """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """class Person:
              #Represents a person.
              def __init__(self, name, age):
                self.name = name
                self.age = age

            # Start typing the name of the Person class
            Person(
          """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### Completing Statements


```
prefix = """if age >= 21:
              print("You are an adult. ")
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """if x < 10:
              # Complete the statement
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### Completing Expressions


```
prefix = """x = 10 +
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """1 + 2 * 3
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### Completing Variables


```
prefix = """# Define a variable
            name = "Alice"
            #get uppercase of the variable
            name.upper()
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """x = 10
            y = x +
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

#### Completing Imports


```
prefix = """import math
            import numpy as np
            #import machine learning libraries
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```


```
prefix = """import math
            import time
            import random
            import sys
         """
response = code_completion_model.predict(prefix=prefix, max_output_tokens=64)

print(response.text)
```

### Feedback Loop Code Completion


```
prefix = "def find_max_element(list):"
i = 0
while i < 3:
    response = code_completion_model.predict(
        prefix=prefix,
    )
    print(response.text)
    prefix = response.text
    i += 1
```


```
prefix = """class Dog:
              def bark(self):
                print("Woof!")
          """
i = 0
while i < 3:
    response = code_completion_model.predict(
        prefix=prefix,
    )
    print(response.text)
    prefix = response.text
    i += 1
```

### Best Practices

#### **How to write effective code completion prompts**

When writing code completion prompts, it is important to be as specific and descriptive as possible. The more information you can provide the model, the better it will be able to understand what you are trying to achieve.

Here are some tips for writing effective code completion prompts:

* Start with a natural language description of what you want the model to generate. This should be a clear and concise statement of your goal, such as "Complete the following function to print the sum of two numbers" or "Generate a function to check if a string is a palindrome."

* Include any relevant context in the prompt. This could include the code that you have already written, the programming language you are using, or any other information that the model might need to know.

* Use examples to illustrate what you are looking for. If you can, provide the model with examples of the code that you want it to generate. This will help the model to better understand your intentions.
Here is an example of a good code completion prompt:




```
Complete the following Python function to check if a string is a palindrome:

def is_palindrome(string):
  """Checks if a string is a palindrome.

  Args:
    string: A string.

  Returns:
    A boolean value indicating whether the string is a palindrome.
  """
  # TODO: Implement this function.
```


#### **How to choose the right temperature and max output tokens**

The temperature and max output tokens are two important parameters that control the behavior of the code completion model.

* Temperature: The temperature controls how creative the model is. A higher temperature will lead to more creative and unexpected suggestions, while a lower temperature will lead to more conservative and predictable suggestions.

* Max output tokens: The max output tokens controls the maximum length of the code that the model can generate. A higher max output tokens will allow the model to generate longer code snippets, while a lower max output tokens will limit the length of the generated code.

When choosing the right temperature and max output tokens, it is important to consider the specific task that you are trying to accomplish. If you need the model to generate creative and unexpected suggestions, you should use a higher temperature. If you need the model to generate code snippets that are of a specific length, you should use the appropriate max output tokens.

#### **How to interpret and use code completion suggestions**

Once you have generated some code completion suggestions, it is important to carefully interpret and use them.

The code completion model is not perfect, and it is possible that it will generate suggestions that are incorrect or incomplete. It is important to review the suggestions carefully and to test them before using them in your code.

Here are some tips for interpreting and using code completion suggestions:

* Make sure that the suggestions are syntactically correct. The code completion model may generate suggestions that are syntactically incorrect. It is important to check the syntax of the suggestions before using them in your code.

* Test the suggestions before using them in your code. Once you have found some suggestions that you are happy with, it is important to test them before using them in your code. This will help to ensure that the suggestions are correct and that they will work as expected.

#### **How to avoid common code completion pitfalls**

Here are some common code completion pitfalls to avoid:

* Do not rely on the code completion model to generate all of your code. The code completion model is a tool, but it should not be used to generate all of your code. It is important to understand the code that you are writing and to be able to review and test it carefully.

* Do not use code completion suggestions without understanding them. It is important to understand the code completion suggestions before using them in your code. This will help you to identify any potential errors or problems.

* Do not use code completion suggestions for tasks that are too complex. The code completion model is not designed to generate complex code snippets. If you need to generate complex code, it is best to write it yourself.
