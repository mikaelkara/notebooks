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

# Getting Started with the Vertex AI Codey APIs - Code Chat

> **NOTE:** This notebook uses the PaLM generative model, which will reach its [discontinuation date in October 2024](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text#model_versions). Please refer to [this updated notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_chat.ipynb) for a version which uses the latest Gemini model.

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_chat.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_chat.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/code/code_chat.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Lavi Nigam](https://github.com/lavinigam-gcp), [Polong Lin](https://github.com/polong-lin) |

## Overview

This notebook provides an introduction to code chat generation using [Codey chat models](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview), specifically the codechat-bison foundation model. Code chat enables developers to interact with a chatbot for assistance with code-related tasks, including debugging, documentation, and learning new concepts. The codechat-bison model is designed to facilitate multi-turn conversations, allowing for a more natural and interactive coding experience.


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

In this tutorial, you will learn:

* Code Debugging
* Code Refactoring
* Code Review
* Code Learning
* Code Boilerplates
* Prompt Design for Chat
  * Chain of Verification
  * Self-Consistency
  * Tree of Thought
  

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

## Code Chat

Vertex AI Codey APIs offer a code chat API geared towards multi-turn conversations tailored for coding scenarios. Leverage the generative AI foundation model, `codechat-bison`, to interface with the code chat API and craft prompts that initiate chatbot-based code dialogues. This guide walks you through the process of creating effective prompts to engage in code-oriented chatbot conversations using the `codechat-bison` model.

### Import libraries

**Colab only:** Uncomment the following cell to initialize the Vertex AI SDK. For Vertex AI Workbench, you don't need to run this.  


```
# import vertexai

# PROJECT_ID = ""  # @param {type:"string"}
# vertexai.init(project=PROJECT_ID, location="us-central1")
```


```
from vertexai.language_models import CodeChatModel
```

## Code chat with codechat-bison

The 'codechat-bison' model lets you have a freeform conversation across multiple turns from a code context. The application tracks what was previously said in the conversation. As such, if you expect to use conversations in your application for code generation, use the 'codechat-bison' model because it has been fine-tuned for multi-turn conversation use cases.

### Load model


```
code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
```

### Start Chat Session


```
code_chat = code_chat_model.start_chat()
```

### Send Message
Once the session is established, you can send prompts, and the model will generate output per the instructions and remember the context.


```
print(
    code_chat.send_message(
        "Please help write a function to calculate the min of two numbers in python",
    ).text
)
```

You can see that it knows the code generated in the previous step.


```
print(
    code_chat.send_message(
        "can you explain the code line by line?",
    ).text
)
```


```
print(
    code_chat.send_message(
        "can you add docstring, typehints and pep8 formating to the code?",
    ).text
)
```

## Use-cases

### Code Debugging

If you want to minimize variation of the responses, then keep the temperature=0. If you want more samples (outputs), keep greater than 0.2.


```
code_chat = code_chat_model.start_chat(temperature=0, max_output_tokens=2048)

print(
    code_chat.send_message(
        '''
        Debug the following scenario based on the problem statement, logic, code and error. Suggest possible cause of error and how to fix that.
        Explain the error in detail.

        Problem statement: I am trying to write a Python function to implement a simple recommendation system.
        The function should take a list of users and a list of items as input and return a list of recommended items for each user.
        The recommendations should be based on the user's past ratings of items.

        Logic: The function should first create a user-item matrix, where each row represents a user and each column represents an item.
        The value of each cell in the matrix represents the user's rating of the item.
        The function should then use a recommendation algorithm, such as collaborative filtering or content-based filtering, \
        to generate a list of recommended items for each user.

        Code:
        ```
        import numpy as np

        def generate_recommendations(users, items):
          """Generates a list of recommended items for each user.

          Args:
            users: A list of users.
            items: A list of items.

          Returns:
            A list of recommended items for each user.
          """

          # Create a user-item matrix.
          user_item_matrix = np.zeros((len(users), len(items)))
          for user_index, user in enumerate(users):
            for item_index, item in enumerate(items):
              user_item_matrix[user_index, item_index] = user.get_rating(item)

          # Generate recommendations using a recommendation algorithm.
          # ...

          # Return the list of recommended items for each user.
          return recommended_items

        # Example usage:
        users = [User1(), User2(), User3()]
        items = [Item1(), Item2(), Item3()]

        recommended_items = generate_recommendations(users, items)

        print(recommended_items)
        ```
        Error:
        AttributeError: 'User' object has no attribute 'get_rating'

                ```
        ''',
    ).text
)
```


```
print(
    code_chat.send_message(
        """
       can you re-write the function to address the bug of conversion to int inside the function itself?
        """,
    ).text
)
```

### Code Refactoring


```
code_chat = code_chat_model.start_chat(max_output_tokens=2048)

print(
    code_chat.send_message(
        """
        Given the following C++ code snippet:
        ```c++
        class User {
        public:
          User(const std::string& name, int age)
            : name_(name), age_(age) {}

          std::string GetName() const { return name_; }
          int GetAge() const { return age_; }

        private:
          std::string name_;
          int age_;
        };

        // This function takes a vector of users and returns a new vector containing only users over the age of 18.
        std::vector<User> GetAdultUsers(const std::vector<User>& users) {
          std::vector<User> adult_users;
          for (const User& user : users) {
            if (user.GetAge() >= 18) {
              adult_users.push_back(user);
            }
          }
          return adult_users;
        }
        ```
        Refactor this code to make it more efficient and idiomatic.
        Make sure to identify and fix potential problems.
        Explain the refactoring step by step in detail.
        List down potential changes that can be recommended to the user.
        """,
    ).text
)
```

### Code Review


```
code_chat = code_chat_model.start_chat(temperature=0, max_output_tokens=2048)

print(
    code_chat.send_message(
        """
        provide the code review line by line for the following python code: \n\n
```
# Import the requests and json modules
import requestz
import JSON

# Define a class called User
class User:
    # Define a constructor that takes the user's ID, name, and email as arguments
    def __init__(self, id, name, email):
        # Set the user's ID
        self.userId = id

        # Set the user's name
        self.userName = name

        # Set the user's email
        self.userEmail = email

    # Define a method called get_posts that gets the user's posts from the API
    def getPosts(self):
        # Create a URL to the user's posts endpoint
        url = "https://api.example.com/users/{}/posts".format(self.userId)

        # Make a GET request to the URL
        response = requestz.get(url)

        # Check if the response status code is 200 OK
        if response.statusCode != 200:
            # Raise an exception if the response status code is not 200 OK
            raise Exception("Failed to get posts for user {}".format(self.userId))

        # Convert the response content to JSON
        posts = JSON.loads(response.content)

        # Create a list of Posts
        postList = []

        # Iterate over the JSON posts and create a Post object for each post
        for post in posts:
            # Create a new Post object
            newPost = Post(post["id"], post["title"], post["content"])

            # Add the new Post object to the list of Posts
            postList.append(newPost)

        # Return the list of Posts
        return postList

# Define a class called Post
class Post:
    # Define a constructor that takes the post's ID, title, and content as arguments
    def __init__(self, id, title, content):
        # Set the post's ID
        self.postId = id

        # Set the post's title
        self.postTitle = title

        # Set the post's content
        self.postContent = content

# Define a main function
def main():
    # Create a User object for John Doe
    user = User(1, "John Doe", "john.doe@example.com")

    # Get the user's posts
    posts = user.getPosts()

    # Print the title and content of each post
    for post in posts:
        print("Post title: {}".format(post.postTitle))
        print("Post content: {}".format(post.postContent))

# Check if the main function is being called directly
if __name__ == "__main__":
    # Call the main function
    main()
```

        """,
    ).text
)
```

### Code Learning


```
code_chat = code_chat_model.start_chat(
    temperature=0,
    max_output_tokens=2048,
)

print(
    code_chat.send_message(
        '''
    I am new to Python and i have not read advanced concepts as of now. can you explain this code line by line.
    Include fundamental explanation of some of the advance concepts used in the code as well.
    Also provide an explanation as why somebody made a choice of using complex code vs simple code.  \n\n

    ```
    import functools

    def memoize(func):
      """Memoizes a function, caching its results for future calls.

      Args:
        func: The function to memoize.

      Returns:
        A memoized version of func.
      """

      cache = {}

      @functools.wraps(func)
      def memoized_func(*args, **kwargs):
        key = tuple(args) + tuple(kwargs.items())
        if key in cache:
          return cache[key]
        else:
          result = func(*args, **kwargs)
          cache[key] = result
          return result

      return memoized_func

    def lru_cache(maxsize=128):
      """A least recently used (LRU) cache decorator.

      Args:
        maxsize: The maximum number of items to keep in the cache.

      Returns:
        A decorator that wraps a function and caches its results. The least recently
        used results are evicted when the cache is full.
      """

      def decorating_function(func):
        cache = {}
        queue = collections.deque()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
          key = tuple(args) + tuple(kwargs.items())

          if key in cache:
            value = cache[key]
            queue.remove(key)
            queue.appendleft(key)
            return value

          value = func(*args, **kwargs)
          cache[key] = value
          queue.appendleft(key)

          if len(cache) > maxsize:
            key = queue.pop()
            del cache[key]

          return value

        return wrapper

      return decorating_function
    ```

    ''',
    ).text
)
```

### Code Boilerplate


```
code_chat = code_chat_model.start_chat(temperature=0, max_output_tokens=2048)

print(
    code_chat.send_message(
        """
        Write a boilerplate code for FastAPI to serve Llama 7b llm using Hugging Face locally. Add extra with some boilerplate and #todo for user to fill later:
        - input validation steps,
        - caching user inputs,
        - health check of API,
        - database connection with Redis server and
        - Database connection Google Cloud SQL,  and
        - load balance features

        Also, add some test cases that can check the functionality of the API endpoint with examples.
        """,
    ).text
)
```

## Prompt Design Patterns for Chat

### Chain of Verification

Chain of Verification (CoVe) prompting is a technique for refining the code generated by large language models (LLMs) by employing a self-verification process. It aims to mitigate the potential for hallucinations and inaccuracies in the generated code.

The CoVe process involves four key steps:

1. Drafting an Initial Response: The LLM generates an initial code response based on the provided natural language description.

1. Planning Verification Questions: A set of verification questions is formulated to scrutinize the accuracy and completeness of the initial code response.

1. Executing Verification: The verification questions are independently answered, either by the LLM itself or by external sources, to minimize potential biases in the verification process.

1. Generating a Final Verified Response: Based on the answers to the verification questions, the LLM refines the initial code response to produce a final, more accurate, and reliable code output.

CoVe prompting has demonstrated improved performance in code generation tasks compared to traditional prompting methods, resulting in more accurate and reliable code outputs.


```
code_chat = code_chat_model.start_chat(max_output_tokens=2048)

print(
    code_chat.send_message(
        """
        You are a software developer who can take instructions and follow them to generate and modify code.
        Your goal is to generate code based on what a user has asked, and to keep modifying the code based on the user's verification rules.
        Verification rules are not the same as test functions or test cases.
        Instead, they are steps that the user provides to ensure that the code meets their requirements.

        For example, if a user asks you to generate a code to calculate the factorial of a number:
          Step 1: Initial Setup for function 'calculate_n_factorial'
            - Add input:
              - n: number
            - variables
              - temp: store temporary values
          As, first step, generate a code to calculate the factorial of a number and setup the function and variables.
        and then provides the following verification rules:
          Step 2: Verification steps for the factorial function
            - The code should return 1 for the input 0.
            - The code should return 2 for the input 1.
            - The code should return 6 for the input 3.
        Now you would modify the code to ensure that it meets the verification rules.

        It's very important to adjust each and every verification in the modification of the code. Each time when the code is modified,
        explain your processes. Your job is to self-reflect and correct based on the user input and verification rule.
        Do not add anything from your end, just follow user input.
        Respond to this context with "Yes, I understand" and do not add any code at this stage. Wait for next instructions.

        """,
    ).text
)
```


```
print(
    code_chat.send_message(
        """
        Step 1:
            - Python function 'calculate_total_cost_parcel'
            - Add Input:
                - weight
                - distance
                - shipping_method
                - insurance_coverage
                - discount_code
            - Add Variable:
            base_shipping_cost: weight * distance * shipping_method
            shipping_method_multiplier = { "standard": 1.0, "expedited": 1.5, "overnight": 2.0 }
            insurance_coverage_multiplier =  { "none": 1.0, "basic": 1.1, "premium": 1.2 }
            shipping_cost = shipping_method_multiplier[shipping_method]
            insurance_cost = insurance_coverage_multiplier[insurance_coverage]
            shipping_cost = base_shipping_cost*shipping_cost *insurance_cost
            discount = 0
            total_cost = shipping_cost-discount [this is what function will return]

        """,
    ).text
)
```


```
print(
    code_chat.send_message(
        """
        Step 2: Verification steps for the type hints, docstring, and description for the input to the function:
        weight: The weight of the package in kilograms.
        distance: The distance the package will be shipped in kilometres.
        shipping_method: The shipping method, which can be one of the following:
              - "standard": Standard shipping, which takes 3-5 business days.
              - "expedited": Expedited shipping, which takes 1-2 business days.
              -  "overnight": Overnight shipping, which takes 1 business day.
        insurance_coverage: The insurance coverage, which can be one of the following:
              - "none": No insurance coverage.
              -" basic": Basic insurance coverage, which covers up to $100 in losses.
              - "premium": Premium insurance coverage, which covers up to $500 in losses.
        """,
    ).text
)
```


```
print(
    code_chat.send_message(
        """
        Step 3: Verification steps for the input to the function:
          - Check if the weight is non-negative.
          - Check if the distance is non-negative.
          - Check if the shipping method is valid.
          - Check if the insurance coverage is valid.
          - Check if the discount code is valid.
        """,
    ).text
)
```


```
print(
    code_chat.send_message(
        """
        Step 4: Verification for Discount
          - If the discount code is "SHIP10", multiply the base shipping cost by 0.10 and subtract the result from the total shipping cost.
          - If the discount code is "SHIP20", multiply the base shipping cost by 0.20 and subtract the result from the total shipping cost.
          - Otherwise, the discount code is invalid, so do not apply any discount.
        """,
    ).text
)
```


```
print(
    code_chat.send_message(
        """
        Step 5: Generate test cases that can be used to test the function 'calculate_total_cost_parcel'. The test cases \
        should include incorrect inputs, unexpected inputs, edge cases that are generally not thought by a developer or a QA.

        """,
    ).text
)
```


```
print(
    code_chat.send_message(
        """
        How did all the verification steps improve the 'calculate_total_cost_parcel' function that was generated?
        Explain in details with example, code before and after, and bullet points.
        """,
    ).text
)
```

### Self-Consistency

Self-consistency prompting is a technique for enhancing the quality of code generated by large language models (LLMs) by leveraging the model's ability to identify and favor consistent patterns in its reasoning. It aims to address the issue of inconsistent or erroneous code generation by introducing a mechanism for selecting the most consistent and reliable code output among multiple possible options.

The self-consistency prompting process involves three key steps:

* Generate Multiple Reasoning Paths: The LLM generates multiple distinct reasoning paths, which represent different approaches to solving the given code generation task.

* Evaluate Consistency: For each reasoning path, the LLM evaluates the consistency of its intermediate steps and the final code output. This involves identifying patterns, checking for contradictions, and ensuring alignment with the natural language description.

* Select the Most Consistent Response: Based on the consistency evaluation, the LLM selects the reasoning path that exhibits the highest level of consistency and produces the most reliable code output.

Self-consistency prompting has shown effectiveness in improving the accuracy and reliability of generated code, particularly for complex or ambiguous tasks. It has been demonstrated to reduce the occurrence of inconsistencies and errors, leading to more robust and trustworthy code generation.


```
code_chat = code_chat_model.start_chat(max_output_tokens=2048)

print(
    code_chat.send_message(
        """
        Input: any english words or group of characters.
        Output: reverse of the input string.

        Goal:
          1) Generate 3 different python code snippets for reverse_string() based on algorithmic complexity and mentioning it along the code.
          2) For each code snippet add typehints, docstrings, classes if required, pep8 formatting.
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        Going forward, i want you to follow each instruction one by one based on the code that is generated in the previous steps:

        Step 1:  For each code snippet, generate a test case that checks if the function reverses the string correctly. The test cases \
        should include incorrect inputs, unexpected inputs, edge cases that are generally not thought by a developer or a QA.
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        Step 2: For each code snippet, Integrate the exception handling for incorrect inputs, unexpected inputs, \
        edge cases that based on previous step and re-write the functions
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        Step 3 : Based on the test written, code completeness and algorithm complexity, select the code which is best

        Step 4:  Explain the reasoning in detail as bullet points of why this is selected compared to other options.
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        Step 5: Show the code which is selected along with its test cases.
        """,
    )
)
```

### Tree of Thought

Tree of Thought (ToT) prompting is a technique for guiding large language models (LLMs) to generate code by breaking down the task into a hierarchical structure of intermediate natural language steps. This approach aims to address the limitations of traditional prompting methods, which can lead to LLM getting stuck in local optima or generating code that is not well-structured or optimized.

The ToT prompting process involves three key steps:

* Decomposing the Task: The natural language description of the task is broken down into a series of smaller subtasks, forming a tree-like structure.

* Generating Intermediate Thoughts: For each subtask in the tree, the LLM generates a corresponding intermediate thought, which is a natural language explanation of how to solve that subtask.

* Constructing the Code: The LLM combines the intermediate thoughts into a cohesive and structured code output, following the hierarchical organization of the tree.

ToT prompting has demonstrated advantages over traditional prompting methods in code generation tasks, particularly for complex or multi-step problems. It helps the LLM to reason about the problem in a more structured and systematic way, leading to more efficient and reliable code generation.


```
code_chat = code_chat_model.start_chat(max_output_tokens=2048, temperature=0.5)

print(
    code_chat.send_message(
        """
        Imagine a tree of thoughts, where each thought represents a different step in the data preprocessing pipeline.
        The goal of this pipeline is to run a regression model on a ecommerce data from bigquery.
        Start at the root of the tree, and write down a thought that captures the main goal of the data preprocessing pipeline.
        Then, branch out from that thought and write down two more thoughts that represent related steps in the pipeline.
        Continue this process until you have a complete tree of thoughts, with each leaf representing a single line of Python code.
        For each branch and leaf, only write the thoughts and not code. Do not write code for each branch and leaves and put them in proper markdown.
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        The data also needs to be joined across different tables in BigQuery before starting pre-processing.
        For example customer table has to be merged with the order table.
        this should be added at the initial branches a thought.
        After that Add more branches for model building using BQML once the data is scaled.
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        Reconfigure the branches from the root as per the newly added thoughts. Follow the proper flow. rewrite the whole branches and leaves
        """,
    )
)
```


```
print(
    code_chat.send_message(
        """
        Generate the code for each branch and leaves.
        """,
    )
)
```
