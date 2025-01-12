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

# Getting Started with the Vertex AI Codey APIs - Code Generation

> **NOTE:** This notebook uses the PaLM generative model, which will reach its [discontinuation date in October 2024](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text#model_versions).

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Run in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/code/code_generation.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/language/code/code_generation.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Vertex AI Workbench
    </a>
  </td>
</table>


| | |
|-|-|
|Author(s) | [Lavi Nigam](https://github.com/lavinigam-gcp), [Polong Lin](https://github.com/polong-lin) |

## Overview

This notebook will provide an introduction to code generation using [Codey models](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview). Codey for Code Generation (code-bison) is a foundation model that can generate code based on a natural language description. It can be used to create functions, web pages, unit tests, and other types of code. Codey for Code Generation is supported by the code generation Codey APIs, which are part of the PaLM API family.


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

In this tutorial, you will learn.

- Setting up the environment for code generation models
- Writing basic prompts for Codey
- Various prompt design patterns:
	- Problem Statement Template
	- SQL Metadata & Performance
	- Code Optimization
	- Chain of Thought
	- Few Shot Prompts
	- DevOps Templates
	- Web Templates
- Best Practices with Code Generation
  

### Costs
This tutorial uses billable components of Google Cloud:

* Vertex AI Generative AI Studio

Learn about [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing),
and use the [Pricing Calculator](https://cloud.google.com/products/calculator/)
to generate a cost estimate based on your projected usage.

## Getting Started

### Install Vertex AI SDK


```
%pip install google-cloud-aiplatform==1.36.2 --upgrade --user
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

## Vertex AI PaLM API models

The Vertex AI PaLM API enables you to test, customize, and deploy instances of Google's large language model (LLM) called as PaLM,so that you can leverage the capabilities of PaLM in your applications.

### Model naming scheme
Foundation model names have three components: use case, model size, and version number. The naming convention is in the format:  
`<use case>-<model size>@<version number>`

For example, text-bison@002 represents the Bison text model, version 002.

The model sizes are as follows:
- **Bison**: The best value in terms of capability and cost.
- **Gecko**: The smallest and cheapest model for simple tasks.

### Available models

The Vertex AI Codey API currently supports three models:

* `code-bison@002`: A model fine-tuned to generate code based on a natural language description of the desired code. For example, it can generate a unit test for a function.

* `code-gecko@002`: A model fine-tuned to suggest code completion based on the context in code that's written.

* `codechat-bison@002`: A model fine-tuned for chatbot conversations that help with code-related questions.

You can find more information about the properties of these [foundational models in the Generative AI Studio documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models#foundation_models).


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

## Code generation with code-bison@002

The code generation model (Codey) from PaLM API that you will use in this notebook is code-bison@002. It is fine-tuned to follow natural language instructions to generate required code and is suitable for a variety of coding tasks, such as:

- writing functions
- writing classes
- web pages
- unit tests
- docstrings
- code translations, and many more use-cases.

Currently it supports the following languages:
- C++
- C#
- Go
- GoogleSQL
- Java
- JavaScript
- Kotlin
- PHP
- Python
- Ruby
- Rust
- Scala
- Swift
- TypeScript

You can find our more details [here](https://cloud.google.com/vertex-ai/docs/generative-ai/code/code-models-overview).

### Load model


```
code_generation_model = CodeGenerationModel.from_pretrained("code-bison@002")
```

### Model parameters for `code-bison`

You can customize how the PaLM API code generation behaves in response to your prompt by using the following parameters for `code-bison@002`:

 - `prefix`: it represents the beginning of a piece of meaningful programming code or a natural language prompt that describes code to be generated.
 - `temperature`: higher means more "creative" code responses. range: (0.0 - 1.0, default 0).
 - `max_output_tokens`: sets the max number of tokens in the output. range: (1 - 2048, default 2048)


### Hello Codey

You can test the code that is being generated [here](https://onecompiler.com/)


```
prefix = "write a python function to do binary search"

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

### Try out your own prompt

Some examples:
* write Go program to extract ip addresses from the text file
* write Java program that can extract pincodes from addresses
* write a standard SQL function that strips all non-alphabet characters from the string and encodes it to utf-8


```
prefix = """write a python function that can do cosine similarity between two vectors,
            named as "calculate_cosine_similarity" and two input arguments "vector1" and "vector2". \
          """

response = code_generation_model.predict(prefix=prefix, max_output_tokens=1024)

print(response.text)
```

### Prompt templates

Prompt templates are useful if you have found a good way to structure your prompt that you can re-use. This can be also be helpful in limiting the open-endedness of freeform prompts. There are many ways to implement prompt templates, and below is just one example using f-strings. This way you can structure the prompts as per the expected functionality of the code.


```
language = "python"
prefix = f"""Write a {language} function that can input unsorted array or list and return the sorted array or list.
             it should not use any pre-built function or library.
              """

response = code_generation_model.predict(prefix=prefix, max_output_tokens=1024)

print(response.text)
```


```
language = "python"
file_format = "json"
extract_info = "names"
requirements = """
              - the name should be start with capital letters.
              - There should be no duplicate names in the final list.
              """

prefix = f"""Create a {language} to parse {file_format} and extract {extract_info} with the following requirements: {requirements}.
              """

response = code_generation_model.predict(prefix=prefix, max_output_tokens=1024)

print(response.text)
```

## Prompt Design Patterns

### Problem-statement Template

In the problem-statement template, you can leverage prompt template ideas and pass your specific problem-statement without focusing on the language. The language can be a separate parameter. You can also pass the input-output example to ensure that generation follows the test case. Here are examples of different languages; you can see the model's capability to generate many supported languages.

#### C


```
language = "c"
problem_statement = "find the smallest element in an unordered list"

prefix = "write a " + language + " function to " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### C++


```
language = "cpp"
problem_statement = """Sort an array in one swap whose two elements are swapped and rest are in sorted order \
                      for example: \
                      input: {1, 5, 3, 7, 9}
                      output: {1,3,5,7,9}
                    """

prefix = "write a " + language + " function to " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Clojure


```
language = "clojure"
problem_statement = """that takes a string and calculates if its palindrome or not.
                      print the outputs with two example: 'radar' and 'happy'
                    """

prefix = "write a " + language + " function " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Elixir


```
language = "elixir"
problem_statement = """print the first non-repeated character from a string.
                      take example of 'Mississippi' and 'hello' as example.
                    """

prefix = "write a " + language + " function " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Erlang


```
language = "erlang"
problem_statement = """reverse an array in place. Take input of array.Take input of array and output the reversed array.
                       For example: [1,2,3,4,5] -> [5,4,3,2,1]
                    """

prefix = "write a " + language + " program " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Fortran


```
language = "Fortran"
problem_statement = """ to remove duplicate elements from given array.
                    """

prefix = "write a " + language + " program " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Go


```
language = "Go"
problem_statement = """that can extract ipv4 addresses from the each line in the log file. use fmt and regexp package.
                        input:
                        03/22 08:51:06 INFO   :...read_physical_netif: index #0, interface VLINK1 has address 129.1.1.1, ifidx 0
                        output:
                        129.1.1.1
                        \n\n
                    """

prefix = "write a " + language + " function " + problem_statement

response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

### SQL Metadata & Performance

In the SQL prompt template, you can define multiple criteria like problem statement, table metadata, styling of the code, and performance expectations. Passing the table metadata is crucial since it allows the model to generate consistent code that follows the structure of the metadata. Ensure that your problem statement is very clear, concise, and contains all relevant context.

#### SQL


```
problem_statement = """
                    You are a developer working on an e-commerce platform.
                    The marketing team has requested a report on the total number of orders and the average order \
                    value for each product category for the past month.
                    Your task is to generate a SQL queries to retrieve the total number of orders and the average order \
                    value for each product category for the orders placed in the:
                    1) past month,
                    2) given data range,
                    3) end of each month for given year,
                    4) christmas and new year's eve.
                    """
table_metadata = """
                 - **Orders:**
                    - `OrderID` (integer)
                    - `ProductID` (integer)
                    - `ProductName` (string)
                    - `Category` (string)
                    - `OrderDate` (date)
                    - `OrderAmount` (decimal)
                """
code_style = """
            Write a SQL query that follows best practices, is readable, and well-commented.
             """

performance_requirement = """
                          Optimize the query for performance considering the potential size of the "Orders" table.
                          Consider using appropriate indexing if necessary.
                          """

prefix = f""" Solve the following: {problem_statement}. The given table metadata is : {table_metadata} .
          Follow the following code style:{code_style} . The following performance requirement is: {performance_requirement} .
          """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### BigQuery


```
metadata = """
          A table of customer data, with the following columns:

          customer_id: The unique identifier for the customer.
          first_name: The customer's first name.
          last_name: The customer's last name.
          email: The customer's email address.
          phone_number: The customer's phone number.
          country: The customer's country of residence.
          order_history: A JSON object containing the customer's order history, including the following information for each order:
          order_id: The unique identifier for the order.
          order_date: The date the order was placed.
          order_total: The total amount of the order.
          order_items: A list of the items ordered, including the following information for each item:
          item_id: The unique identifier for the item.
          item_name: The name of the item.
          item_quantity: The quantity of the item ordered.
          item_price: The price of the item.

            """
language = "BigQuery"
problem = """solve following queries: \n
            - Total number of orders placed. \n
            - Total amount of money spent on orders. \n
            - Average order value. \n
            - Most popular item ordered (by quantity). \n
            - Most recent order placed. \n
          """
additional_requirement = """
            - The query should be efficient and scalable, as the customer table may contain millions of rows. \n
            - The query should be easy to read and maintain.
            """
prefix = f"""Write a {language} query to {problem}.
          use this as the table metadata: {metadata}.
          Here are some additional requirement for the query: {additional_requirement}.
          Generate each query as a separate query separated with a comment.
              """
response = code_generation_model.predict(prefix=prefix, max_output_tokens=2000)

print(response.text)
```

### Code Optimization

You can define your problem statement's specific optimization requirements in the code optimization prompt. The Codey models are great at following particular instructions and generating code that can meet user-specific conditions. You can experiment below with three languages and see how to give specific instructions on data structures, algorithmic complexity, and maintainability.  

#### Haskell


```
sample_input = "A list of integers, e.g. [1, 2, 3, 4, 5]"
language = "Haskell"
problem_statement = (
    f"a {language} program that calculates the sum of all the integers in the list."
)
additional_requirement = """
                      - The function should be efficient and recursive.
                      - The function should be polymorphic, so that it can be used to sum lists of any type of number.
                      """
prefix = f"""
        Write {problem_statement}. \n
        Also add example use case that take {sample_input} calling the function generated.
        Here are some additional requirement for the function: {additional_requirement}.
        """
response = code_generation_model.predict(prefix=prefix, max_output_tokens=1000)

print(response.text)
```

#### Java


```
sample_input = """ "(1 + 2) * 3"
              """
language = "Java"
problem_statement = f"a {language} program that evaluates the mathematical expression and prints the result to the console."
additional_requirement = """
                      - The program should handle all valid mathematical expressions, including those with parentheses, operators, and variables.
                      """
prefix = f"""
        Write {problem_statement}. \n
        Use {sample_input} to test the generated code.
        Here are some additional requirement for the function: {additional_requirement}.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### JavaScript


```
sample_input = """
              A JSON object containing a list of products, each product with the following properties:
              id: The unique identifier for the product.
              name: The name of the product.
              price: The price of the product.
              quantity: The quantity of the product in stock.
              """
language = "JavaScript"
problem_statement = f"""a {language} function that takes the JSON object as input and returns a new JSON object containing the following properties:
                      total_price: The total price of all the products in the list.
                      average_price: The average price of all the products in the list.
                      most_expensive_product: The most expensive product in the list.
                      least_expensive_product: The least expensive product in the list.
                      out_of_stock_products: A list of all the products that are out of stock.
                    """
additional_requirement = """
                      - The function should be efficient and scalable, as the JSON object may contain millions of products.
                      - The function should be easy to read and maintain.
                      """
prefix = f"""
        Write {problem_statement}. \n
        Also add example use case that take {sample_input} calling the function generated.
        Here are some additional requirement for the function: {additional_requirement}.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

### Chain of Though

Chain-of-thought (CoT) prompting is a technique for guiding large language models (LLMs) to generate code by breaking down the task into a series of intermediate natural language reasoning steps. This approach has been shown to improve the quality of generated code compared to traditional prompting methods.

To use CoT prompting for code generation, you first need to provide the LLM with a natural language description of the task you want to accomplish. The LLM will then generate a chain of thought, a sequence of natural language steps describing how to solve the task. Finally, the LLM will use the chain of thought to generate the code.

Here is an example of using CoT prompting to generate a C++ and Java function for specific use cases.

#### C++


```
language = "C++"
sample_input = """
                17/06/09 20:10:41 INFO slf4j.Slf4jLogger: Slf4jLogger started
                17/06/09 20:10:41 INFO Remoting: Starting remoting
                17/06/09 20:10:41 INFO Remoting: Remoting started; listening on addresses :[akka.tcp://sparkExecutorActorSystem@mesos-slave-07:55904]
                17/06/09 20:10:41 INFO util.Utils: Successfully started service 'sparkExecutorActorSystem' on port 55904.
              """
additional_requirement = """
                      - It should not use regex to find the given line
                      - the solution should be scalable and should scale linearly with additional data
                      - the output should print a flag and port number both.
                      - All the variable should be properly in scope and should be declared only once.
                      - The variables should have scope to be called in the main() function.
                      - The code should be easy to read and maintain
                      - The code should have proper typehints and comments
                      """

prefix = f"""
        Prompt 1: What is the problem we are trying to solve?

        Identify the status of the sparkExecutorActorSystem service in network log and output True or False along with the port, if True.

        Prompt 2: What is the language you want to use to solve this problem?
        {language}

        Prompt 3: What are the inputs and outputs of the function?

        Input: Network log
        Output: Boolean value indicating the status of the sparkExecutorActorSystem service and the port, if True

        Prompt 4: What are the steps involved in identifying the status of the sparkExecutorActorSystem service in network log?

        Split the network log into lines.
        Iterate over the lines and search for the line that contains the following string: Successfully started service 'sparkExecutorActorSystem' on port.
        If the line is found, extract the port number and save it in the variable
        Make sure the variables are scoped to be called in the main() function.
        If the line is not found,
        Return True along with the port number variable.
        Otherwise, return False.
        call the function passing the sample input
        Prompt 5: What is the sample input that can be tested as a test use case?
        {sample_input}

        prompt 6: Any additional expectation from the code logic?
        {additional_requirement}

        Prompt 7: Write the code for the scenario keeping additional expectation and expected language while generation along with the test case.

        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Java


```
prefix = """
        q: What is the input to the function?
        a: The input to the function is a string.

        q: What is the output of the function?
        a: The output of the function is a reversed string.

        q: What are the steps involved in reversing a string?
        a: 1) Iterate over the string from the back.
            2) Add each character to a new string in reverse order.
            3) Return the new string.

        q: Write pseudocode for the function.
        a: function reverse_string(string):
              new_string = ""
              for i in range(len(string) - 1, -1, -1):
                  new_string += string[i]
              return new_string

        q: how would you test the function?
        a: the input "hello" should return "olleh"

        q: write java code for the function following all the question-answer pairs.
        """
response = code_generation_model.predict(prefix=prefix, max_output_tokens=2048)

print(response.text)
```

### Few-shot with User-journey and Pseudo/Starter Code

In the few-shot prompt, you can also pass user-journey or pseudo/starter code examples for the code generation to adhere to specific instructions. The user journey can include sample input and output data structures as well. You can also complement that by passing some starter code so the model can follow the code generation per the structure you want. Here are some examples of different languages.

#### Kotlin


```
user_journey = """
              A Kotlin developer is working on a new Android app.
              They need to implement a feature that allows users to search for nearby restaurants.
              """
sample_input = """
              A list of restaurants and a search query:
              val restaurants = listOf(
                                Restaurant("The Grill", "123 Main Street", cuisine = "American"),
                                Restaurant("Thai Paradise", "456 Elm Street", cuisine = "Thai"),
                                Restaurant("Little Italy", "789 Pine Street", cuisine = "Italian"),
                              )

              val searchQuery = "Italian"

              """
sample_output = """
          A list of restaurants that match the search query:
          val matchingRestaurants = listOf(
              Restaurant("Little Italy", "789 Pine Street", cuisine = "Italian"),
            )
          """
language = "Kotlin"
problem_statement = f"""a {language} function that takes a list of restaurants and a search query as input and returns  \
                        list of restaurants that match the search query. The search query can be a substring of the restaurant name, \
                        address, or cuisine type.
                    """
additional_requirement = """
                     - The function should be efficient and scalable.
                     - The function should be easy to read and maintain.
                      """
prefix = f"""
        Write {problem_statement}. \n
        Also add example use case that take {sample_input} and {sample_output} calling the function generated.
        Here are some additional requirement for the function: {additional_requirement}.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Rust


```
real_world_case = """You are a software engineer at a company that develops trading software.
                     You need to write a Rust program to calculate the moving average of a stock price over a given period of time."""
problem_statement = """The program should take two inputs:
                        A vector of stock prices.
                        The period over which to calculate the moving average.
                        The program should output the moving average of the stock price over the given period.
                    """
code_style = "Idiomatic Rust"
algorithmic_complexity = "O(n)"
pseudocode = """
              1) Initialize a variable to store the moving average.
              2) Iterate over the vector of stock prices, adding each price to the moving average.
              3) Divide the moving average by the period to get the average price over the given period.
              4) Return the moving average price.
             """
test_cases = """
            // Test case 1
            let stock_prices = vec![100, 110, 120, 130, 140];
            let period = 3;

            let moving_average = calculate_moving_average(&stock_prices, period);

            assert_eq!(moving_average, 200.0);

            """
sample_input = "[500, 100, 300, 450, 120]"
prefix = f"""Write a Rust program based on a {real_world_case} .
             The problem statement that needs to be addressed is {problem_statement} .
             You can use this pseudocode as an example to generate the code step by step:  {pseudocode} .
             Add an example to call the generated function in main() {sample_input}
             It should follow the code style pattern as {code_style} and should have {algorithmic_complexity} as algorithmic complexity.
             Make sure that the code generated passes the following test cases: {test_cases}
             """
response = code_generation_model.predict(
    prefix=prefix, max_output_tokens=2000, temperature=0.2
)

print(response.text)
```

#### Scala


```
persona = """You are a Scala developer working on a backend service for an e-commerce platform. """
goal = """Your task is to generate Scala code for a data model representing products in the platform's catalog.
          The product data model should include information such as product ID, name, price, and availability.
       """
user_journey = """
              As a developer, your day-to-day tasks often involve designing data models to represent various entities in your application.
              In this scenario, you are tasked with creating a Scala case class for the product data model and a companion object with utility methods.
              """
requirements = """
            1. Create a Scala case class named `Product` with the following fields:
                - `id` (String)
                - `name` (String)
                - `price` (Double)
                - `available` (Boolean)

            2. Implement a companion object for the `Product` case class with the following methods:
                - `create` method that takes parameters for ID, name, price, and availability and returns an instance of the `Product` case class.
                - `format` method that takes a `Product` instance and returns a formatted string representation of the product.

            3. Ensure that the `create` method sets the availability to `true` by default if not provided.
              """
code_structure = """
                You have been provided with a starter code structure. Your task is to complete the code to meet the above requirements. The input code structure is as follows:

                ```scala
                // Starter code
                case class Product(id: String, name: String, price: Double, available: Boolean)

                object Product {
                  // Your generated Scala code for the companion object goes here
                }

                object Main extends App {
                  // Your test cases go here
                }
                """

prefix = f"""{persona} {goal} {user_journey} {requirements} {code_structure} """
response = code_generation_model.predict(
    prefix=prefix, max_output_tokens=2000, temperature=0.2
)

print(response.text)
```

#### Shell Script


```
persona = """
          Imagine you are a system administrator responsible for managing a Linux server environment.
          Your daily tasks often involve creating shell scripts to automate various system maintenance and monitoring tasks.
          In this scenario, you are tasked with generating a shell script to automate a common backup task.
          """
user_journey = """
          As a system administrator, you frequently need to create backup scripts to ensure data integrity and disaster recovery.
          Your goal is to generate a simple shell script that backs up a specified directory to a target backup location using the `rsync` command.
              """
requirements = """
          1. Create a shell script named `backup.sh` that takes two command-line arguments:
            - Source directory: The directory to be backed up.
            - Target directory: The directory where the backup should be stored.

          2. The script should use the `rsync` command to perform the backup. The `rsync` command should:
            - Synchronize the contents of the source directory to the target directory.
            - Preserve file permissions and timestamps.
            - Display progress information during the backup.

          3. Add comments to the script to explain its purpose and usage.
              """
starter_code = """
          You have been provided with a starter code structure.
          Your task is to complete the code to meet the above requirements.
          The input code structure is as follows:

          ```bash
          #!/bin/bash

          # Your generated Shell script code goes here
          """

prefix = f"""{persona} {user_journey} {requirements} {starter_code} """
response = code_generation_model.predict(
    prefix=prefix,
    max_output_tokens=2000,
)

print(response.text)
```

#### Solidity [BlockChain]


```
user_input = """
         - The address of the user withdrawing the tokens
         - The amount of tokens being withdrawn
         """
return_output = """
         - A boolean value indicating whether the withdrawal was successful
         """
requirements = """
          - The code should be written in Solidity using the latest best practices.
          - algorithm should be O(1) time complexity.
          """


prefix = f"""
         Generate a Solidity function called withdraw() that allows users to withdraw tokens from a decentralized exchange (DEX).
         The function should take the following inputs: {user_input} and should return: {return_output}.
         The function should also meet the following requirements: {requirements}
         """
response = code_generation_model.predict(
    prefix=prefix, max_output_tokens=2000, temperature=0.2
)

print(response.text)
```

#### Verilog


```
prefix = """
        Generate Verilog code for a 3-bit adder circuit that performs addition of two 3-bit numbers using only D flip-flops.
        The circuit should have the following inputs:
        a[2:0]
        b[2:0]
        The circuit should have the following outputs:
        sum[2:0]
        carry
        The circuit should implement the following logic:
        The circuit should add the two 3-bit numbers and store the result in the sum output. The carry output should be set to 1 if the addition results in a carry-out, and 0 otherwise.
        Constraints:
        The circuit must use only D flip-flops.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

### DevOps Templates

Codey is also capable of helping you generate DevOps-related code samples. You can generate Docker, Jenkins, GitLab CI, Prometheus Configurations, and many more such code blocks with the following templates.

#### Docker


```
prefix = """
        Generate a Dockerfile for a Python application that:
          * Builds the image from a python:latest base image
          * Exposes the following ports: 8000
          * Installs the following dependencies: pip install flask
          * Sets the working directory to /app
          * Copies the following files to the image: app.py requirements.txt ./
          * Runs the following command on startup: flask run --host=0.0.0.0
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Docker Compose


```
prefix = """
        Generate a Compose file for the following services: web
          * Define the following networks: default
          * Define the following volumes: ./app:/app
          * Define the following environments: FLASK_APP=app.py
          * Define the following links: none
          * Define the following depends_on relationships: none
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Jenkins


```
prefix = """
        Objective: Generate a Jenkinsfile for a parametrized pipeline for a Java project.
        Instructions:
        Allow the user to input parameters like "Environment" and "Feature Toggle."
        Use these parameters in the build and test stages.
        Provide default values for parameters.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### GitLab CI


```
prefix = """
        Generate GitLab CI YAML configuration for a Node.js project.
        Instructions:
          - Assume the project uses Node.js and has unit tests.
          - Include stages for "Build," "Test," and "Deploy."
          - Specify Node.js version and test script.
          - Use GitLab CI variables for sensitive information.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

#### Prometheus Configurations


```
prefix = """
         Generate YAML configurations for Prometheus to monitor a containerized application.
          Instructions:
            - Specify the job for scraping metrics.
            - Include targets, labels, and metric relabeling as needed.
            - Set up alerting rules for key metrics.
        """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

### Web Templates

You can also generate HTML and CSS codes based on prompts. In the below example, you can see how you can mention various details of a simple HTML page and its characteristics.

#### HTML


```
prefix = """write a html code for a simple page:
            - Has a button "click me" and it shows how many time user has hit that button as its counter.
            - Add style element that has page header as "My Page Counter Demo" in brown color
            - Everything should be displayed as 'center'.
            - The counter button should be blue by default and when clicked it should be red.
            - The counter value should be in green color.
            - The bottom of the page should display this message "Codey Generated this page" in bold and big font.
"""
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```


```
prefix = """
        Generate a responsive HTML code for a landing page of a new e-commerce website that sells clothes. The landing page should have the following sections:
         - A header with a logo, a navigation bar, and a search bar.
         - A hero section with a large image of a model wearing clothes from the website and a call to action button.
         - A featured products section with a grid of images of the best-selling products on the website.
         - A testimonial section with quotes from satisfied customers.
         - A footer with contact information and social media links.
       """
response = code_generation_model.predict(prefix=prefix)

print(response.text)
```

## Best Practices

### How to write effective code generation prompts

When writing code generation prompts, it is important to be as clear and specific as possible. The more information you can provide to the model, the better it will be able to understand your intent and generate the desired code.

Here are some tips for writing effective code generation prompts:

* Start with a clear and concise description of the task you want the model to perform. For example, instead of saying "Generate a function to sort a list of numbers," you could say "Generate a Python function to sort a list of integers in ascending order."

* Provide examples of the desired input and output. This will help the model to understand the format of the data and the expected output. For example, you could provide a list of unsorted numbers and the corresponding sorted list.

* Use natural language to describe the task. The model is trained on a massive dataset of text and code, so it is able to understand natural language prompts. For example, you could say "Generate a function to reverse a string in Python."

### How to choose the right temperature and max output tokens

The temperature parameter controls the randomness of the model's output. A higher temperature will result in more creative and varied output, but it may also be less accurate. A lower temperature will result in more accurate output, but it may also be less creative.

The max output tokens parameter controls the maximum number of tokens that the model will generate. This is useful for limiting the length of the output code or preventing the model from generating infinite loops.

Here are some tips for choosing the right temperature and max output tokens:

* Use a lower temperature for tasks that require high accuracy, such as generating code for machine learning models.

* Use a higher temperature for tasks that require creativity, such as generating code for web applications or games.

* Use a max output tokens parameter to limit the length of the output code or prevent the model from generating infinite loops.


How to interpret and use code generation suggestions

The code generation suggestions generated by the model are not always perfect. It is important to review the generated code carefully and make any necessary changes.

Here are some tips for interpreting and using code generation suggestions:

* Check the output code for any syntax errors.
* Make sure that the output code is consistent with your coding standards.
* Test the output code to make sure that it works as expected.

### How to avoid common code generation pitfalls

Here are some common code generation pitfalls to avoid:

* Using ambiguous or unclear prompts. The more specific you can be in your prompts, the better the model will be able to understand your intent and generate the desired code.
* Using too high of a temperature. A higher temperature can lead to less accurate and more creative output. It is important to choose the right temperature for the task you are trying to perform.
* Not reviewing the generated code carefully. The generated code is not always perfect. It is important to review the generated code carefully and make any necessary changes.

