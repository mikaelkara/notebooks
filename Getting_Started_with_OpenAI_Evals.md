# Getting Started with OpenAI Evals

The [OpenAI Evals](https://github.com/openai/evals/tree/main) framework consists of
1. A framework to evaluate an LLM (large language model) or a system built on top of an LLM.
2. An open-source registry of challenging evals

This notebook will cover:
* Introduction to Evaluation and the [OpenAI Evals](https://github.com/openai/evals/tree/main) library
* Building an Eval
* Running an Eval

#### What are evaluations/ `evals`?

Evaluation is the process of validating and testing the outputs that your LLM applications are producing. Having strong evaluations ("evals") will mean a more stable, reliable application that is resilient to code and model changes. An eval is a task used to measure the quality of the output of an LLM or LLM system. Given an input prompt, an output is generated. We evaluate this output with a set of ideal answers and find the quality of the LLM system.

#### Importance of Evaluations

If you are building with foundational models like `GPT-4`, creating high quality evals is one of the most impactful things you can do. Developing AI solutions involves an iterative design process. [Without evals, it can be very difficult and time intensive to understand](https://youtu.be/XGJNo8TpuVA?feature=shared&t=1089) how different model versions and prompts might affect your use case.

With OpenAI’s [continuous model upgrades](https://platform.openai.com/docs/models/continuous-model-upgrades), evals allow you to efficiently test model performance for your use cases in a standardized way. Developing a suite of evals customized to your objectives will help you quickly and effectively understand how new models may perform for your use cases. You can also make evals a part of your CI/CD pipeline to make sure you achieve the desired accuracy before deploying.

#### Types of evals

There are two main ways we can evaluate/grade completions: writing some validation logic in code
or using the model itself to inspect the answer. We’ll introduce each with some examples.

**Writing logic for answer checking**

The simplest and most common type of eval has an input and an ideal response or answer. For example,
we can have an eval sample where the input is "What year was Obama elected president for the first
time?" and the ideal answer is "2008". We feed the input to a model and get the completion. If the model
says "2008", it is then graded as correct. We can write a string match to check if the completion includes the phrase "2008". If it does, we consider it correct.

Consider another eval where the input is to generate valid JSON: We can write some code that
attempts to parse the completion as JSON and then considers the completion correct if it is
parsable.

**Model grading: A two stage process where the model first answers the question, then we ask a
model to look at the response to check if it’s correct.**

Consider an input that asks the model to write a funny joke. The model then generates a
completion. We then create a new input to the model to answer the question: "Is this following
joke funny? First reason step by step, then answer yes or no" that includes the completion." We
finally consider the original completion correct if the new model completion ends with "yes".

Model grading works best with the latest, most powerful models like `GPT-4` and if we give them the ability
to reason before making a judgment. Model grading will have an error rate, so it is important to validate
the performance with human evaluation before running the evals at scale. For best results, it makes
sense to use a different model to do grading from the one that did the completion, like using `GPT-4` to
grade `GPT-3.5` answers.

#### OpenAI Eval Templates

In using evals, we have discovered several "templates" that accommodate many different benchmarks. We have implemented these templates in the OpenAI Evals library to simplify the development of new evals. For example, we have defined 2 types of eval templates that can be used out of the box:

* **Basic Eval Templates**: These contain deterministic functions to compare the output to the ideal_answers. In cases where the desired model response has very little variation, such as answering multiple choice questions or simple questions with a straightforward answer, we have found this following templates to be useful.

* **Model-Graded Templates**: These contain functions where an LLM compares the output to the ideal_answers and attempts to judge the accuracy. In cases where the desired model response can contain significant variation, such as answering an open-ended question, we have found that using the model to grade itself is a viable strategy for automated evaluation.


### Getting Setup

First, go to [github.com/openai/evals](https://github.com/openai/evals), clone the repository with `git clone git@github.com:openai/evals.git` and go through the [setup instructions](https://github.com/openai/evals). 

To run evals later in this notebook, you will need to set up and specify your OpenAI API key. After you obtain an API key, specify it using the `OPENAI_API_KEY` environment variable. 

Please be aware of the costs associated with using the API when running evals.


```python
from openai import OpenAI
import pandas as pd

client = OpenAI()
```

## Building an evaluation for OpenAI Evals framework

At its core, an eval is a dataset and an eval class that is defined in a YAML file. To start creating an eval, we need

1. The test dataset in the `jsonl` format.
2. The eval template to be used

### Creating the eval dataset
Lets create a dataset for a use case where we are evaluating the model's ability to generate syntactically correct SQL. In this use case, we have a series of tables that are related to car manufacturing

First we will need to create a system prompt that we would like to evaluate. We will pass in instructions for the model as well as an overview of the table structure:
```
"TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]"
```

For this prompt, we can ask a specific question:
```
"Q: how many car makers are their in germany?"
```

And we have an expected answer:
```
"A: SELECT count ( * )  FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country   =   T2.CountryId WHERE T2.CountryName   =   'germany'"
```

The dataset needs to be in the following format:
```
"input": [{"role": "system", "content": "<input prompt>"}, {"role": "user", "content": <user input>}, "ideal": "correct answer"]
```

Putting it all together, we get:
```
{"input": [{"role": "system", "content": "TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]\n"}, {"role": "system", "content": "Q: how many car makers are their in germany"}, "ideal": ["A: SELECT count ( * )  FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country   =   T2.CountryId WHERE T2.CountryName   =   'germany'"]}
```


One way to speed up the process of building eval datasets, is to use `GPT-4` to generate synthetic data


```python
## Use GPT-4 to generate synthetic data
# Define the system prompt and user input (these should be filled as per the specific use case)
system_prompt = """You are a helpful assistant that can ask questions about a database table and write SQL queries to answer the question.
    A user will pass in a table schema and your job is to return a question answer pairing. The question should relevant to the schema of the table,
    and you can speculate on its contents. You will then have to generate a SQL query to answer the question. Below are some examples of what this should look like.

    Example 1
    ```````````
    User input: Table museum, columns = [*,Museum_ID,Name,Num_of_Staff,Open_Year]\nTable visit, columns = [*,Museum_ID,visitor_ID,Num_of_Ticket,Total_spent]\nTable visitor, columns = [*,ID,Name,Level_of_membership,Age]\nForeign_keys = [visit.visitor_ID = visitor.ID,visit.Museum_ID = museum.Museum_ID]\n
    Assistant Response:
    Q: How many visitors have visited the museum with the most staff?
    A: SELECT count ( * )  FROM VISIT AS T1 JOIN MUSEUM AS T2 ON T1.Museum_ID   =   T2.Museum_ID WHERE T2.Num_of_Staff   =   ( SELECT max ( Num_of_Staff )  FROM MUSEUM ) 
    ```````````

    Example 2
    ```````````
    User input: Table museum, columns = [*,Museum_ID,Name,Num_of_Staff,Open_Year]\nTable visit, columns = [*,Museum_ID,visitor_ID,Num_of_Ticket,Total_spent]\nTable visitor, columns = [*,ID,Name,Level_of_membership,Age]\nForeign_keys = [visit.visitor_ID = visitor.ID,visit.Museum_ID = museum.Museum_ID]\n
    Assistant Response:
    Q: What are the names who have a membership level higher than 4?
    A: SELECT Name   FROM VISITOR AS T1 WHERE T1.Level_of_membership   >   4 
    ```````````

    Example 3
    ```````````
    User input: Table museum, columns = [*,Museum_ID,Name,Num_of_Staff,Open_Year]\nTable visit, columns = [*,Museum_ID,visitor_ID,Num_of_Ticket,Total_spent]\nTable visitor, columns = [*,ID,Name,Level_of_membership,Age]\nForeign_keys = [visit.visitor_ID = visitor.ID,visit.Museum_ID = museum.Museum_ID]\n
    Assistant Response:
    Q: How many tickets of customer id 5?
    A: SELECT count ( * )  FROM VISIT AS T1 JOIN VISITOR AS T2 ON T1.visitor_ID   =   T2.ID WHERE T2.ID   =   5 
    ```````````
    """

user_input = "Table car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]"

messages = [{
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_input
    }
]

completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,
    temperature=0.7,
    n=5
)

for choice in completion.choices:
    print(choice.message.content + "\n")

```

    Q: What is the average horsepower for cars made in Europe?
    A: SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN model_list ON car_names.Model = model_list.Model JOIN car_makers ON model_list.Maker = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId JOIN continents ON countries.Continent = continents.ContId WHERE continents.Continent = 'Europe'
    
    Q: What is the average horsepower for cars made in the USA?
    A: SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN car_makers ON car_names.MakeId = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId WHERE countries.CountryName = 'USA'
    
    Q: What is the average horsepower for cars produced in countries from the continent with the id '3'?
    A: SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN model_list ON car_names.Model = model_list.Model JOIN car_makers ON model_list.Maker = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId JOIN continents ON countries.Continent = continents.ContId WHERE continents.ContId = '3'
    
    Q: What is the average horsepower for cars made by makers from Europe?
    A: SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN model_list ON car_names.Model = model_list.Model JOIN car_makers ON model_list.Maker = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId JOIN continents ON countries.Continent = continents.ContId WHERE continents.Continent = 'Europe'
    
    Q: What is the average horsepower for cars made in the USA?
    
    A: SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN car_makers ON car_names.MakeId = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId WHERE countries.CountryName = 'USA'
    
    

Once we have the synthetic data, we need to convert it to match the format of the eval dataset.


```python
eval_data = []
input_prompt = "TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]"

for choice in completion.choices:
    question = choice.message.content.split("Q: ")[1].split("\n")[0]  # Extracting the question
    answer = choice.message.content.split("\nA: ")[1].split("\n")[0]  # Extracting the answer
    eval_data.append({
        "input": [
            {"role": "system", "content": input_prompt},
            {"role": "user", "content": question},
        ],
        "ideal": answer
    })

for item in eval_data:
    print(item)
```

    {'input': [{'role': 'system', 'content': 'TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]'}, {'role': 'user', 'content': 'What is the average horsepower for cars made in Europe?'}], 'ideal': "SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN model_list ON car_names.Model = model_list.Model JOIN car_makers ON model_list.Maker = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId JOIN continents ON countries.Continent = continents.ContId WHERE continents.Continent = 'Europe'"}
    {'input': [{'role': 'system', 'content': 'TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]'}, {'role': 'user', 'content': 'What is the average horsepower for cars made in the USA?'}], 'ideal': "SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN car_makers ON car_names.MakeId = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId WHERE countries.CountryName = 'USA'"}
    {'input': [{'role': 'system', 'content': 'TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]'}, {'role': 'user', 'content': "What is the average horsepower for cars produced in countries from the continent with the id '3'?"}], 'ideal': "SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN model_list ON car_names.Model = model_list.Model JOIN car_makers ON model_list.Maker = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId JOIN continents ON countries.Continent = continents.ContId WHERE continents.ContId = '3'"}
    {'input': [{'role': 'system', 'content': 'TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]'}, {'role': 'user', 'content': 'What is the average horsepower for cars made by makers from Europe?'}], 'ideal': "SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN model_list ON car_names.Model = model_list.Model JOIN car_makers ON model_list.Maker = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId JOIN continents ON countries.Continent = continents.ContId WHERE continents.Continent = 'Europe'"}
    {'input': [{'role': 'system', 'content': 'TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]'}, {'role': 'user', 'content': 'What is the average horsepower for cars made in the USA?'}], 'ideal': "SELECT AVG(cars_data.Horsepower) FROM cars_data JOIN car_names ON cars_data.Id = car_names.MakeId JOIN car_makers ON car_names.MakeId = car_makers.Id JOIN countries ON car_makers.Country = countries.CountryId WHERE countries.CountryName = 'USA'"}
    

Next we need to create the eval registry to run it in the framework.

The evals framework requires a `.yaml` file structured with the following properties:
* `id` - An identifier for your eval
* `description` - A short description of your eval
* `disclaimer` - An additional notes about your eval
* `metrics` - There are three types of eval metrics we can choose from: match, includes, fuzzyMatch

For our eval, we will configure the following:


```python
"""
spider-sql:
  id: spider-sql.dev.v0
  metrics: [accuracy]
  description: Eval that scores SQL code from 194 examples in the Spider Text-to-SQL test dataset. The problems are selected by taking the first 10 problems for each database that appears in the test set.
    Yu, Tao, et al. \"Spider; A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.\" Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, https://doi.org/10.18653/v1/d18-1425.
  disclaimer: Problems are solved zero-shot with no prompting other than the schema; performance may improve with training examples, fine tuning, or a different schema format. Evaluation is currently done through model-grading, where SQL code is not actually executed; the model may judge correct SQL to be incorrect, or vice-versa.
spider-sql.dev.v0:
  class: evals.elsuite.modelgraded.classify:ModelBasedClassify
  args:
    samples_jsonl: sql/spider_sql.jsonl
    eval_type: cot_classify
    modelgraded_spec: sql
  """""
```




    '\nspider-sql:\n  id: spider-sql.dev.v0\n  metrics: [accuracy]\n  description: Eval that scores SQL code from 194 examples in the Spider Text-to-SQL test dataset. The problems are selected by taking the first 10 problems for each database that appears in the test set.\n    Yu, Tao, et al. "Spider; A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, https://doi.org/10.18653/v1/d18-1425.\n  disclaimer: Problems are solved zero-shot with no prompting other than the schema; performance may improve with training examples, fine tuning, or a different schema format. Evaluation is currently done through model-grading, where SQL code is not actually executed; the model may judge correct SQL to be incorrect, or vice-versa.\nspider-sql.dev.v0:\n  class: evals.elsuite.modelgraded.classify:ModelBasedClassify\n  args:\n    samples_jsonl: sql/spider_sql.jsonl\n    eval_type: cot_classify\n    modelgraded_spec: sql\n  '



## Running an evaluation

We can run this eval using the `oaieval` CLI. To get setup, install the library: `pip install .` (if you are running the [OpenAI Evals library](github.com/openai/evals) locally) or `pip install oaieval` if you are running an existing eval.

Then, run the eval using the CLI: `oaieval gpt-3.5-turbo spider-sql`

This command expects a model name and an eval set name. Note that we provide two command line interfaces (CLIs): `oaieval` for running a single eval and `oaievalset` for running a set of evals. The valid eval names are specified in the YAML files under `evals/registry/evals` and their corresponding implementations can be found in `evals/elsuite`.


```python
!pip install evals --quiet
```

The `oaieval` CLI can accept various flags to modify the default behavior. You can run `oaieval --help` to see a full list of CLI options. 

After running that command, you’ll see the final report of accuracy printed to the console, as well as a file path to a temporary file that contains the full report.

`oaieval` will search for the `spider-sql` eval YAML file in the `evals/registry/evals` directory, following the format specified in cell 4 above. The path to the eval dataset is specified in the eval YAML file under the args: parameter as `samples_jsonl: sql/spider_sql.jsonl`, with the file content in JSONL format (as generated in step 3 above).

After running that command, you’ll see the final report of accuracy printed to the console, as well as a file path to a temporary file that contains the full report.


```python
!oaieval gpt-3.5-turbo spider-sql --max_samples 25
```

    [2024-03-26 19:44:39,836] [registry.py:257] Loading registry from /Users/shyamal/.virtualenvs/openai/lib/python3.11/site-packages/evals/registry/evals
    [2024-03-26 19:44:43,623] [registry.py:257] Loading registry from /Users/shyamal/.evals/evals
    [2024-03-26 19:44:43,635] [oaieval.py:189] [1;35mRun started: 240327024443FACXGMKA[0m
    [2024-03-26 19:44:43,663] [registry.py:257] Loading registry from /Users/shyamal/.virtualenvs/openai/lib/python3.11/site-packages/evals/registry/modelgraded
    [2024-03-26 19:44:43,851] [registry.py:257] Loading registry from /Users/shyamal/.evals/modelgraded
    [2024-03-26 19:44:43,853] [data.py:90] Fetching /Users/shyamal/.virtualenvs/openai/lib/python3.11/site-packages/evals/registry/data/sql/spider_sql.jsonl
    [2024-03-26 19:44:43,878] [eval.py:36] Evaluating 25 samples
    [2024-03-26 19:44:43,952] [eval.py:144] Running in threaded mode with 10 threads!
      0%|                                                    | 0/25 [00:00<?, ?it/s][2024-03-26 19:44:44,810] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:44,829] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:44,991] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:45,090] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:45,145] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:45,971] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:46,040] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:46,069] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:46,378] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:46,587] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:47,412] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
      4%|█▊                                          | 1/25 [00:03<01:23,  3.46s/it][2024-03-26 19:44:47,714] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
      8%|███▌                                        | 2/25 [00:03<00:36,  1.60s/it][2024-03-26 19:44:47,947] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     12%|█████▎                                      | 3/25 [00:03<00:21,  1.02it/s][2024-03-26 19:44:48,413] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:48,643] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     16%|███████                                     | 4/25 [00:04<00:18,  1.15it/s][2024-03-26 19:44:48,909] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     20%|████████▊                                   | 5/25 [00:04<00:12,  1.54it/s][2024-03-26 19:44:49,131] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:49,500] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:49,530] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     24%|██████████▌                                 | 6/25 [00:05<00:12,  1.56it/s][2024-03-26 19:44:49,962] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:49,964] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:49,967] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     28%|████████████▎                               | 7/25 [00:06<00:10,  1.73it/s][2024-03-26 19:44:50,577] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:50,602] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:50,634] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:50,862] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:51,503] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:51,608] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     40%|█████████████████▏                         | 10/25 [00:07<00:08,  1.79it/s][2024-03-26 19:44:51,801] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     44%|██████████████████▉                        | 11/25 [00:07<00:06,  2.09it/s][2024-03-26 19:44:51,856] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:51,969] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:52,227] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     52%|██████████████████████▎                    | 13/25 [00:08<00:04,  2.65it/s][2024-03-26 19:44:52,450] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:52,526] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:52,615] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     56%|████████████████████████                   | 14/25 [00:08<00:04,  2.64it/s][2024-03-26 19:44:52,625] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:52,777] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:53,653] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     60%|█████████████████████████▊                 | 15/25 [00:09<00:05,  1.87it/s][2024-03-26 19:44:53,670] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:54,028] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     68%|█████████████████████████████▏             | 17/25 [00:10<00:03,  2.54it/s][2024-03-26 19:44:54,388] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:54,396] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     72%|██████████████████████████████▉            | 18/25 [00:10<00:02,  2.58it/s][2024-03-26 19:44:54,529] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    [2024-03-26 19:44:54,585] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     76%|████████████████████████████████▋          | 19/25 [00:10<00:02,  2.94it/s][2024-03-26 19:44:54,980] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     80%|██████████████████████████████████▍        | 20/25 [00:11<00:01,  2.82it/s][2024-03-26 19:44:55,152] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     84%|████████████████████████████████████       | 21/25 [00:11<00:01,  3.27it/s][2024-03-26 19:44:56,420] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     88%|█████████████████████████████████████▊     | 22/25 [00:12<00:01,  1.75it/s][2024-03-26 19:44:56,984] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     92%|███████████████████████████████████████▌   | 23/25 [00:13<00:01,  1.76it/s][2024-03-26 19:44:57,370] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
     96%|█████████████████████████████████████████▎ | 24/25 [00:13<00:00,  1.94it/s][2024-03-26 19:44:59,589] [_client.py:1026] HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
    100%|███████████████████████████████████████████| 25/25 [00:15<00:00,  1.60it/s]
    [2024-03-26 19:44:59,607] [record.py:360] Final report: {'counts/Correct': 20, 'counts/Incorrect': 5, 'score': 0.8}. Logged to /tmp/evallogs/240327024443FACXGMKA_gpt-3.5-turbo_spider-sql.jsonl
    [2024-03-26 19:44:59,608] [oaieval.py:229] Final report:
    [2024-03-26 19:44:59,608] [oaieval.py:231] counts/Correct: 20
    [2024-03-26 19:44:59,608] [oaieval.py:231] counts/Incorrect: 5
    [2024-03-26 19:44:59,608] [oaieval.py:231] score: 0.8
    [2024-03-26 19:44:59,640] [record.py:349] Logged 75 rows of events to /tmp/evallogs/240327024443FACXGMKA_gpt-3.5-turbo_spider-sql.jsonl: insert_time=27.915ms
    

`oaievalset` expects a model name and an eval set name, for which the valid options are specified in the YAML files under `evals/registry/eval_sets`.

### Going through eval logs

The eval logs are located at `/tmp/evallogs` and different log files are created for each evaluation run. 


```python
log_name = '240327024443FACXGMKA_gpt-3.5-turbo_spider-sql.jsonl' # "EDIT THIS" - copy from above
events = f"/tmp/evallogs/{log_name}"
display(pd.read_json(events, lines=True).head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spec</th>
      <th>final_report</th>
      <th>run_id</th>
      <th>event_id</th>
      <th>sample_id</th>
      <th>type</th>
      <th>data</th>
      <th>created_by</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{'completion_fns': ['gpt-3.5-turbo'], 'eval_name': 'spider-sql.dev.v0', 'base_eval': 'spider-sql', 'split': 'dev', 'run_config': {'completion_fns': ['gpt-3.5-turbo'], 'eval_spec': {'cls': 'evals.elsuite.modelgraded.classify:ModelBasedClassify', 'registry_path': '/Users/shyamal/.virtualenvs/openai/lib/python3.11/site-packages/evals/registry', 'args': {'samples_jsonl': 'sql/spider_sql.jsonl', 'eval_type': 'cot_classify', 'modelgraded_spec': 'sql'}, 'key': 'spider-sql.dev.v0', 'group': 'sql'}, 'seed': 20220722, 'max_samples': 25, 'command': '/Users/shyamal/.virtualenvs/openai/bin/oaieval gpt-3.5-turbo spider-sql --max_samples 25', 'initial_settings': {'visible': False}}, 'created_by': '', 'run_id': '240327024443FACXGMKA', 'created_at': '2024-03-27 02:44:43.626043'}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>{'counts/Correct': 20, 'counts/Incorrect': 5, 'score': 0.8}</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>240327024443FACXGMKA</td>
      <td>0.0</td>
      <td>spider-sql.dev.88</td>
      <td>sampling</td>
      <td>{'prompt': [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.
Use only the following tables and columns:
Table: players. Columns: player_id (number), first_name (text), last_name (text), hand (text), birth_date (time), country_code (text)
Table: matches. Columns: best_of (number), draw_size (number), loser_age (number), loser_entry (text), loser_hand (text), loser_ht (number), loser_id (number), loser_ioc (text), loser_name (text), loser_rank (number), loser_rank_points (number), loser_seed (number), match_num (number), minutes (number), round (text), score (text), surface (text), tourney_date (time), tourney_id (text), tourney_level (text), tourney_name (text), winner_age (number), winner_entry (text), winner_hand (text), winner_ht (number), winner_id (number), winner_ioc (text), winner_name (text), winner_rank (number), winner_rank_points (number), winner_seed (number), year (number)
Table: rankings. Columns: ranking_date (time), ranking (number), player_id (number), ranking_points (number), tours (number)

Question: Find the average rank of winners in all matches.
', 'role': 'system'}], 'sampled': ['SELECT AVG(winner_rank) AS average_rank_of_winners
FROM matches;']}</td>
      <td></td>
      <td>2024-03-27 02:44:44.821110+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>240327024443FACXGMKA</td>
      <td>1.0</td>
      <td>spider-sql.dev.82</td>
      <td>sampling</td>
      <td>{'prompt': [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.
Use only the following tables and columns:
Table: players. Columns: player_id (number), first_name (text), last_name (text), hand (text), birth_date (time), country_code (text)
Table: matches. Columns: best_of (number), draw_size (number), loser_age (number), loser_entry (text), loser_hand (text), loser_ht (number), loser_id (number), loser_ioc (text), loser_name (text), loser_rank (number), loser_rank_points (number), loser_seed (number), match_num (number), minutes (number), round (text), score (text), surface (text), tourney_date (time), tourney_id (text), tourney_level (text), tourney_name (text), winner_age (number), winner_entry (text), winner_hand (text), winner_ht (number), winner_id (number), winner_ioc (text), winner_name (text), winner_rank (number), winner_rank_points (number), winner_seed (number), year (number)
Table: rankings. Columns: ranking_date (time), ranking (number), player_id (number), ranking_points (number), tours (number)

Question: Find the total number of matches.
', 'role': 'system'}], 'sampled': ['SELECT COUNT(*) AS total_matches
FROM matches;']}</td>
      <td></td>
      <td>2024-03-27 02:44:44.831848+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>240327024443FACXGMKA</td>
      <td>2.0</td>
      <td>spider-sql.dev.25</td>
      <td>sampling</td>
      <td>{'prompt': [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.
Use only the following tables and columns:
Table: continents. Columns: ContId (number), Continent (text)
Table: countries. Columns: CountryId (number), CountryName (text), Continent (number)
Table: car_makers. Columns: Id (number), Maker (text), FullName (text), Country (text)
Table: model_list. Columns: ModelId (number), Maker (number), Model (text)
Table: car_names. Columns: MakeId (number), Model (text), Make (text)
Table: cars_data. Columns: Id (number), MPG (text), Cylinders (number), Edispl (number), Horsepower (text), Weight (number), Accelerate (number), Year (number)

Question: How many countries exist?
', 'role': 'system'}], 'sampled': ['SELECT COUNT(*) AS TotalCountries
FROM countries;']}</td>
      <td></td>
      <td>2024-03-27 02:44:44.996647+00:00</td>
    </tr>
  </tbody>
</table>
</div>



```python
# processing the log events generated by oaieval

with open(events, "r") as f:
    events_df = pd.read_json(f, lines=True)
```

This file will contain structured logs of the evaluation. The first entry provides a detailed specification of the evaluation, including the completion functions, evaluation name, run configuration, creator’s name, run ID, and creation timestamp.


```python
display(events_df.iloc[0].spec)
```


    {'completion_fns': ['gpt-3.5-turbo'],
     'eval_name': 'spider-sql.dev.v0',
     'base_eval': 'spider-sql',
     'split': 'dev',
     'run_config': {'completion_fns': ['gpt-3.5-turbo'],
      'eval_spec': {'cls': 'evals.elsuite.modelgraded.classify:ModelBasedClassify',
       'registry_path': '/Users/shyamal/.virtualenvs/openai/lib/python3.11/site-packages/evals/registry',
       'args': {'samples_jsonl': 'sql/spider_sql.jsonl',
        'eval_type': 'cot_classify',
        'modelgraded_spec': 'sql'},
       'key': 'spider-sql.dev.v0',
       'group': 'sql'},
      'seed': 20220722,
      'max_samples': 25,
      'command': '/Users/shyamal/.virtualenvs/openai/bin/oaieval gpt-3.5-turbo spider-sql --max_samples 25',
      'initial_settings': {'visible': False}},
     'created_by': '',
     'run_id': '240327024443FACXGMKA',
     'created_at': '2024-03-27 02:44:43.626043'}


Let's also look at the entry which provides the final report of the evaluation.


```python
display(events_df.dropna(subset=['final_report']).iloc[0]['final_report'])
```


    {'counts/Correct': 20, 'counts/Incorrect': 5, 'score': 0.8}


We can also review individual evaluation events that provide specific samples (`sample_id`), results, event types, and metadata.


```python
pd.set_option('display.max_colwidth', None)  # None means no truncation
display(events_df.iloc[2][['run_id', 'event_id', 'sample_id', 'type', 'data', 'created_at']])
```


    run_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                240327024443FACXGMKA
    event_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0.0
    sample_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                spider-sql.dev.88
    type                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              sampling
    data          {'prompt': [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.
    Use only the following tables and columns:
    Table: players. Columns: player_id (number), first_name (text), last_name (text), hand (text), birth_date (time), country_code (text)
    Table: matches. Columns: best_of (number), draw_size (number), loser_age (number), loser_entry (text), loser_hand (text), loser_ht (number), loser_id (number), loser_ioc (text), loser_name (text), loser_rank (number), loser_rank_points (number), loser_seed (number), match_num (number), minutes (number), round (text), score (text), surface (text), tourney_date (time), tourney_id (text), tourney_level (text), tourney_name (text), winner_age (number), winner_entry (text), winner_hand (text), winner_ht (number), winner_id (number), winner_ioc (text), winner_name (text), winner_rank (number), winner_rank_points (number), winner_seed (number), year (number)
    Table: rankings. Columns: ranking_date (time), ranking (number), player_id (number), ranking_points (number), tours (number)
    
    Question: Find the average rank of winners in all matches.
    ', 'role': 'system'}], 'sampled': ['SELECT AVG(winner_rank) AS average_rank_of_winners
    FROM matches;']}
    created_at                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2024-03-27 02:44:44.821110+00:00
    Name: 2, dtype: object



```python
# Inspect samples
for i, row in events_df[events_df['type'] == 'sampling'].head(5).iterrows():
    data = pd.json_normalize(row['data'])
    print(f"Prompt: {data['prompt'].iloc[0]}")
    print(f"Sampled: {data['sampled'].iloc[0]}")
    print("-" * 10)
```

    Prompt: [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.\nUse only the following tables and columns:\nTable: players. Columns: player_id (number), first_name (text), last_name (text), hand (text), birth_date (time), country_code (text)\nTable: matches. Columns: best_of (number), draw_size (number), loser_age (number), loser_entry (text), loser_hand (text), loser_ht (number), loser_id (number), loser_ioc (text), loser_name (text), loser_rank (number), loser_rank_points (number), loser_seed (number), match_num (number), minutes (number), round (text), score (text), surface (text), tourney_date (time), tourney_id (text), tourney_level (text), tourney_name (text), winner_age (number), winner_entry (text), winner_hand (text), winner_ht (number), winner_id (number), winner_ioc (text), winner_name (text), winner_rank (number), winner_rank_points (number), winner_seed (number), year (number)\nTable: rankings. Columns: ranking_date (time), ranking (number), player_id (number), ranking_points (number), tours (number)\n\nQuestion: Find the average rank of winners in all matches.\n', 'role': 'system'}]
    Sampled: ['SELECT AVG(winner_rank) AS average_rank_of_winners\nFROM matches;']
    ----------
    Prompt: [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.\nUse only the following tables and columns:\nTable: players. Columns: player_id (number), first_name (text), last_name (text), hand (text), birth_date (time), country_code (text)\nTable: matches. Columns: best_of (number), draw_size (number), loser_age (number), loser_entry (text), loser_hand (text), loser_ht (number), loser_id (number), loser_ioc (text), loser_name (text), loser_rank (number), loser_rank_points (number), loser_seed (number), match_num (number), minutes (number), round (text), score (text), surface (text), tourney_date (time), tourney_id (text), tourney_level (text), tourney_name (text), winner_age (number), winner_entry (text), winner_hand (text), winner_ht (number), winner_id (number), winner_ioc (text), winner_name (text), winner_rank (number), winner_rank_points (number), winner_seed (number), year (number)\nTable: rankings. Columns: ranking_date (time), ranking (number), player_id (number), ranking_points (number), tours (number)\n\nQuestion: Find the total number of matches.\n', 'role': 'system'}]
    Sampled: ['SELECT COUNT(*) AS total_matches\nFROM matches;']
    ----------
    Prompt: [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.\nUse only the following tables and columns:\nTable: continents. Columns: ContId (number), Continent (text)\nTable: countries. Columns: CountryId (number), CountryName (text), Continent (number)\nTable: car_makers. Columns: Id (number), Maker (text), FullName (text), Country (text)\nTable: model_list. Columns: ModelId (number), Maker (number), Model (text)\nTable: car_names. Columns: MakeId (number), Model (text), Make (text)\nTable: cars_data. Columns: Id (number), MPG (text), Cylinders (number), Edispl (number), Horsepower (text), Weight (number), Accelerate (number), Year (number)\n\nQuestion: How many countries exist?\n', 'role': 'system'}]
    Sampled: ['SELECT COUNT(*) AS TotalCountries\nFROM countries;']
    ----------
    Prompt: [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.\nUse only the following tables and columns:\nTable: TV_Channel. Columns: id (text), series_name (text), Country (text), Language (text), Content (text), Pixel_aspect_ratio_PAR (text), Hight_definition_TV (text), Pay_per_view_PPV (text), Package_Option (text)\nTable: TV_series. Columns: id (number), Episode (text), Air_Date (text), Rating (text), Share (number), 18_49_Rating_Share (text), Viewers_m (text), Weekly_Rank (number), Channel (text)\nTable: Cartoon. Columns: id (number), Title (text), Directed_by (text), Written_by (text), Original_air_date (text), Production_code (number), Channel (text)\n\nQuestion: What is the name and directors of all the cartoons that are ordered by air date?\n', 'role': 'system'}]
    Sampled: ['SELECT Title, Directed_by\nFROM Cartoon\nORDER BY Original_air_date;']
    ----------
    Prompt: [{'content': 'Answer the following question with syntactically correct SQLite SQL. Be creative but the SQL must be correct.\nUse only the following tables and columns:\nTable: stadium. Columns: Stadium_ID (number), Location (text), Name (text), Capacity (number), Highest (number), Lowest (number), Average (number)\nTable: singer. Columns: Singer_ID (number), Name (text), Country (text), Song_Name (text), Song_release_year (text), Age (number), Is_male (others)\nTable: concert. Columns: concert_ID (number), concert_Name (text), Theme (text), Stadium_ID (text), Year (text)\nTable: singer_in_concert. Columns: concert_ID (number), Singer_ID (text)\n\nQuestion: Show the name and the release year of the song by the youngest singer.\n', 'role': 'system'}]
    Sampled: ['```sql\nSELECT s.Name, s.Song_release_year\nFROM singer s\nWHERE s.Age = (SELECT MIN(Age) FROM singer)\n```']
    ----------
    

Let's review our failures to understand which tests did not succeed.


```python
def pretty_print_text(prompt):
    # Define markers for the sections
    markers = {
        "question": "[Question]:",
        "expert": "[Expert]:",
        "submission": "[Submission]:",
        "end": "[END DATA]"
    }
    
    # Function to extract text between markers
    def extract_text(start_marker, end_marker):
        start = prompt.find(start_marker) + len(start_marker)
        end = prompt.find(end_marker)
        text = prompt[start:end].strip()
        if start_marker == markers["question"]:
            text = text.split("\n\nQuestion:")[-1].strip() if "\n\nQuestion:" in text else text
        elif start_marker == markers["submission"]:
            text = text.replace("```sql", "").replace("```", "").strip()
        return text
    
    # Extracting text for each section
    question_text = extract_text(markers["question"], markers["expert"])
    expert_text = extract_text(markers["expert"], markers["submission"])
    submission_text = extract_text(markers["submission"], markers["end"])
    
    # HTML color codes and formatting
    colors = {
        "question": '<span style="color: #0000FF;">QUESTION:<br>', 
        "expert": '<span style="color: #008000;">EXPECTED:<br>',  
        "submission": '<span style="color: #FFA500;">SUBMISSION:<br>' 
    }
    color_end = '</span>'
    
    # Display each section with color
    from IPython.display import display, HTML
    display(HTML(f"{colors['question']}{question_text}{color_end}"))
    display(HTML(f"{colors['expert']}{expert_text}{color_end}"))
    display(HTML(f"{colors['submission']}{submission_text}{color_end}"))
```


```python
# Inspect metrics where choice is made and print only the prompt, result, and expected result if the choice is incorrect
for i, row in events_df[events_df['type'] == 'metrics'].iterrows():
    if row['data']['choice'] == 'Incorrect':
        # Get the previous row's data, which contains the prompt and the expected result
        prev_row = events_df.iloc[i-1]
        prompt = prev_row['data']['prompt'][0]['content'] if 'prompt' in prev_row['data'] and len(prev_row['data']['prompt']) > 0 else "Prompt not available"
        expected_result = prev_row['data'].get('ideal', 'Expected result not provided')
        
        # Current row's data will be the actual result
        result = row['data'].get('result', 'Actual result not provided')
        
        pretty_print_text(prompt)
        print("-" * 40)
```


<span style="color: #0000FF;">QUESTION:<br>How many countries have a republic as their form of government?

************</span>



<span style="color: #008000;">EXPECTED:<br>SELECT count(*) FROM country WHERE GovernmentForm  =  "Republic"
************</span>



<span style="color: #FFA500;">SUBMISSION:<br>SELECT COUNT(*) 
FROM country 
WHERE GovernmentForm LIKE '%Republic%'

************</span>


    ----------------------------------------
    


<span style="color: #0000FF;">QUESTION:<br>Return the document id, template id, and description for the document with the name Robbin CV.

************</span>



<span style="color: #008000;">EXPECTED:<br>SELECT document_id ,  template_id ,  Document_Description FROM Documents WHERE document_name  =  "Robbin CV"
************</span>



<span style="color: #FFA500;">SUBMISSION:<br>SELECT Documents.Document_ID, Documents.Template_ID, Documents.Document_Description
FROM Documents
JOIN Templates ON Documents.Template_ID = Templates.Template_ID
WHERE Documents.Document_Name = 'Robbin CV';

************</span>


    ----------------------------------------
    


<span style="color: #0000FF;">QUESTION:<br>Which professionals live in the state of Indiana or have done treatment on more than 2 treatments? List his or her id, last name and cell phone.

************</span>



<span style="color: #008000;">EXPECTED:<br>SELECT professional_id ,  last_name ,  cell_number FROM Professionals WHERE state  =  'Indiana' UNION SELECT T1.professional_id ,  T1.last_name ,  T1.cell_number FROM Professionals AS T1 JOIN Treatments AS T2 ON T1.professional_id  =  T2.professional_id GROUP BY T1.professional_id HAVING count(*)  >  2
************</span>



<span style="color: #FFA500;">SUBMISSION:<br>SELECT professional_id, last_name, cell_number
FROM Professionals
WHERE state = 'Indiana'
OR professional_id IN (
    SELECT professional_id
    FROM Treatments
    GROUP BY professional_id
    HAVING COUNT(*) > 2
);
************</span>


    ----------------------------------------
    


<span style="color: #0000FF;">QUESTION:<br>What is the continent name which Anguilla belongs to?

************</span>



<span style="color: #008000;">EXPECTED:<br>SELECT Continent FROM country WHERE Name  =  "Anguilla"
************</span>



<span style="color: #FFA500;">SUBMISSION:<br>SELECT c.Continent
FROM country c
WHERE c.Code = 'AIA';

************</span>


    ----------------------------------------
    


<span style="color: #0000FF;">QUESTION:<br>How many airlines do we have?

************</span>



<span style="color: #008000;">EXPECTED:<br>SELECT count(*) FROM AIRLINES
************</span>



<span style="color: #FFA500;">SUBMISSION:<br>SELECT COUNT(DISTINCT Airline) AS TotalAirlines
FROM airlines;
************</span>


    ----------------------------------------
    

Reviewing some of the failures we see the following:
* The second incorrect answer had an unnecessary join with the 'Templates' table. Our eval was able to accurately identify this and flag this as incorrect. 
* Few other answers have minor syntax differences that caused the answers to get flagged.
  * In situations like this, it would be worthwhile exploring whether we should continue iterating on the prompt to ensure certain stylistic choices, or if we should modify the evaluation suite to capture this variation.
  * This type of failure hints at the potential need for model-graded evals as a way to ensure accuracy in grading the results

# Conclusion

Building out effective evals is a core part of the development cycle of LLM-based applications. The OpenAI Evals framework provides the core structure of building evals out of the box, and allows you to quickly spin up new tests for your various use cases. In this guide, we demonstrated step-by-step how to create an eval, run it, and analyze the results.

The example shown in this guide represent a straightfoward use case for evals. As you continue to explore this framework, we recommend you explore creating more complex model-graded evals for actual production use cases. Happy evaluating!
