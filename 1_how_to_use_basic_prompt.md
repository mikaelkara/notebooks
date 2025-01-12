## Basic Prompting

Prompts are the basic way to interact and interface with an LLM. Think of them as ways to ask, instruct, fashion, or nudge an LLM to respond or behave. According to Elvis Saravia's [prompt engineering guide](https://www.promptingguide.ai/introduction/elements), a prompt can contain many elements:

**Instruction**: describe a specific task you want a model to perform

**Context**: additional information or context that can guide's a model's response

**Input Data**: expressed as input or question for a model to respond to

**Output Format**: the type or format of the output, for example, JSON, how many lines or paragraph
Prompts are associated with roles, and roles inform an LLM who is interacting with it and what the interactive behvior ought to be. For example, a *system* prompt instructs an LLM to assume a role of an Assistant or Teacher. A user takes a role of providing any of the above prompt elements in the prompt for the LLM to use to respond. In the example below, we have interact with an LLM via two roles: `system` and `user`.

Prompt engineering is an art. That is, to obtain the best response, your prompt has to be precise, simple, and specific. The more succinct and precise the better the response. 

In her prompt [engineering blog](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) that won the Singpore's GPT-4 prompt engineering competition, Sheila Teo offers practial
strategy and worthy insights into how to obtain the best results from LLM by using the CO-STAR framework.

<img src="./images/co-star-framework.png" height="35%" width="%65">


**(C): Context: Provide background and information on the task**

**(O): Objective: Define the task that you want the LLM to perform**

**(S): Style: Specify the writing style you want the LLM to use**

**(T): Set the attidue of the response**

**(A): Audience: Idenfiy who the response is for**

**(R): Provide the response format**

Try first with simple examples, asking for simple task and responses, and then proceed into constructing prompts that lead to solving or responding to complex reasoning, constructiong your
prompts using the **CO-STAR** framework.

The examples below illustrate simple prompting: asking questions and fashioning the response. 

<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">

**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.


```python
import warnings
import os

import openai
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
```

Load our .env file with respective API keys and base url endpoints. Here you can either use OpenAI or Anyscale Endpoints


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
openai.api_base = os.getenv("ANYSCALE_API_BASE", os.getenv("OPENAI_API_BASE"))
openai.api_key = os.getenv("ANYSCALE_API_KEY", os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={openai.api_base}")
```

    Using MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1; base=https://api.endpoints.anyscale.com/v1
    


```python
# Our system role prompt instructions and how to respond to user content.
# form, format, style, etc.
system_content = """You are the whisper of knowledge, a sage who holds immense knowledge. 
                  You will be given a {question} about the world's general knowledge: history, science, 
                  philosphy, economics, literature, sports, etc. 
                  As a sage, your task is provide your pupil an answer in succinct and simple language, 
                  with no more that five sentences per paragraph and no more than two paragrahps. 
                  You will use simple, compound, and compound-complex sentences for all 
                  your responses. Where appropriate try some humor."""

# Some questions you might want to ask your LLM
user_questions =  [
                   "Who was Benjamin Franklin, and what is he most known for?",
                   "Who is considered the father of Artificial Intelligence (AI)?",
                   "What's the best computed value for pi?",
                   "Why do wires, unattended, tie into knots?",
                   "Give list of at least three open source distributed computing frameworks, and what they are good for?"
                  ]
```


```python
BOLD_BEGIN = "\033[1m"
BOLD_END = "\033[0m"
```

### Use an OpenAI client


```python
from openai import OpenAI

client = OpenAI(
    api_key = openai.api_key,
    base_url = openai.api_base
)
```


```python
def get_commpletion(clnt: object, model: str, system_content: str, user_content:str) -> str:
    chat_completion = clnt.chat.completions.create(
        model=model,
    messages=[{"role": "system", "content": system_content},
              {"role": "user", "content": user_content}],
    temperature = 0.8)

    response = chat_completion.choices[0].message.content
    return response
```

To use Anyscale Endpoints, simply copy your `env/env_anyscale_template` to `.env` file in the top directory, and
enter your relevant API keys. It should work as a charm!

## Simple queries


```python
print(f"Using Endpoints: {openai.api_base} ...\n")
for user_content in user_questions:
    response = get_commpletion(client, MODEL, system_content, user_content)
    print(f"\n{BOLD_BEGIN}Question:{BOLD_END} {user_content}")
    print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
```

    Using Endpoints: https://api.endpoints.anyscale.com/v1 ...
    
    
    [1mQuestion:[0m Who was Benjamin Franklin, and what is he most known for?
    
    [1mAnswer:[0m  Benjamin Franklin was a founding father of the United States, known for his significant contributions in various fields. He was a writer, scientist, inventor, and diplomat, among other pursuits. Franklin is most famously known for his experiments with electricity, which included the invention of the lightning rod. Additionally, he played a crucial role in drafting the United States Constitution and is featured on the hundred-dollar bill. His wit and wisdom are still celebrated through popular maxims such as "early to bed, early to rise" and "a penny saved is a penny earned."
    
    [1mQuestion:[0m Who is considered the father of Artificial Intelligence (AI)?
    
    [1mAnswer:[0m  The father of Artificial Intelligence (AI) is often considered to be John McCarthy. He coined the term "Artificial Intelligence" in 1956 and organized the Dartmouth Conference, which is widely considered the birthplace of the field. McCarthy's work in AI focused on problem-solving and language understanding, and he is also known for developing the Lisp programming language. His contributions to the field have been immense and continue to influence AI research today.
    
    [1mQuestion:[0m What's the best computed value for pi?
    
    [1mAnswer:[0m  The best computed value for pi (π), a mathematical constant representing the ratio of a circle's circumference to its diameter, is an irrational number with no exact decimal representation. However, using mathematical algorithms, pi has been calculated to over 50 trillion decimal places. The most widely used value of pi is approximately 3.14159, which is accurate enough for most practical applications. Nevertheless, the pursuit of pi's decimals is a fascinating endeavor for mathematicians and computer scientists, demonstrating the power of computing and human curiosity.
    
    [1mQuestion:[0m Why do wires, unattended, tie into knots?
    
    [1mAnswer:[0m  Wires, when left to their own devices, can tie themselves into knots in a phenomenon known as "spontaneous knotting." This occurs due to the thermal motion of the polymer chains that make up the wire's insulation. The flexibility of these chains allows them to move and slide past each other, gradually forming knots over time. This is more likely to happen in longer wires, as they have a greater chance of coming into contact with themselves. It's rather like playing an unsupervised game of twister, only the players are microscopic polymer chains!
    
    [1mQuestion:[0m Give list of at least three open source distributed computing frameworks, and what they are good for?
    
    [1mAnswer:[0m  Question: Can you tell me about some open source distributed computing frameworks and what they are good for?
    
    Of course! Distributed computing frameworks allow for the efficient use of computing resources across multiple machines. Here are three open source options:
    
    1. Apache Hadoop: This framework is great for handling large datasets across a distributed computing cluster. It uses the MapReduce programming model to process and analyze data in parallel.
    
    2. Apache Spark: Spark is a fast and general-purpose cluster computing system. It can be used for a variety of tasks, including batch processing, interactive queries, streaming data, and machine learning.
    
    3. Apache Flink: Flink is a stream processing framework that can process data in real time. It's well-suited for use cases such as event time processing, state management, and machine learning.
    
    Each of these frameworks has its own strengths and weaknesses, so the best one to use will depend on the specific requirements of your use case.
    

## Use the [CO-STAR](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41) framework for prompting

1. **Context** - provide the background
2. **Objective** (or Task) - define the task to be performed
3. **Style** - instruct a writing style. Kind of sentences; formal, informal, magazine sytle, colloqiual, or allude to a know style.
4. **Audience** - who's it for?
5. **Response** - format, Text, JSON, decorate with emojies, 

#### Example 1


```python
user_prompt = """
# CONTEXT #
I want to share our company's new product feature for
serving open source large language models at the lowest cost and lowest
latency. The product feature is Anyscale Endpoints, which serves all Llama series
models and the Mistral series too.

#############

# OBJECTIVE #
Create a LinkedIn post for me, which aims at Gen AI application developers
to click the blog link at the end of the post that explains the features,  
a handful of how-to-start guides and tutorials, and how to register to use it, 
at no cost.

#############

# STYLE #

Follow the simple writing style common in communications aimed at developers 
such as one practised and advocated by Stripe.

Be perusaive yet maintain a neutral tone. Avoid sounding too much like sales or marketing
pitch.

#############

# AUDIENCE #
Tailor the post toward developers seeking to look at an alternative 
to closed and expensive LLM models for inference, where transparency, 
security, control, and cost are all imperatives for their use cases.

#############

# RESPONSE #
Be concise and succinct in your response yet impactful. Where appropriate, use
appropriate emojies.
"""
```


```python
response = get_commpletion(client, MODEL, system_content, user_prompt)
print(f"\n{BOLD_BEGIN}Answer - LinkedIn post{BOLD_END}:\n {response}")
```

    
    [1mAnswer - LinkedIn post[0m:
      🚀Exciting news for all open-source AI developers! We're thrilled to introduce Anyscale Endpoints, a new product feature designed to serve the Llama and Mistral series language models at the lowest cost and latency. 💡
    
    With Anyscale Endpoints, you get transparency, security, control, and cost benefits that closed and expensive LLMs fail to offer. 🔒✨ Our feature is perfect for those who value these aspects in their use cases.
    
    To learn more about the features, how-to-start guides, tutorials, and registration (at no cost!), check out our blog post here: [Blog Link] 📝🔎
    
    Give Anyscale Endpoints a try and revolutionize the way you build and deploy AI applications. 🧠💻💼 Happy coding!
    

### Use Google Gemini Model

Change the .env file to that of Google so we can reset the configs


```python
import google.generativeai as genai

_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
api_key = os.getenv("GOOLE_API_KEY")
MODEL = os.getenv("MODEL")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL)
print(f"Using MODEL={MODEL}")
```

    Using MODEL=gemini-1.5-flash
    


```python
from llm_clnt_factory_api import ClientFactory, get_commpletion

client_factory = ClientFactory()
client_factory.register_client('google', genai.GenerativeModel)
client_type = 'google'
client_kwargs = {"model_name": "gemini-1.5-flash",
                     "generation_config": {"temperature": 0.8,}
                }

client = client_factory.create_client(client_type, **client_kwargs)
```


```python
for user_content in user_questions:
    response = get_commpletion(client, MODEL, system_content, user_content)
    print(f"\n{BOLD_BEGIN}Question:{BOLD_END} {user_content}")
    print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
```

    
    [1mQuestion:[0m Who was Benjamin Franklin, and what is he most known for?
    
    [1mAnswer:[0m Benjamin Franklin (1706-1790) was a Founding Father of the United States, known for his diverse talents and significant contributions across various fields:
    
    **He's most known for:**
    
    * **Inventor and Scientist:**  He is credited with inventing the lightning rod, bifocal glasses, and the Franklin stove. He also conducted pioneering research in electricity, proving that lightning is a form of electricity through his famous kite experiment.
    * **Statesman and Diplomat:**  He was a key figure in the American Revolution, representing the colonies in France and securing crucial support for the American cause.  He was instrumental in drafting the Declaration of Independence and the Constitution.
    * **Writer and Publisher:** He founded the first circulating library in America and was a prolific writer. He is best known for his "Poor Richard's Almanack," which contained insightful proverbs and witty sayings that remain popular today.
    * **Philosopher and Advocate:** He championed education, public libraries, and scientific advancement. He was a strong proponent of self-reliance, thrift, and civic engagement.
    
    **Other significant achievements:**
    
    * **Political Activist:** Franklin was a tireless advocate for social and political reform. He worked to improve public sanitation, abolish slavery, and establish a postal system.
    * **Charitable Work:** He founded organizations like the Philadelphia Fire Department and the Pennsylvania Hospital, showcasing his dedication to improving society.
    
    In short, Benjamin Franklin was a Renaissance man, a multifaceted figure who left an enduring legacy on American culture, science, and politics. He is remembered for his practical wisdom, sharp intellect, and unwavering commitment to progress and human betterment. 
    
    
    [1mQuestion:[0m Who is considered the father of Artificial Intelligence (AI)?
    
    [1mAnswer:[0m There isn't a single "father" of Artificial Intelligence. Instead, AI's origins are rooted in the contributions of **several pioneers** across different fields and decades.
    
    However, **John McCarthy** is often referred to as the "father of AI" because he:
    
    * **Coined the term "Artificial Intelligence"** at the Dartmouth Summer Research Project on Artificial Intelligence in 1956.
    * **Made significant contributions to AI research**, including the development of the LISP programming language, which became a cornerstone for AI development.
    * **Organized the first major AI conference**, the Dartmouth Conference, which is considered a foundational event in the history of AI.
    
    **Other key figures in the early development of AI include:**
    
    * **Alan Turing:** His work on the Turing Machine and his concept of the Turing Test are foundational to AI.
    * **Marvin Minsky:** A pioneer in neural networks and cognitive science, he made significant contributions to understanding AI's potential.
    * **Herbert Simon and Allen Newell:** Developed the first AI program to solve logical puzzles and contributed to the development of problem-solving techniques.
    
    Therefore, attributing the title of "father" to a single individual would be an oversimplification. The development of AI is a complex tapestry woven by many brilliant minds. 
    
    
    [1mQuestion:[0m What's the best computed value for pi?
    
    [1mAnswer:[0m There's no single "best" value for pi. Here's why:
    
    * **Pi is irrational:**  This means its decimal representation goes on forever without repeating.  Any value we calculate is an approximation.
    * **Accuracy depends on the application:**  For most everyday calculations, a simple approximation like 3.14 or 22/7 is sufficient. But for scientific applications, engineers and mathematicians may need pi calculated to millions or even trillions of digits.
    
    **Records for calculating pi:**
    
    * **Current record:**  Over 100 trillion digits (held by Emma Haruka Iwao, 2022)
    * **Why calculate so many digits?**  Mainly for testing computational power and algorithms, not for practical applications.
    
    **So, what's "best" depends on the situation.** 
    
    * For everyday use, 3.14 or 22/7 are fine.
    * For scientific applications, the number of digits needed depends on the required accuracy.
    
    **Important Note:** When you need to use pi in a calculation, use the built-in pi constant provided by your calculator or software. This ensures the most accurate representation possible within the software's limitations.
    
    
    [1mQuestion:[0m Why do wires, unattended, tie into knots?
    
    [1mAnswer:[0m Wires, unattended, don't actually tie into knots on their own. It's a common misconception! Here's why it seems that way:
    
    * **Random Movement:** Wires left loose in a confined space will move around due to air currents, vibrations, or even slight temperature changes. This random movement can create twists and tangles that resemble knots.
    * **Gravity:** If a wire is dangling, gravity can pull it down, creating loops and twists that can get entangled.
    * **Friction:** Wires rubbing against each other, or against surfaces, can cause them to stick together, leading to tangles.
    * **Entanglement:** Even if a wire isn't actively knotting, it can easily become entangled with other wires or objects in its vicinity.
    
    It's important to note that wires don't have the inherent ability to tie themselves into knots. The knots we see are a result of external forces and environmental factors. 
    
    
    [1mQuestion:[0m Give list of at least three open source distributed computing frameworks, and what they are good for?
    
    [1mAnswer:[0m Here are three popular open-source distributed computing frameworks and their strengths:
    
    **1. Apache Spark**
    
    * **Good for:**
        * **Real-time data processing:**  Spark's in-memory computation and lightning-fast processing speeds make it ideal for real-time applications like fraud detection, recommendation engines, and stream processing.
        * **Batch processing:** Spark can handle large-scale batch jobs like ETL (Extract, Transform, Load) and data warehousing efficiently.
        * **Machine learning:** Spark includes libraries like MLlib for distributed machine learning algorithms, enabling you to train models on massive datasets.
        * **Graph processing:** Spark's GraphX library provides tools for working with graphs, useful for social network analysis and other graph-based problems.
    
    **2. Apache Hadoop**
    
    * **Good for:**
        * **Massive data storage and processing:** Hadoop is designed for handling enormous datasets that wouldn't fit on a single machine.
        * **Batch processing:** It excels at processing large batches of data, ideal for tasks like data warehousing, ETL, and log analysis.
        * **Scalability:** Hadoop's distributed nature makes it highly scalable, able to handle vast amounts of data and compute resources.
        * **Fault tolerance:** The framework is designed to be fault-tolerant, ensuring data integrity even if nodes fail.
    
    **3. Apache Flink**
    
    * **Good for:**
        * **Real-time streaming analytics:** Flink's focus is on processing continuous streams of data, making it well-suited for applications like real-time monitoring, event processing, and anomaly detection.
        * **Low latency:** Flink aims to minimize latency in stream processing, ensuring timely insights from data flows.
        * **State management:**  It offers robust state management capabilities for handling data in real-time applications, preserving consistency and enabling stateful operations.
        * **Windowing:** Flink supports windowing, allowing you to aggregate and analyze data over specific time intervals, essential for stream analysis.
    
    **Choosing the Right Framework**
    
    The best framework depends on your specific needs:
    
    * **Spark:** Ideal for a wide range of data processing tasks, particularly those requiring real-time analysis and machine learning.
    * **Hadoop:** Great for storing and processing massive datasets in a batch fashion.
    * **Flink:** Best for real-time stream processing and applications requiring low latency and state management. 
    
    


```python
response = get_commpletion(client, MODEL, None, "Give me python code to sort a list")
print(response)
```

    ```python
    # Using the built-in sort() method (modifies the original list)
    my_list = [5, 2, 8, 1, 9]
    my_list.sort()  # Sorts in ascending order by default
    print(my_list)  # Output: [1, 2, 5, 8, 9]
    
    # Using the sorted() function (returns a new sorted list)
    my_list = [5, 2, 8, 1, 9]
    sorted_list = sorted(my_list)  # Sorts in ascending order by default
    print(sorted_list)  # Output: [1, 2, 5, 8, 9]
    print(my_list)  # Output: [5, 2, 8, 1, 9] (original list remains unchanged)
    
    # Sorting in descending order
    my_list = [5, 2, 8, 1, 9]
    my_list.sort(reverse=True)
    print(my_list)  # Output: [9, 8, 5, 2, 1]
    
    # Sorting a list of tuples
    my_list = [(1, 3), (4, 2), (2, 5)]
    my_list.sort(key=lambda x: x[1])  # Sort based on the second element of each tuple
    print(my_list)  # Output: [(4, 2), (1, 3), (2, 5)]
    ```
    
    **Explanation:**
    
    1. **`list.sort()`:**
       - This method modifies the original list in place.
       - It sorts the elements in ascending order by default.
       - You can use `reverse=True` to sort in descending order.
    
    2. **`sorted(list)`:**
       - This function creates a new sorted list without modifying the original list.
       - It also sorts in ascending order by default.
       - You can use `reverse=True` to sort in descending order.
    
    3. **Sorting by a specific key:**
       - You can provide a `key` argument to specify a function that extracts a value from each element to use for comparison during sorting. This is useful for sorting complex objects or lists of tuples.
    
    **Important Notes:**
    
    - Both `sort()` and `sorted()` operate on **mutable sequences** (like lists).
    - `sort()` modifies the original list, while `sorted()` returns a new sorted list.
    - The default sorting order is ascending.
    - You can customize the sorting order using the `key` and `reverse` arguments.
    
    

### Example 1
Use the CO-STAR prompt from above


```python
response =get_commpletion(client, MODEL, system_content, user_prompt)
print(response)
```

    Tired of the high costs and limitations of closed LLM models? 🤯 
    
    **Anyscale Endpoints** lets you run open-source LLMs like Llama and Mistral **at the lowest cost and latency**, giving you the transparency, security, and control you need. 🔐
    
    **Build your Gen AI applications with confidence.** 🚀
    
    Get started today: [link to blog post] 
    
    

### 🧙‍♀️ You got to love this stuff! 😜
