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

The examples below illustracte simple prompting: asking questions and fashioning the response. 

<img src="./images/llm_prompt_req_resp.png" height="35%" width="%65">


**Note**: 
To run any of these relevant notebooks you will need an account on Anyscale Endpoints, Anthropic, or OpenAI, depending on what model you elect, along with the respective environment file. Use the template environment files to create respective `.env` file for either Anyscale Endpoints, Anthropic, or OpenAI.


```python
import warnings
import os

from anthropic import Anthropic

from dotenv import load_dotenv, find_dotenv
from llm_clnt_factory_api import ClientFactory
```

Load our .env file with respective API keys and base url endpoints and use Anthropic endpoints


```python
_ = load_dotenv(find_dotenv()) # read local .env file
warnings.filterwarnings('ignore')
api_key = os.getenv("ANTHROPIC_API_KEY", None)
MODEL = os.getenv("MODEL")
print(f"Using MODEL={MODEL}; base={'Anthropic'}")
```

    Using MODEL=claude-3-opus-20240229; base=Anthropic
    


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

#### Creat an anthropic client using our factory class


```python
client_factory = ClientFactory()
client_type = "anthropic"
client_factory.register_client(client_type, Anthropic)
client_kwargs = {"api_key": api_key}

# create the client
client = client_factory.create_client(client_type, **client_kwargs)
```


```python
def get_commpletion(clnt: object, model: str, system_content: str, user_content:str) -> str:
    chat_completion = clnt.messages.create(
        model=model,
        system = system_content,
        messages=[{"role": "user", "content": user_content}],
         max_tokens=1000,
        temperature = 0.8)

    response = chat_completion.content[0].text
    return response
```

To use Anthropic Endpoints, simply copy your `env/env_anthropic_template` to `.env` file in the top directory, and enter your relevant API keys. It should work as a charm!

## Simple queries


```python
print(f"Using Endpoints: {'Anthropic'} with model {MODEL} ...\n")
for user_content in user_questions:
    response = get_commpletion(client, MODEL, system_content, user_content)
    print(f"\n{BOLD_BEGIN}Question:{BOLD_END} {user_content}")
    print(f"\n{BOLD_BEGIN}Answer:{BOLD_END} {response}")
```

    Using Endpoints: Anthropic with model claude-3-opus-20240229 ...
    
    
    [1mQuestion:[0m Who was Benjamin Franklin, and what is he most known for?
    
    [1mAnswer:[0m Benjamin Franklin was a remarkable figure in American history, known for his numerous talents and accomplishments. Born in 1706 in Boston, Massachusetts, Franklin was a polymath who excelled as a printer, writer, scientist, inventor, diplomat, and statesman. He is perhaps most famous for his experiments with electricity, which led to his invention of the lightning rod, a device that protects buildings from lightning strikes.
    
    In addition to his scientific pursuits, Franklin was also a key figure in the American Revolution and helped draft the Declaration of Independence and the U.S. Constitution. He was a strong advocate for education, establishing the first public library in America and the University of Pennsylvania. Franklin's wit, wisdom, and inventiveness made him an iconic figure, and his legacy continues to inspire people around the world today.
    
    [1mQuestion:[0m Who is considered the father of Artificial Intelligence (AI)?
    
    [1mAnswer:[0m Alan Turing, a British mathematician and computer scientist, is often regarded as the father of Artificial Intelligence. In 1950, he proposed the famous "Turing Test," which evaluates a machine's ability to exhibit intelligent behavior indistinguishable from that of a human. This groundbreaking idea laid the foundation for the field of AI and inspired generations of researchers to pursue the goal of creating intelligent machines. Turing's visionary work, including his contributions to computer science, cryptography, and theoretical biology, made him a pivotal figure in the development of AI as we know it today.
    

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
     🚀 Exciting news for Gen AI developers!
    
    We've just launched Anyscale Endpoints, a new feature that lets you serve open-source LLMs like Llama and Mistral with the lowest latency and cost. 💸
    
    With Anyscale Endpoints, you get:
    
    ✅ Transparent, secure, and fully-controlled model hosting
    ✅ Predictable and affordable pricing
    ✅ Easy integration and lightning-fast inference
    
    Ready to build the next big thing in AI? Get started for free with our quick start guides and tutorials. 👨‍💻👩‍💻
    
    Learn more and sign up at: [link to blog post]
    
    #AI #MachineLearning #Developers #OpenSource
    

### 🧙‍♀️ You got to love this stuff! 😜
