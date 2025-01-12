# Bash chain
This notebook showcases using LLMs and a bash process to perform simple filesystem commands.


```python
from langchain_experimental.llm_bash.base import LLMBashChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain = LLMBashChain.from_llm(llm, verbose=True)

bash_chain.invoke(text)
```

    
    
    [1m> Entering new LLMBashChain chain...[0m
    Please write a bash script that prints 'Hello World' to the console.[32;1m[1;3m
    
    ```bash
    echo "Hello World"
    ```[0m
    Code: [33;1m[1;3m['echo "Hello World"'][0m
    Answer: [33;1m[1;3mHello World
    [0m
    [1m> Finished chain.[0m
    




    'Hello World\n'



## Customize Prompt
You can also customize the prompt that is used. Here is an example prompting to avoid using the 'echo' utility


```python
from langchain.prompts.prompt import PromptTemplate
from langchain_experimental.llm_bash.prompt import BashOutputParser

_PROMPT_TEMPLATE = """If someone asks you to perform a task, your job is to come up with a series of bash commands that will perform the task. There is no need to put "#!/bin/bash" in your answer. Make sure to reason step by step, using this format:
Question: "copy the files in the directory named 'target' into a new directory at the same level as target called 'myNewDirectory'"
I need to take the following actions:
- List all files in the directory
- Create a new directory
- Copy the files from the first directory into the second directory
```bash
ls
mkdir myNewDirectory
cp -r target/* myNewDirectory
```

Do not use 'echo' when writing the script.

That is the format. Begin!
Question: {question}"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
    output_parser=BashOutputParser(),
)
```


```python
bash_chain = LLMBashChain.from_llm(llm, prompt=PROMPT, verbose=True)

text = "Please write a bash script that prints 'Hello World' to the console."

bash_chain.invoke(text)
```

    
    
    [1m> Entering new LLMBashChain chain...[0m
    Please write a bash script that prints 'Hello World' to the console.[32;1m[1;3m
    
    ```bash
    printf "Hello World\n"
    ```[0m
    Code: [33;1m[1;3m['printf "Hello World\\n"'][0m
    Answer: [33;1m[1;3mHello World
    [0m
    [1m> Finished chain.[0m
    




    'Hello World\n'



## Persistent Terminal

By default, the chain will run in a separate subprocess each time it is called. This behavior can be changed by instantiating with a persistent bash process.


```python
from langchain_experimental.llm_bash.bash import BashProcess

persistent_process = BashProcess(persistent=True)
bash_chain = LLMBashChain.from_llm(llm, bash_process=persistent_process, verbose=True)

text = "List the current directory then move up a level."

bash_chain.invoke(text)
```

    
    
    [1m> Entering new LLMBashChain chain...[0m
    List the current directory then move up a level.[32;1m[1;3m
    
    ```bash
    ls
    cd ..
    ```[0m
    Code: [33;1m[1;3m['ls', 'cd ..'][0m
    Answer: [33;1m[1;3mcpal.ipynb  llm_bash.ipynb  llm_symbolic_math.ipynb
    index.mdx   llm_math.ipynb  pal.ipynb[0m
    [1m> Finished chain.[0m
    




    'cpal.ipynb  llm_bash.ipynb  llm_symbolic_math.ipynb\r\nindex.mdx   llm_math.ipynb  pal.ipynb'




```python
# Run the same command again and see that the state is maintained between calls
bash_chain.invoke(text)
```

    
    
    [1m> Entering new LLMBashChain chain...[0m
    List the current directory then move up a level.[32;1m[1;3m
    
    ```bash
    ls
    cd ..
    ```[0m
    Code: [33;1m[1;3m['ls', 'cd ..'][0m
    Answer: [33;1m[1;3m_category_.yml	data_generation.ipynb		   self_check
    agents		graph
    code_writing	learned_prompt_optimization.ipynb[0m
    [1m> Finished chain.[0m
    




    '_category_.yml\tdata_generation.ipynb\t\t   self_check\r\nagents\t\tgraph\r\ncode_writing\tlearned_prompt_optimization.ipynb'


