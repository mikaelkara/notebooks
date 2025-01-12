# [CodeShield](https://github.com/meta-llama/PurpleLlama/tree/main/CodeShield) Usage Walkthrough

This notebook shows examples of how to use CodeShield. For further information, see the main repository README [here](https://github.com/meta-llama/PurpleLlama/tree/main/CodeShield).

# Getting Started 

Either install via PyPi using pip, or install it locally from source. 

#### Install Option 1. Install CodeShield package. Run the following in your terminal

```
pip3 install codeshield
```

#### Install Option 2. Install CodeShield package locally from source code

```
git clone https://github.com/meta-llama/PurpleLlama
cd PurpleLlama/CodeShield
pip install .
```

# Examples

Define a helper function to run scan and process output.


```python
from codeshield.cs import CodeShield

async def scan_llm_output(llm_output_code):
    result = await CodeShield.scan_code(llm_output_code)
    if result.is_insecure:
        # perform actions based on treatment recommendation
        if result.recommended_treatment == "block":
            llm_output_code = "*** Code Security issues found, blocking the code ***"
        if result.recommended_treatment == "warn":
            llm_output_code = llm_output_code + "*** Warning: The generated snippit contains insecure code ***"
    
    
    summary = "Security issue detected" if result.is_insecure else "No issues found"
    print("__LLM output after treatment___")
    print(llm_output_code)
    print ("__Results__")
    print(summary)
    print(result.recommended_treatment)
    print ("__Details__")
    print(result.issues_found)

```

## Example 1 - Simple hard-coded example
In this example, we are taking a hard coded example where the code snippet has weak hash md5 function usage. 


```python
llm_output_code = """
def hashString(input):
    return hashlib.md5(input)
"""

await scan_llm_output(llm_output_code)
```

    __LLM output after treatment___
    
    def hashString(input):
        return hashlib.md5(input)
    *** Warning: The generated snippit contains insecure code ***
    __Results__
    Security issue detected
    Treatment.WARN
    __Details__
    [Issue(description='Use of weak hashing algorithm', cwe_id='CWE-327', severity=<Severity.WARNING: 'warning'>, rule='\\.getMd5Digest\\(\\)|\\.md5\\(|\\.md5Hex\\(|\\.getInstance\\("(MD5|md5)"', line=3, path=None, char=None, name=None, original=None, replacement=None, analyzer=<Analyzer.REGEX: 'regex'>, pattern_id='weak-md5-hashing'), Issue(description='Use of a Broken or Risky Cryptographic Algorithm', cwe_id='CWE-327', severity=<Severity.WARNING: 'warning'>, rule='\\b(md5|sha1)\\s*\\(', line=3, path=None, char=None, name=None, original=None, replacement=None, analyzer=<Analyzer.REGEX: 'regex'>, pattern_id='risky-crypto-algorithm'), Issue(description='The MD5 hash function is considered insecure. Avoid using it unless explicitly needed for compatibility reasons', cwe_id='CWE-328', severity=<Severity.WARNING: 'warning'>, rule='\\bhashlib\\.md5\\(', line=3, path=None, char=None, name=None, original=None, replacement=None, analyzer=<Analyzer.REGEX: 'regex'>, pattern_id='insecure-md5-hash-usage')]
    

## Example 2 - use openAI API

Requires openai package (pip install openai)


```python
prompt = "please generate some example code to demonstrate strcpy usage"

import openai
client = openai.OpenAI(api_key="YOUR_OPEN_AI_KEY")
response = client.chat.completions.create(
    model= "gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ],
    max_tokens=1000,
)

await scan_llm_output(response.choices[0].message.content)
```

## Example 3 - use externally hosted LLM 

Requires [llama-recipes package](https://github.com/meta-llama/llama-recipes)


```python
import os
import getpass

from llama_recipes.inference.llm import TOGETHER, OPENAI, ANYSCALE

if "EXTERNALLY_HOSTED_LLM_TOKEN" not in os.environ:
    os.environ["EXTERNALLY_HOSTED_LLM_TOKEN"] = getpass.getpass(prompt="Provide token for LLM provider")

# Delete as appropriate
model = TOGETHER("togethercomputer/CodeLlama-13b-Instruct", os.environ["EXTERNALLY_HOSTED_LLM_TOKEN"])
model = OPENAI("gpt-4",os.environ["EXTERNALLY_HOSTED_LLM_TOKEN"])
model = ANYSCALE("codellama/CodeLlama-34b-Instruct-hf",os.environ["EXTERNALLY_HOSTED_LLM_TOKEN"])

llm_output_code = model.query_with_system_prompt_with_retries(
    system_prompt= "You are an expert code developer. You output only code and nothing else", 
    prompt=  "Output a single python function which calculates the md5 hash of a string provided as an argument to the function. Output only the code and nothing else."
    )
await scan_llm_output(llm_output_code)
```
