```python
!pip install litellm=="0.1.363"
```

    Collecting litellm==0.1.363
      Downloading litellm-0.1.363-py3-none-any.whl (34 kB)
    Requirement already satisfied: openai<0.28.0,>=0.27.8 in /usr/local/lib/python3.10/dist-packages (from litellm==0.1.363) (0.27.8)
    Requirement already satisfied: python-dotenv<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from litellm==0.1.363) (1.0.0)
    Requirement already satisfied: tiktoken<0.5.0,>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from litellm==0.1.363) (0.4.0)
    Requirement already satisfied: requests>=2.20 in /usr/local/lib/python3.10/dist-packages (from openai<0.28.0,>=0.27.8->litellm==0.1.363) (2.31.0)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from openai<0.28.0,>=0.27.8->litellm==0.1.363) (4.65.0)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from openai<0.28.0,>=0.27.8->litellm==0.1.363) (3.8.5)
    Requirement already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken<0.5.0,>=0.4.0->litellm==0.1.363) (2022.10.31)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai<0.28.0,>=0.27.8->litellm==0.1.363) (3.2.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai<0.28.0,>=0.27.8->litellm==0.1.363) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai<0.28.0,>=0.27.8->litellm==0.1.363) (1.26.16)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.20->openai<0.28.0,>=0.27.8->litellm==0.1.363) (2023.7.22)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai<0.28.0,>=0.27.8->litellm==0.1.363) (23.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai<0.28.0,>=0.27.8->litellm==0.1.363) (6.0.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai<0.28.0,>=0.27.8->litellm==0.1.363) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai<0.28.0,>=0.27.8->litellm==0.1.363) (1.9.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai<0.28.0,>=0.27.8->litellm==0.1.363) (1.4.0)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->openai<0.28.0,>=0.27.8->litellm==0.1.363) (1.3.1)
    Installing collected packages: litellm
      Attempting uninstall: litellm
        Found existing installation: litellm 0.1.362
        Uninstalling litellm-0.1.362:
          Successfully uninstalled litellm-0.1.362
    Successfully installed litellm-0.1.363
    


```python
# @title Import litellm & Set env variables
import litellm
import os

os.environ["ANTHROPIC_API_KEY"] = " " #@param
```


```python
# @title Request Claude Instant-1 and Claude-2
messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Who won the world series in 2020?"}
  ]

result = litellm.completion('claude-instant-1', messages)
print("\n\n Result from claude-instant-1", result)
result = litellm.completion('claude-2', messages, max_tokens=5, temperature=0.2)
print("\n\n Result from claude-2", result)
```

    
    
     Result from claude-instant-1 {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': " The Los Angeles Dodgers won the 2020 World Series, defeating the Tampa Bay Rays 4-2. It was the Dodgers' first World Series title since 1988."}}], 'created': 1691536677.2676156, 'model': 'claude-instant-1', 'usage': {'prompt_tokens': 30, 'completion_tokens': 32, 'total_tokens': 62}}
    
    
     Result from claude-2 {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': ' The Los Angeles Dodgers won'}}], 'created': 1691536677.944753, 'model': 'claude-2', 'usage': {'prompt_tokens': 30, 'completion_tokens': 5, 'total_tokens': 35}}
    


```python
# @title Streaming Example: Request Claude-2
messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "how does a court case get to the Supreme Court?"}
  ]

result = litellm.completion('claude-2', messages, stream=True)
for part in result:
    print(part.choices[0].delta.content or "")


```

     Here
    's
     a
     quick
     overview
     of
     how
     a
     court
     case
     can
     reach
     the
     U
    .
    S
    .
     Supreme
     Court
    :
    
    
    -
     The
     case
     must
     first
     be
     heard
     in
     a
     lower
     trial
     court
     (
    either
     a
     state
     court
     or
     federal
     district
     court
    ).
     The
     trial
     court
     makes
     initial
     r
    ulings
     and
     produces
     a
     record
     of
     the
     case
    .
    
    
    -
     The
     losing
     party
     can
     appeal
     the
     decision
     to
     an
     appeals
     court
     (
    a
     state
     appeals
     court
     for
     state
     cases
    ,
     or
     a
     federal
     circuit
     court
     for
     federal
     cases
    ).
     The
     appeals
     court
     reviews
     the
     trial
     court
    's
     r
    ulings
     and
     can
     affirm
    ,
     reverse
    ,
     or
     modify
     the
     decision
    .
    
    
    -
     If
     a
     party
     is
     still
     unsat
    isf
    ied
     after
     the
     appeals
     court
     rules
    ,
     they
     can
     petition
     the
     Supreme
     Court
     to
     hear
     the
     case
     through
     a
     writ
     of
     cert
    ior
    ari
    .
     
    
    
    -
     The
     Supreme
     Court
     gets
     thousands
     of
     cert
     petitions
     every
     year
     but
     usually
     only
     agrees
     to
     hear
     about
     100
    -
    150
     of
     cases
     that
     have
     significant
     national
     importance
     or
     where
     lower
     courts
     disagree
     on
     federal
     law
    .
     
    
    
    -
     If
     4
     out
     of
     the
     9
     Just
    ices
     vote
     to
     grant
     cert
     (
    agree
     to
     hear
     the
     case
    ),
     it
     goes
     on
     the
     Supreme
     Court
    's
     do
    cket
     for
     arguments
    .
    
    
    -
     The
     Supreme
     Court
     then
     hears
     oral
     arguments
    ,
     considers
     written
     brief
    s
    ,
     examines
     the
     lower
     court
     records
    ,
     and
     issues
     a
     final
     ruling
     on
     the
     case
    ,
     which
     serves
     as
     binding
     precedent
    
