# LLM Symbolic Math 
This notebook showcases using LLMs and Python to Solve Algebraic Equations. Under the hood is makes use of [SymPy](https://www.sympy.org/en/index.html).


```python
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
from langchain_openai import OpenAI

llm = OpenAI(temperature=0)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)
```

## Integrals and derivates


```python
llm_symbolic_math.invoke("What is the derivative of sin(x)*exp(x) with respect to x?")
```




    'Answer: exp(x)*sin(x) + exp(x)*cos(x)'




```python
llm_symbolic_math.invoke(
    "What is the integral of exp(x)*sin(x) + exp(x)*cos(x) with respect to x?"
)
```




    'Answer: exp(x)*sin(x)'



## Solve linear and differential equations


```python
llm_symbolic_math.invoke('Solve the differential equation y" - y = e^t')
```




    'Answer: Eq(y(t), C2*exp(-t) + (C1 + t/2)*exp(t))'




```python
llm_symbolic_math.invoke("What are the solutions to this equation y^3 + 1/3y?")
```




    'Answer: {0, -sqrt(3)*I/3, sqrt(3)*I/3}'




```python
llm_symbolic_math.invoke("x = y + 5, y = z - 3, z = x * y. Solve for x, y, z")
```




    'Answer: (3 - sqrt(7), -sqrt(7) - 2, 1 - sqrt(7)), (sqrt(7) + 3, -2 + sqrt(7), 1 + sqrt(7))'


