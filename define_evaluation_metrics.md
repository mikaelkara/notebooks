```python
from langchain_openai import ChatOpenAI 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator
from langchain_core.pydantic_v1 import BaseModel, Field

# from langchain.evaluation.criteria import {
#     CriteriaEvalChain,
#     LabeledCriteriaEvalChain
# }
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

```


```python
llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

```


```python
class ResultScore(BaseModel):
    score: float = Field(..., description="The score of the result, ranging from 0 to 1 where 1 is the best possible score.")
    # explanation: str = Field(..., description="An extensive explanation of the score.")

```


```python
correctness_prompt = PromptTemplate(
input_variables=["question", "ground_truth", "generated_answer"],
template="""
Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

Evaluate the correctness of the generated answer compared to the ground truth.
Score from 0 to 1, where 1 is perfectly correct and 0 is completely incorrect.
any score between 0 and 1 is acceptable and depends on how correct the generated answer is.

Score:
"""
)
correctness_chain = correctness_prompt | llm.with_structured_output(ResultScore)


def evaluate_correctness(question, ground_truth, generated_answer):
    """Evaluates the correctness of the generated answer compared to the ground truth.

    Args:
        question: The question.
        ground_truth: The ground truth answer.
        generated_answer: The generated answer.

    Returns:
        A float between 0 and 1, where 1 is the best possible score.
    """
    result = correctness_chain.invoke({"question": question, "ground_truth": ground_truth, "generated_answer": generated_answer})
    return result.score

```


```python
# test create_correctness_chain
question = "What is the capital of France and Spain?"
ground_truth = "Paris and Barcelona"
generated_answer = "Paris"
score = evaluate_correctness(question, ground_truth, generated_answer)
```


```python
score
```




    0.5




```python
faithfulness_prompt = PromptTemplate(
input_variables=["question","context", "generated_answer"],
template="""
Question: {question}
Context: {context}
Generated Answer: {generated_answer}

Evaluate if the generate answer to the question can be deduced from the context.
Score of 0 or 1, where 1 is perfectly faithful *AND CAN BE DERIVED FROM THE CONTEXT* and 0 otherwise.
you don't mind if the answer is correct, all you care about is if the answer can be deduced from the context.

example:
Question: What are the capitals of France and Spain?
Context: Paris is the capital of France and Madrid is the capital of Spain.
Generated Answer: Paris
in this case the generated answer is faithful to the context so the score should be *1*.

example:
Question: What are the capital cities of France and Spain?
Context: London is the capital of France and Barcelona is the capital of Spain.
Generated Answer: London and Barcelona.
in this case the generated answer is faithful to the context so the score should be *1*.

example:
Question: What are the capital cities of France and Spain?
Context: Paris is the capital of France and Madrid is the capital of Spain.
Generated Answer: Paris.
in this case the generated answer is faithful to the context so the score should be *1*.

exmaple:
Question: What are the capitals of France and Spain?
Context: London is the capital of France and Madrid is the Capital of Spain.
Generated Answer: Paris and Madrid.
in this case the generated answer is based on the pretrained knowledge of the llm and is not faithful to the context so the score should be *0*.

example:
Question: What is the capital of France and Spain?
Context: Monkeys like to eat bananas.
Generated Answer: Paris and Madrid.
in this case the generated answer is not based on the context so the score should be *0*.

example:
Question: What is the capital of France?
Context: Paris.
Generated Answer: Paris.
in this case the context doesn't specify that Paris is the capital of France, and it cannot be deduced from the context, so the score should be *0*.


Example:
Question: What is 2+2?
Context: 4.
Generated Answer: 4.
In this case, the context states '4', but it does not provide information to deduce the answer to 'What is 2+2?', so the score should be *0*.
"""
)
faithfulness_chain = faithfulness_prompt | llm.with_structured_output(ResultScore)
```


```python
def evaluate_faithfulness(question, context, generated_answer):
    """Evaluates if the generate answer to the question can be deduced from the context.

    Args:
        question: The question.
        context: The context.
        generated_answer: The generated answer.

    Returns:
        A float between 0 and 1, where 1 is the best possible score.
    """
    result = faithfulness_chain.invoke({"question": question, "context": context, "generated_answer": generated_answer})
    return result.score, result.explanation
```


```python
# test create_faithfulness_chain
question = "what is 3+3?"
context = "6"
generated_answer = "6"
score, explanation = evaluate_faithfulness(question, context, generated_answer)
print(score)
print(explanation)
```

    0.0
    The context states '6', but it does not provide information to deduce the answer to 'What is 3+3?'. The answer is correct, but it cannot be derived from the context.
    


```python
from langchain import PromptTemplate

relevancy_score_prompt = PromptTemplate(
    input_variables=["question", "contexts"],
    template="""
Q: {question}
Docs: {contexts}

Score each doc's relevance:
0.00 - Irrelevant: No relation to the question
0.33 - Somewhat relevant: Contains related keywords or concepts
0.66 - Relevant: Partially answers or strongly implies the answer
1.00 - Highly relevant: Directly and fully answers the question

Consider: Relevance, Directness, Completeness, Accuracy

Final Score: [Average of all scores]
"""
)
ratio_of_relevant_docs_chain = ratio_of_relevant_docs_prompt | llm.with_structured_output(ResultScore)
```


```python
def evaluate_ratio_of_relevant_docs(question, contexts):
    """Evaluates the ratio of relevant documents in the contexts to the question.

    Args:
        question: The question.
        contexts: A list of documents.

    Returns:
        A float between 0 and 1, where 1 is the best possible score.
    """
    result = ratio_of_relevant_docs_chain.invoke({"question": question, "contexts": contexts})
    return result.score
```


```python
# test create_ratio_of_relevant_docs_chain
question = "What is the capital of France?"
contexts = ["Paris.", "i was traveling in France."]
score = evaluate_ratio_of_relevant_docs(question, contexts)
# score, explanation = evaluate_ratio_of_relevant_docs(question, contexts)
print(score)
# print(explanation)

```

    1.0
    
