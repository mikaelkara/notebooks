[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1arL7bWuF2P3soS3p19MWJeUDtW0Eu5tk?usp=sharing)

# Get Started with Llama 3.1 Models


Llama 3.1 release comes with three sizes of models 7B, 70B and 405B

In this notebook, we will look at :

*  How to access the Llama 3.1 models over a API?
*  Generate Structured Synthetic Instruction Dataset with Llama 3.1 405B


## Setup

Install all the dependencies and import the required python modules.


```python
!pip3 install --upgrade fireworks-ai
```

    Requirement already satisfied: fireworks-ai in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.15.8)
    Requirement already satisfied: httpx in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (0.27.2)
    Requirement already satisfied: httpx-ws in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (0.6.2)
    Requirement already satisfied: httpx-sse in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (0.4.0)
    Requirement already satisfied: pydantic in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (2.9.2)
    Requirement already satisfied: Pillow in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (10.4.0)
    Requirement already satisfied: anyio in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx->fireworks-ai) (4.6.0)
    Requirement already satisfied: certifi in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx->fireworks-ai) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx->fireworks-ai) (1.0.5)
    Requirement already satisfied: idna in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx->fireworks-ai) (3.10)
    Requirement already satisfied: sniffio in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx->fireworks-ai) (1.3.1)
    Requirement already satisfied: h11<0.15,>=0.13 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpcore==1.*->httpx->fireworks-ai) (0.14.0)
    Requirement already satisfied: wsproto in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx-ws->fireworks-ai) (1.2.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic->fireworks-ai) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic->fireworks-ai) (2.23.4)
    Requirement already satisfied: typing-extensions>=4.6.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic->fireworks-ai) (4.12.2)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    

## Setup your API Key

In order to use the Llama 3.1, you must first obtain Fireworks API Keys. If you don't already have one, you can one by following the instructions [here](https://docs.fireworks.ai/getting-started/quickstart).


```python
from fireworks.client import Fireworks

#replace the FIREWORKS_API_KEY with the key copied in the above step.
client = Fireworks(api_key="FIREWORKS_API_KEY")
```

## Accessing Llama 3.1 Models using API

We are sending a request to Llama 3.1 405B model, alternatively you can change the model string to access the otherm models.

* accounts/fireworks/models/llama-v3p1-70b-instruct
* accounts/fireworks/models/llama-v3p1-8B-instruct

### Chat Completions API


```python
model_name = "accounts/fireworks/models/llama-v3p1-405b-instruct"

response = client.chat.completions.create(
	model=model_name,
	messages=[{
		"role": "user",
		"content": "Who are you?",
	}],
)
print(response.choices[0].message.content)
```

    I'm an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."
    

## Generate Synthetic Data




```python
pip install pydantic
```

    Requirement already satisfied: pydantic in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (2.9.2)
    Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic) (2.23.4)
    Requirement already satisfied: typing-extensions>=4.6.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic) (4.12.2)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.
    


```python
from pydantic import BaseModel, Field
```


```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Category(str, Enum):
    COUNTRIES = "Countries"
    CAPITALS = "Capitals"
    RIVERS = "Rivers"
    MOUNTAINS = "Mountains"
    LANDMARKS = "Landmarks"
    CLIMATE = "Climate"
    CULTURE = "Culture"

class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "Multiple Choice"
    TRUE_FALSE = "True/False"
    FILL_IN_THE_BLANK = "Fill in the Blank"
    SHORT_ANSWER = "Short Answer"

class Question(BaseModel):
    instruction: str
    context: str
    response: str
    question_type: QuestionType
    category: Category
    difficulty: Difficulty

class GeographyQuizDataset(BaseModel):
    title: str = "World Geography Challenge Dataset"
    description: str = "Dataset for geography quiz questions and answers"
    questions: List[Question]
```


```python
import json
def generate_question():
    prompt = """Generate a geography quiz question. Format your response as a JSON object with the following structure:
    {
        "instruction": "The full question text",
        "context": "Provide context about the question",
        "response": "The correct answer",
        "question_type": "The type of question (e.g., 'Multiple Choice')",
        "category": "The category should be marked as one of these: Countries, Capitals, Rivers, Mountains, Landmarks, Climate, Culture",
        "difficulty": "The difficulty level of the question (e.g., 'Easy')"
    }"""

    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a geography expert creating quiz questions."},
            {"role": "user", "content": prompt}
        ]
    )

    question_data = json.loads(response.choices[0].message.content)
    print(question_data)
    return Question(**question_data)

def main(num_questions=10):
    with open("geography_quiz_dataset.jsonl", "w") as f:
        for i in range(num_questions):
            question = generate_question()
            json.dump(question.dict(), f)
            f.write("\n")
            print(f"Generated question {i+1}/{num_questions}: {question.instruction}")

    print(f"Generated and saved {num_questions} questions to geography_quiz_dataset.jsonl")

if __name__ == "__main__":
    main()
```

    {'instruction': 'Which river is the longest in South America and flows through Brazil, Peru, and Colombia before emptying into the Pacific Ocean?', 'context': 'Rivers of the World', 'response': 'Amazon River', 'question_type': 'Multiple Choice', 'category': 'Rivers', 'difficulty': 'Medium'}
    Generated question 1/10: Which river is the longest in South America and flows through Brazil, Peru, and Colombia before emptying into the Pacific Ocean?
    {'instruction': "What is the world's largest desert, covering about 9,200,000 square kilometers (3,600,000 sq mi), and spanning across several countries in North Africa?", 'context': 'Deserts are large areas of land with very little rainfall and limited vegetation. They can be hot or cold and are found on every continent. The largest hot desert in the world is a significant geographical feature that affects climate, culture, and ecosystems across North Africa.', 'response': 'Sahara', 'question_type': 'Short Answer', 'category': 'Landmarks', 'difficulty': 'Medium'}
    Generated question 2/10: What is the world's largest desert, covering about 9,200,000 square kilometers (3,600,000 sq mi), and spanning across several countries in North Africa?
    {'instruction': 'Which river, approximately 6,400 kilometers long, flows through Brazil, Peru, and Colombia before emptying into the Pacific Ocean?', 'context': 'This question tests knowledge of major rivers in South America.', 'response': 'Amazon River', 'question_type': 'Multiple Choice', 'category': 'Rivers', 'difficulty': 'Medium'}
    Generated question 3/10: Which river, approximately 6,400 kilometers long, flows through Brazil, Peru, and Colombia before emptying into the Pacific Ocean?
    {'instruction': 'Which river is the longest in South America?', 'context': 'South America is home to many significant rivers, including the Orinoco, São Francisco, and Magdalena. However, one river stands out for its exceptional length.', 'response': 'Amazon River', 'question_type': 'Multiple Choice', 'category': 'Rivers', 'difficulty': 'Easy'}
    Generated question 4/10: Which river is the longest in South America?
    {'instruction': 'What is the name of the largest island in the Mediterranean Sea?', 'context': 'The Mediterranean Sea is a semi-enclosed sea connected to the Atlantic Ocean, surrounded by the Mediterranean region and almost completely enclosed by land: on the north by Southern Europe and Anatolia, on the south by North Africa, and on the east by the Levant.', 'response': 'Sicily', 'question_type': 'Multiple Choice', 'category': 'Landmarks', 'difficulty': 'Easy'}
    Generated question 5/10: What is the name of the largest island in the Mediterranean Sea?
    {'instruction': 'What is the name of the strait that separates the continents of Asia and Africa?', 'context': 'This strait is a significant shipping route and connects the Red Sea to the Gulf of Aden.', 'response': 'Bab-el-Mandeb', 'question_type': 'Short Answer', 'category': 'Landmarks', 'difficulty': 'Medium'}
    Generated question 6/10: What is the name of the strait that separates the continents of Asia and Africa?
    {'instruction': "What is the world's largest desert, covering over 9,000,000 square kilometers (3,500,000 sq mi), and spanning across several countries in North Africa?", 'context': 'Deserts of the world', 'response': 'Sahara', 'question_type': 'Short Answer', 'category': 'Landmarks', 'difficulty': 'Medium'}
    Generated question 7/10: What is the world's largest desert, covering over 9,000,000 square kilometers (3,500,000 sq mi), and spanning across several countries in North Africa?
    {'instruction': "What is the world's largest desert, covering about 9,200,000 square kilometers (3,600,000 sq mi), and spanning across several countries in North Africa?", 'context': 'Deserts are known for their extreme heat and arid conditions. This particular desert covers a significant portion of the African continent.', 'response': 'Sahara', 'question_type': 'Short Answer', 'category': 'Landmarks', 'difficulty': 'Medium'}
    Generated question 8/10: What is the world's largest desert, covering about 9,200,000 square kilometers (3,600,000 sq mi), and spanning across several countries in North Africa?
    {'instruction': "What is the world's largest desert, covering about 9,200,000 square kilometers (3,600,000 sq mi), and spanning across several countries in North Africa?", 'context': 'Deserts are vast expanses of arid land, often characterized by extreme heat and limited precipitation.', 'response': 'Sahara', 'question_type': 'Short Answer', 'category': 'Landmarks', 'difficulty': 'Medium'}
    Generated question 9/10: What is the world's largest desert, covering about 9,200,000 square kilometers (3,600,000 sq mi), and spanning across several countries in North Africa?
    {'instruction': 'Which river is the longest in South America and flows through Brazil, Peru, and Colombia before emptying into the Pacific Ocean?', 'context': 'Rivers of South America', 'response': 'Amazon River', 'question_type': 'Multiple Choice', 'category': 'Rivers', 'difficulty': 'Medium'}
    Generated question 10/10: Which river is the longest in South America and flows through Brazil, Peru, and Colombia before emptying into the Pacific Ocean?
    Generated and saved 10 questions to geography_quiz_dataset.jsonl
    

## Conclusion

We’re excited to see how the community leverages Llama 3.1 API to create interesting applications.


For more information and to get started with Llama 3.1, visit [docs.fireworks.ai](https://docs.fireworks.ai) or join our [discord community](https://discord.gg/fireworks-ai)
