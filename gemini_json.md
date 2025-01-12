### Extract JSON using prompt engineering ###

### The following code recieves text and summerises the text into topics with summaries using Google Gemnini Flash ###

### Install libraries ###


```python
%pip install -U google-generativeai
```

    Requirement already satisfied: google-generativeai in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (0.5.3)
    Collecting google-generativeai
      Downloading google_generativeai-0.6.0-py3-none-any.whl.metadata (3.9 kB)
    Collecting google-ai-generativelanguage==0.6.4 (from google-generativeai)
      Downloading google_ai_generativelanguage-0.6.4-py3-none-any.whl.metadata (5.6 kB)
    Requirement already satisfied: google-api-core in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (2.17.1)
    Requirement already satisfied: google-api-python-client in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (2.129.0)
    Requirement already satisfied: google-auth>=2.15.0 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (2.28.1)
    Requirement already satisfied: protobuf in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (4.25.3)
    Requirement already satisfied: pydantic in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (2.6.1)
    Requirement already satisfied: tqdm in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (4.66.2)
    Requirement already satisfied: typing-extensions in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-generativeai) (4.9.0)
    Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-ai-generativelanguage==0.6.4->google-generativeai) (1.23.0)
    Requirement already satisfied: googleapis-common-protos<2.0.dev0,>=1.56.2 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-core->google-generativeai) (1.62.0)
    Requirement already satisfied: requests<3.0.0.dev0,>=2.18.0 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-core->google-generativeai) (2.31.0)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-auth>=2.15.0->google-generativeai) (5.3.3)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-auth>=2.15.0->google-generativeai) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-auth>=2.15.0->google-generativeai) (4.9)
    Requirement already satisfied: httplib2<1.dev0,>=0.19.0 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-python-client->google-generativeai) (0.22.0)
    Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-python-client->google-generativeai) (0.2.0)
    Requirement already satisfied: uritemplate<5,>=3.0.1 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-python-client->google-generativeai) (4.1.1)
    Requirement already satisfied: annotated-types>=0.4.0 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from pydantic->google-generativeai) (0.6.0)
    Requirement already satisfied: pydantic-core==2.16.2 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from pydantic->google-generativeai) (2.16.2)
    Requirement already satisfied: colorama in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from tqdm->google-generativeai) (0.4.6)
    Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.4->google-generativeai) (1.62.0)
    Requirement already satisfied: grpcio-status<2.0.dev0,>=1.33.2 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.4->google-generativeai) (1.62.0)
    Requirement already satisfied: pyparsing!=3.0.0,!=3.0.1,!=3.0.2,!=3.0.3,<4,>=2.4.2 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from httplib2<1.dev0,>=0.19.0->google-api-python-client->google-generativeai) (3.1.1)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai) (0.5.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai) (2.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\shaig\appdata\local\programs\python\python312\lib\site-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai) (2024.2.2)
    Downloading google_generativeai-0.6.0-py3-none-any.whl (158 kB)
       ---------------------------------------- 0.0/158.8 kB ? eta -:--:--
       -- ------------------------------------- 10.2/158.8 kB ? eta -:--:--
       ---------- ---------------------------- 41.0/158.8 kB 393.8 kB/s eta 0:00:01
       ---------------------- ---------------- 92.2/158.8 kB 655.4 kB/s eta 0:00:01
       ------------------------------------ - 153.6/158.8 kB 919.0 kB/s eta 0:00:01
       -------------------------------------- 158.8/158.8 kB 794.2 kB/s eta 0:00:00
    Downloading google_ai_generativelanguage-0.6.4-py3-none-any.whl (679 kB)
       ---------------------------------------- 0.0/679.1 kB ? eta -:--:--
       ---------- ----------------------------- 174.1/679.1 kB 5.1 MB/s eta 0:00:01
       ---------------------- ----------------- 389.1/679.1 kB 4.0 MB/s eta 0:00:01
       ---------------------------------- ----- 593.9/679.1 kB 4.1 MB/s eta 0:00:01
       ---------------------------------------  675.8/679.1 kB 4.2 MB/s eta 0:00:01
       ---------------------------------------- 679.1/679.1 kB 3.1 MB/s eta 0:00:00
    Installing collected packages: google-ai-generativelanguage, google-generativeai
      Attempting uninstall: google-ai-generativelanguage
        Found existing installation: google-ai-generativelanguage 0.6.3
        Uninstalling google-ai-generativelanguage-0.6.3:
          Successfully uninstalled google-ai-generativelanguage-0.6.3
      Attempting uninstall: google-generativeai
        Found existing installation: google-generativeai 0.5.3
        Uninstalling google-generativeai-0.5.3:
          Successfully uninstalled google-generativeai-0.5.3
    Successfully installed google-ai-generativelanguage-0.6.4 google-generativeai-0.6.0
    Note: you may need to restart the kernel to use updated packages.
    

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    langchain-google-genai 0.0.9 requires google-generativeai<0.4.0,>=0.3.1, but you have google-generativeai 0.6.0 which is incompatible.
    
    [notice] A new release of pip is available: 23.3.1 -> 24.0
    [notice] To update, run: python.exe -m pip install --upgrade pip
    

### Import necessary libraries ###


```python
import google.generativeai as genai
import google.ai.generativelanguage as glm

import json
```

    c:\Users\shaig\AppData\Local\Programs\Python\Python312\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

### Define prompt ###


```python
PROMPT = """
You are an assistant that segments text into topics and provides a summary for each topic.

Generate multiple distinct topics.
Provide a unique concise summary for each topic. Summaries should be at most 40 words / 2 sentences long.
Topics and summaries should be unique and should not repeat.
Summaries must accuratly reflect the topic.

Both topics and summaries must be in English.

Return the result in a valid JSON format.
The format to return in, as plain text without Markdown formatting:
[
    {{
        "topic": "subject that covers a substantial portion of the text",
        "summary": "brief summary of the segment's content that includes several keywords that encapsulate it"
    }}
]

Ensure to escape double quotation marks ("), by using \\, in the generated subjects and summaries for valid JSON output.
For example, if the summary contains the text: I work at "Google", it should be escaped as: I work at \\"Google\\".
"""
```

### Define text to analyze ###


```python
TEXT = """
Gemini Prompting: Crafting Effective Instructions for AI Interaction
Gemini, developed by Google, is a powerful language model capable of comprehending and generating text across various domains. To harness its full potential, understanding the art of crafting effective prompts is crucial. A well-crafted prompt acts as a guiding light, directing the AI towards the desired response and ensuring accurate, relevant, and insightful output.
At its core, a Gemini prompt is a simple text input, but its impact is far-reaching. It can be a direct question, a detailed instruction, or even a creative scenario. The key is to be clear, concise, and specific. Ambiguity can lead to confusion and misinterpretation, resulting in unsatisfactory responses. Instead, aim for prompts that are focused and unambiguous, providing all the necessary context for Gemini to understand your intent.
One effective technique is to provide examples within your prompt. This "few-shot learning" approach allows Gemini to quickly grasp the desired format or style of the response. For instance, if you want a summary of an article, you could provide a short example summary to guide the AI's output. Similarly, if you're looking for a specific type of code, including a snippet of similar code can significantly improve the results.
Another important aspect is to consider the tone and style of your prompt. Do you want a formal, informative response or a more casual, conversational one? By adjusting the language and tone of your prompt, you can influence the style of Gemini's output. Additionally, don't hesitate to experiment with different approaches. Try rephrasing your prompt, adjusting its structure, or adding more detail. The iterative process of refining your prompts will lead to more accurate and satisfying results over time.
In conclusion, mastering Gemini prompting is an ongoing journey of experimentation and learning. By understanding the principles of clarity, context, and specificity, and by utilizing techniques like few-shot learning and tone adjustment, you can unlock the full potential of this powerful language model. With practice and refinement, you can transform your prompts into precise tools, guiding Gemini to deliver the information, insights, and creative outputs you seek.
"""
```

### Initiate Gemini model ###


```python
genai.configure(api_key="AIzaSyAUU5LUcYgycmop2F_YkFTnzWECIAdtR5g")
flash_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
```


```python
def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except ValueError:
        return False
    
def generate_json_flash():
    response = flash_model.generate_content(PROMPT + "\nHere is the text:\n " + TEXT)
    return response.text
```

### Generate JSON topics and summaries ###


```python
output_json = generate_json_flash()
print(output_json)

```

    [
        {
            "topic": "Gemini: A Powerful Language Model",
            "summary": "Gemini is a powerful language model developed by Google that can understand and generate text across different domains. To maximize its potential, it is essential to craft effective prompts."
        },
        {
            "topic": "Crafting Effective Prompts for Gemini",
            "summary": "An effective Gemini prompt is clear, concise, and specific. It can be a question, an instruction, or a scenario. Providing examples, adjusting tone, and experimenting with different approaches can improve results."
        },
        {
            "topic": "Importance of Clarity and Specificity in Prompts",
            "summary": "Ambiguity in prompts can lead to confusion and unsatisfactory responses. Focused, unambiguous prompts provide the necessary context for Gemini to understand the intent."
        },
        {
            "topic": "Few-Shot Learning in Gemini Prompts",
            "summary": "Including examples within the prompt, known as \"few-shot learning,\" helps Gemini understand the desired format or style. For instance, providing a summary example can guide the AI's output."
        },
        {
            "topic": "Tone and Style in Gemini Prompts",
            "summary": "The language and tone of a prompt influence the style of Gemini's output. Adjusting these elements can determine whether the response is formal, informative, casual, or conversational."
        },
        {
            "topic": "Experimentation and Refinement in Gemini Prompts",
            "summary": "Experimenting with different prompt approaches, such as rephrasing or adding detail, can lead to more accurate and satisfying results. Refining prompts is an ongoing process."
        }
    ]
    
