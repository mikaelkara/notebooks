# Evaluate Multiple LLM Providers with LiteLLM



*   Quality Testing
*   Load Testing
*   Duration Testing




```python
!pip install litellm python-dotenv
```


```python
import litellm
from litellm import load_test_model, testing_batch_completion
import time
```


```python
from dotenv import load_dotenv
load_dotenv()
```

# Quality Test endpoint

## Test the same prompt across multiple LLM providers

In this example, let's ask some questions about Paul Graham


```python
models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "claude-instant-1", "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781"]
context = """Paul Graham (/ɡræm/; born 1964)[3] is an English computer scientist, essayist, entrepreneur, venture capitalist, and author. He is best known for his work on the programming language Lisp, his former startup Viaweb (later renamed Yahoo! Store), cofounding the influential startup accelerator and seed capital firm Y Combinator, his essays, and Hacker News. He is the author of several computer programming books, including: On Lisp,[4] ANSI Common Lisp,[5] and Hackers & Painters.[6] Technology journalist Steven Levy has described Graham as a "hacker philosopher".[7] Graham was born in England, where he and his family maintain permanent residence. However he is also a citizen of the United States, where he was educated, lived, and worked until 2016."""
prompts = ["Who is Paul Graham?", "What is Paul Graham known for?" , "Is paul graham a writer?" , "Where does Paul Graham live?", "What has Paul Graham done?"]
messages =  [[{"role": "user", "content": context + "\n" + prompt}] for prompt in prompts] # pass in a list of messages we want to test
result = testing_batch_completion(models=models, messages=messages)
```

## Visualize the data


```python
import pandas as pd

# Create an empty list to store the row data
table_data = []

# Iterate through the list and extract the required data
for item in result:
    prompt = item['prompt'][0]['content'].replace(context, "") # clean the prompt for easy comparison
    model = item['response']['model']
    response = item['response']['choices'][0]['message']['content']
    table_data.append([prompt, model, response])

# Create a DataFrame from the table data
df = pd.DataFrame(table_data, columns=['Prompt', 'Model Name', 'Response'])

# Pivot the DataFrame to get the desired table format
table = df.pivot(index='Prompt', columns='Model Name', values='Response')
table
```






  <div id="df-8c39923a-ebb1-42ef-b7a0-5edd2535cb37">
    <div class="colab-df-container">
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
      <th>Model Name</th>
      <th>claude-instant-1</th>
      <th>gpt-3.5-turbo-0613</th>
      <th>gpt-3.5-turbo-16k-0613</th>
      <th>gpt-4-0613</th>
      <th>replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781</th>
    </tr>
    <tr>
      <th>Prompt</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>\nIs paul graham a writer?</th>
      <td>Yes, Paul Graham is considered a writer in ad...</td>
      <td>Yes, Paul Graham is a writer. He has written s...</td>
      <td>Yes, Paul Graham is a writer. He has authored ...</td>
      <td>Yes, Paul Graham is a writer. He is an essayis...</td>
      <td>Yes, Paul Graham is an author. According to t...</td>
    </tr>
    <tr>
      <th>\nWhat has Paul Graham done?</th>
      <td>Paul Graham has made significant contribution...</td>
      <td>Paul Graham has achieved several notable accom...</td>
      <td>Paul Graham has made significant contributions...</td>
      <td>Paul Graham is known for his work on the progr...</td>
      <td>Paul Graham has had a diverse career in compu...</td>
    </tr>
    <tr>
      <th>\nWhat is Paul Graham known for?</th>
      <td>Paul Graham is known for several things:\n\n-...</td>
      <td>Paul Graham is known for his work on the progr...</td>
      <td>Paul Graham is known for his work on the progr...</td>
      <td>Paul Graham is known for his work on the progr...</td>
      <td>Paul Graham is known for many things, includi...</td>
    </tr>
    <tr>
      <th>\nWhere does Paul Graham live?</th>
      <td>Based on the information provided:\n\n- Paul ...</td>
      <td>According to the given information, Paul Graha...</td>
      <td>Paul Graham currently lives in England, where ...</td>
      <td>The text does not provide a current place of r...</td>
      <td>Based on the information provided, Paul Graha...</td>
    </tr>
    <tr>
      <th>\nWho is Paul Graham?</th>
      <td>Paul Graham is an influential computer scient...</td>
      <td>Paul Graham is an English computer scientist, ...</td>
      <td>Paul Graham is an English computer scientist, ...</td>
      <td>Paul Graham is an English computer scientist, ...</td>
      <td>Paul Graham is an English computer scientist,...</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8c39923a-ebb1-42ef-b7a0-5edd2535cb37')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>



    <div id="df-4d5c5cee-4f56-4ad2-b181-59c3ec519d1f">
      <button class="colab-df-quickchart" onclick="quickchart('df-4d5c5cee-4f56-4ad2-b181-59c3ec519d1f')"
              title="Suggest charts."
              style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>
    </div>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

    <script>
      async function quickchart(key) {
        const containerElement = document.querySelector('#' + key);
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      }
    </script>


      <script>

function displayQuickchartButton(domScope) {
  let quickchartButtonEl =
    domScope.querySelector('#df-4d5c5cee-4f56-4ad2-b181-59c3ec519d1f button.colab-df-quickchart');
  quickchartButtonEl.style.display =
    google.colab.kernel.accessAllowed ? 'block' : 'none';
}

        displayQuickchartButton(document);
      </script>
      <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8c39923a-ebb1-42ef-b7a0-5edd2535cb37 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8c39923a-ebb1-42ef-b7a0-5edd2535cb37');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Load Test endpoint

Run 100+ simultaneous queries across multiple providers to see when they fail + impact on latency


```python
models=["gpt-3.5-turbo", "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", "claude-instant-1"]
context = """Paul Graham (/ɡræm/; born 1964)[3] is an English computer scientist, essayist, entrepreneur, venture capitalist, and author. He is best known for his work on the programming language Lisp, his former startup Viaweb (later renamed Yahoo! Store), cofounding the influential startup accelerator and seed capital firm Y Combinator, his essays, and Hacker News. He is the author of several computer programming books, including: On Lisp,[4] ANSI Common Lisp,[5] and Hackers & Painters.[6] Technology journalist Steven Levy has described Graham as a "hacker philosopher".[7] Graham was born in England, where he and his family maintain permanent residence. However he is also a citizen of the United States, where he was educated, lived, and worked until 2016."""
prompt = "Where does Paul Graham live?"
final_prompt = context + prompt
result = load_test_model(models=models, prompt=final_prompt, num_calls=5)
```

## Visualize the data


```python
import matplotlib.pyplot as plt

## calculate avg response time
unique_models = set(result["response"]['model'] for result in result["results"])
model_dict = {model: {"response_time": []} for model in unique_models}
for completion_result in result["results"]:
    model_dict[completion_result["response"]["model"]]["response_time"].append(completion_result["response_time"])

avg_response_time = {}
for model, data in model_dict.items():
    avg_response_time[model] = sum(data["response_time"]) / len(data["response_time"])

models = list(avg_response_time.keys())
response_times = list(avg_response_time.values())

plt.bar(models, response_times)
plt.xlabel('Model', fontsize=10)
plt.ylabel('Average Response Time')
plt.title('Average Response Times for each Model')

plt.xticks(models, [model[:15]+'...' if len(model) > 15 else model for model in models], rotation=45)
plt.show()
```


    
![png](output_11_0.png)
    


# Duration Test endpoint

Run load testing for 2 mins. Hitting endpoints with 100+ queries every 15 seconds.


```python
models=["gpt-3.5-turbo", "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", "claude-instant-1"]
context = """Paul Graham (/ɡræm/; born 1964)[3] is an English computer scientist, essayist, entrepreneur, venture capitalist, and author. He is best known for his work on the programming language Lisp, his former startup Viaweb (later renamed Yahoo! Store), cofounding the influential startup accelerator and seed capital firm Y Combinator, his essays, and Hacker News. He is the author of several computer programming books, including: On Lisp,[4] ANSI Common Lisp,[5] and Hackers & Painters.[6] Technology journalist Steven Levy has described Graham as a "hacker philosopher".[7] Graham was born in England, where he and his family maintain permanent residence. However he is also a citizen of the United States, where he was educated, lived, and worked until 2016."""
prompt = "Where does Paul Graham live?"
final_prompt = context + prompt
result = load_test_model(models=models, prompt=final_prompt, num_calls=100, interval=15, duration=120)
```


```python
import matplotlib.pyplot as plt

## calculate avg response time
unique_models = set(unique_result["response"]['model'] for unique_result in result[0]["results"])
model_dict = {model: {"response_time": []} for model in unique_models}
for iteration in result:
  for completion_result in iteration["results"]:
    model_dict[completion_result["response"]["model"]]["response_time"].append(completion_result["response_time"])

avg_response_time = {}
for model, data in model_dict.items():
    avg_response_time[model] = sum(data["response_time"]) / len(data["response_time"])

models = list(avg_response_time.keys())
response_times = list(avg_response_time.values())

plt.bar(models, response_times)
plt.xlabel('Model', fontsize=10)
plt.ylabel('Average Response Time')
plt.title('Average Response Times for each Model')

plt.xticks(models, [model[:15]+'...' if len(model) > 15 else model for model in models], rotation=45)
plt.show()
```


    
![png](output_14_0.png)
    

