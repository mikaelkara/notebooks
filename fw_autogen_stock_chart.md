[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1peT_mx4XCYQe1o2O6NmK0ITDJj2Ali8o?usp=sharing)

## Preable - Install Deps

There are only a few dependencies for this tutorial.


```python
!pip3 install pyautogen openai fireworks-ai matplotlib opencv-python yfinance
```

    Requirement already satisfied: pyautogen in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.3.1)
    Requirement already satisfied: openai in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (1.47.0)
    Requirement already satisfied: fireworks-ai in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.15.3)
    Requirement already satisfied: matplotlib in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (3.9.2)
    Requirement already satisfied: opencv-python in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (4.10.0.84)
    Requirement already satisfied: yfinance in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (0.2.40)
    Requirement already satisfied: diskcache in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (5.6.3)
    Requirement already satisfied: docker in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (7.1.0)
    Requirement already satisfied: flaml in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (2.3.2)
    Requirement already satisfied: numpy<2,>=1.17.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (1.26.4)
    Requirement already satisfied: packaging in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (23.2)
    Requirement already satisfied: pydantic!=2.6.0,<3,>=1.10 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (2.9.2)
    Requirement already satisfied: python-dotenv in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (1.0.1)
    Requirement already satisfied: termcolor in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (2.5.0)
    Requirement already satisfied: tiktoken in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pyautogen) (0.7.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.6.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (0.27.2)
    Requirement already satisfied: jiter<1,>=0.4.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (0.4.2)
    Requirement already satisfied: sniffio in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.66.5)
    Requirement already satisfied: typing-extensions<5,>=4.11 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from openai) (4.12.2)
    Requirement already satisfied: httpx-sse in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (0.4.0)
    Requirement already satisfied: Pillow in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from fireworks-ai) (10.4.0)
    Requirement already satisfied: contourpy>=1.0.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from matplotlib) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from matplotlib) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: pyparsing>=2.3.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from matplotlib) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.7 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: pandas>=1.3.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (2.2.3)
    Requirement already satisfied: requests>=2.31 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (2.32.3)
    Requirement already satisfied: multitasking>=0.0.7 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (0.0.11)
    Requirement already satisfied: lxml>=4.9.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (5.3.0)
    Requirement already satisfied: platformdirs>=2.0.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (4.3.6)
    Requirement already satisfied: pytz>=2022.5 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (2024.2)
    Requirement already satisfied: frozendict>=2.3.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (2.4.4)
    Requirement already satisfied: peewee>=3.16.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (3.17.6)
    Requirement already satisfied: beautifulsoup4>=4.11.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (4.12.3)
    Requirement already satisfied: html5lib>=1.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from yfinance) (1.1)
    Requirement already satisfied: idna>=2.8 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from anyio<5,>=3.5.0->openai) (3.10)
    Requirement already satisfied: soupsieve>1.2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from beautifulsoup4>=4.11.1->yfinance) (2.6)
    Requirement already satisfied: six>=1.9 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from html5lib>=1.1->yfinance) (1.16.0)
    Requirement already satisfied: webencodings in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from html5lib>=1.1->yfinance) (0.5.1)
    Requirement already satisfied: certifi in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (2024.8.30)
    Requirement already satisfied: httpcore==1.* in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpx<1,>=0.23.0->openai) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: tzdata>=2022.7 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pandas>=1.3.0->yfinance) (2024.1)
    Requirement already satisfied: annotated-types>=0.6.0 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic!=2.6.0,<3,>=1.10->pyautogen) (0.7.0)
    Requirement already satisfied: pydantic-core==2.23.4 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from pydantic!=2.6.0,<3,>=1.10->pyautogen) (2.23.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests>=2.31->yfinance) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from requests>=2.31->yfinance) (2.2.3)
    Requirement already satisfied: regex>=2022.1.18 in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (from tiktoken->pyautogen) (2023.12.25)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m24.2[0m[39;49m -> [0m[32;49m24.3.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip3 install --upgrade pip[0m
    

## Introduction -  Stock Chart Generator

In this example we will use AutoGen framework to construct an agent that is capable of generating an stock price charts.

For this demo, we are going to utilize [function calling](https://readme.fireworks.ai/docs/function-calling) feature launched by Fireworks. We initialize two agents - `UserProxyAgent` and `AssistantAgent`. The `AssistantAgent` is given the ability to issue a call for the provided functions but not execute them while `UserProxyAgent` is given the ability to execute the function calls issues by the `AssistantAgent`. In order to achieve this behaviour we use decorators provided by AutoGen library called `register_for_llm` and `register_for_execution`. Using these decorators allows us to easily define python functions and turn them into [JSON Spec](https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat) needed by function calling API.

Finally, we setup system prompt for both the agents. We ask the `AssistantAgent` to be a helpful agent & focus on generating the correct function calls and we leave `UserProxyAgent` as is. For more advanced use cases we can ask `UserProxyAgent` to be a plan generator.


```python
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import autogen
import pandas as pd
import requests  # to perform HTTP requests
import yfinance as yf
from matplotlib import pyplot as plt
from openai import OpenAI
from typing_extensions import Annotated
```

## Setup

In order to use the Fireworks AI function calling model, you must first obtain Fireworks API Keys. If you don't already have one, you can one by following the instructions [here](https://readme.fireworks.ai/docs/quickstart). Replace FW_API_KEY with your obtained key.


```python
os.environ["AUTOGEN_USE_DOCKER"] = "0"
```


```python
FW_API_KEY = "FW_API_KEY"

config_list = [
    {
        'api_key': FW_API_KEY,
        'base_url': 'https://api.fireworks.ai/inference/v1',
        'model': 'accounts/fireworks/models/firefunction-v1'
    }
]
```

## Configure Tools

For this notebook, we are going to use 2 sets of tools
1. **Get Prices** - We will use the [yfinance](https://pypi.org/project/yfinance/) package to obtain stock prices for a given ticker. The stock prices can be obtained over a custom time range and would be saved as a time series in a csv file.
2. **Show Chart** - This tool, given a valid file path, will draw a time series.


Using the AutoGen framework we demonstrate the co-operative nature of agents working with each other to accomplish a complex task. This tutorial can be extended to perform more complicated tasks such as generating stock price charts etc.


```python
llm_config = {
    "config_list": config_list,
    "timeout": 120,
    "temperature": 0
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message=" If the user request HAS been addressed, respond with a summary of the result. The summary MUST end with the word TERMINATE. You are a helpful AI assistant with access to functions. Use them if required.",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and (
        x.get("content", "").rstrip().endswith("TERMINATE") 
        or x.get("content", "").rstrip().endswith("TERMINATE.")
    ),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False  # Add this line
    },
)


@user_proxy.register_for_execution()
@chatbot.register_for_llm(
    name="get_prices",
    description="Helper function to obtain stock price history of a company over specified period. The price information is written to a file and the path of the file is returned. The file is csv and contains following columns - Date,Open,High,Low,Close,Volume,Dividends,Stock Splits",
)
def get_prices(
    ticker: Annotated[str, "Stock ticker for a company"],
    period: Annotated[
        str,
        "data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max",
    ],
) -> Annotated[
    str, "File which contains the price of the a ticker, each price in a new line"
]:
    # Generate a random UUID as the file name
    file_name: str = "/tmp/" + str(uuid.uuid4()) + ".csv"
    file_path = Path(file_name)

    tk = yf.Ticker(ticker=ticker)
    prices = tk.history(period=period)

    with open(file_path, "w") as f:
        prices.to_csv(f)
    return file_name

@user_proxy.register_for_execution()
@chatbot.register_for_llm(
    name="draw_time_series",
    description="Plot time series chart given stock prices of multiple stock tickers in a csv file.",
)
def plot_time_series(
    title: Annotated[str, "Title of the plot that is generated"],
    file_name: Annotated[str, "File to which to save the chart"],
    csv_files: Annotated[
        Dict[str, str],
        "Dictionary from stock ticker to CSV file that contain stock prices. The csv files should have schema - Date,Open,High,Low,Close,Volume,Dividends,Stock Splits",
    ],
    date_col: Annotated[str, "Column name in csv that contains the date"] = "Date",
    price_col: Annotated[str, "Column name that contains the price"] = "Close",
) -> str:
    file_name = "/tmp/" + file_name
    for label, csv_file in csv_files.items():
        # Load the data into pandas dataframes
        price_data = pd.read_csv(csv_file, index_col=date_col, parse_dates=True)

        # Plot NVDA stock price data
        plt.plot(price_data.index, price_data["Close"], label=label)

    plt.xlabel(date_col)
    plt.ylabel(price_col)
    plt.title(title)
    plt.legend()

    # Save the plot to a file
    plt.savefig(file_name)

    print(f"Plot saved to {file_name}")

    return file_name


user_proxy.initiate_chat(
    chatbot,
    message="Get stock prices of NVDA stock price YTD. Then plot them on a chart.",
    clear_history=True
)
```

    [33muser_proxy[0m (to chatbot):
    
    Get stock prices of NVDA stock price YTD. Then plot them on a chart.
    
    --------------------------------------------------------------------------------
    [autogen.oai.client: 11-08 16:27:08] {409} WARNING - Model accounts/fireworks/models/firefunction-v1 is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
    [33mchatbot[0m (to user_proxy):
    
    [32m***** Suggested tool call (call_Msjlx4BDdoZrOlpd4aE5wVzm): get_prices *****[0m
    Arguments: 
    {"ticker": "NVDA", "period": "ytd"}
    [32m***************************************************************************[0m
    
    --------------------------------------------------------------------------------
    [35m
    >>>>>>>> EXECUTING FUNCTION get_prices...[0m
    [33muser_proxy[0m (to chatbot):
    
    [33muser_proxy[0m (to chatbot):
    
    [32m***** Response from calling tool (call_Msjlx4BDdoZrOlpd4aE5wVzm) *****[0m
    /tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv
    [32m**********************************************************************[0m
    
    --------------------------------------------------------------------------------
    [autogen.oai.client: 11-08 16:27:11] {409} WARNING - Model accounts/fireworks/models/firefunction-v1 is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
    [33mchatbot[0m (to user_proxy):
    
    The stock prices of NVDA stock for the year to date (YTD) have been obtained and saved to a file at /tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv. 
    
    Now, I will plot the time series chart for the stock prices. 
    [32m***** Suggested tool call (call_M9CvkyXQKioKJWz0icwPA5WA): draw_time_series *****[0m
    Arguments: 
    {"title": "NVDA Stock Prices YTD", "file_name": "/tmp/nvda_ytd.png", "csv_files": {"NVDA": "/tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv"}, "date_col": "Date", "price_col": "Close"}
    [32m*********************************************************************************[0m
    
    --------------------------------------------------------------------------------
    [35m
    >>>>>>>> EXECUTING FUNCTION draw_time_series...[0m
    [33muser_proxy[0m (to chatbot):
    
    [33muser_proxy[0m (to chatbot):
    
    [32m***** Response from calling tool (call_M9CvkyXQKioKJWz0icwPA5WA) *****[0m
    Error: [Errno 2] No such file or directory: '/private/tmp/tmp/nvda_ytd.png'
    [32m**********************************************************************[0m
    
    --------------------------------------------------------------------------------
    [autogen.oai.client: 11-08 16:27:13] {409} WARNING - Model accounts/fireworks/models/firefunction-v1 is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
    [33mchatbot[0m (to user_proxy):
    
    The stock prices of NVDA stock for the year to date (YTD) have been obtained and saved to a file at /tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv. However, there was an error while plotting the time series chart for the stock prices. The error message is: [Errno 2] No such file or directory: '/private/tmp/tmp/nvda_ytd.png'. TERMINATE.
    
    --------------------------------------------------------------------------------
    




    ChatResult(chat_id=None, chat_history=[{'content': 'Get stock prices of NVDA stock price YTD. Then plot them on a chart.', 'role': 'assistant', 'name': 'user_proxy'}, {'tool_calls': [{'id': 'call_Msjlx4BDdoZrOlpd4aE5wVzm', 'function': {'arguments': '{"ticker": "NVDA", "period": "ytd"}', 'name': 'get_prices'}, 'type': 'function', 'index': 0}], 'content': None, 'role': 'assistant'}, {'content': '/tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv', 'tool_responses': [{'tool_call_id': 'call_Msjlx4BDdoZrOlpd4aE5wVzm', 'role': 'tool', 'content': '/tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv'}], 'role': 'tool', 'name': 'user_proxy'}, {'content': 'The stock prices of NVDA stock for the year to date (YTD) have been obtained and saved to a file at /tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv. \n\nNow, I will plot the time series chart for the stock prices. ', 'tool_calls': [{'id': 'call_M9CvkyXQKioKJWz0icwPA5WA', 'function': {'arguments': '{"title": "NVDA Stock Prices YTD", "file_name": "/tmp/nvda_ytd.png", "csv_files": {"NVDA": "/tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv"}, "date_col": "Date", "price_col": "Close"}', 'name': 'draw_time_series'}, 'type': 'function', 'index': 0}], 'role': 'assistant'}, {'content': "Error: [Errno 2] No such file or directory: '/private/tmp/tmp/nvda_ytd.png'", 'tool_responses': [{'tool_call_id': 'call_M9CvkyXQKioKJWz0icwPA5WA', 'role': 'tool', 'content': "Error: [Errno 2] No such file or directory: '/private/tmp/tmp/nvda_ytd.png'"}], 'role': 'tool', 'name': 'user_proxy'}, {'content': "The stock prices of NVDA stock for the year to date (YTD) have been obtained and saved to a file at /tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv. However, there was an error while plotting the time series chart for the stock prices. The error message is: [Errno 2] No such file or directory: '/private/tmp/tmp/nvda_ytd.png'. TERMINATE.", 'role': 'user', 'name': 'chatbot'}], summary="The stock prices of NVDA stock for the year to date (YTD) have been obtained and saved to a file at /tmp/eef5738c-81d9-4cca-baef-c6a13f32186a.csv. However, there was an error while plotting the time series chart for the stock prices. The error message is: [Errno 2] No such file or directory: '/private/tmp/tmp/nvda_ytd.png'. .", cost={'usage_including_cached_inference': {'total_cost': 0, 'accounts/fireworks/models/firefunction-v1': {'cost': 0, 'prompt_tokens': 2503, 'completion_tokens': 336, 'total_tokens': 2839}}, 'usage_excluding_cached_inference': {'total_cost': 0, 'accounts/fireworks/models/firefunction-v1': {'cost': 0, 'prompt_tokens': 2503, 'completion_tokens': 336, 'total_tokens': 2839}}}, human_input=[])




    
![png](output_9_2.png)
    

