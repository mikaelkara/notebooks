# Human input chat model

Along with HumanInputLLM, LangChain also provides a pseudo chat model class that can be used for testing, debugging, or educational purposes. This allows you to mock out calls to the chat model and simulate how a human would respond if they received the messages.

In this notebook, we go over how to use this.

We start this with using the HumanInputChatModel in an agent.


```python
from langchain_community.chat_models.human import HumanInputChatModel
```

Since we will use the `WikipediaQueryRun` tool in this notebook, you might need to install the `wikipedia` package if you haven't done so already.


```python
%pip install wikipedia
```

    /Users/mskim58/dev/research/chatbot/github/langchain/.venv/bin/python: No module named pip
    Note: you may need to restart the kernel to use updated packages.
    


```python
from langchain.agents import AgentType, initialize_agent, load_tools
```


```python
tools = load_tools(["wikipedia"])
llm = HumanInputChatModel()
```


```python
agent = initialize_agent(
    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```


```python
agent("What is Bocchi the Rock?")
```

    
    
    [1m> Entering new  chain...[0m
    
     ======= start of message ======= 
    
    
    type: system
    data:
      content: "Answer the following questions as best you can. You have access to the following tools:\n\nWikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.\n\nThe way you use the tools is by specifying a json blob.\nSpecifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).\n\nThe only values that should be in the \"action\" field are: Wikipedia\n\nThe $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:\n\n```\n{\n  \"action\": $TOOL_NAME,\n  \"action_input\": $INPUT\n}\n```\n\nALWAYS use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction:\n```\n$JSON_BLOB\n```\nObservation: the result of the action\n... (this Thought/Action/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Reminder to always use the exact characters `Final Answer` when responding."
      additional_kwargs: {}
    
    ======= end of message ======= 
    
    
    
     ======= start of message ======= 
    
    
    type: human
    data:
      content: 'What is Bocchi the Rock?
    
    
        '
      additional_kwargs: {}
      example: false
    
    ======= end of message ======= 
    
    
    [32;1m[1;3mAction:
    ```
    {
      "action": "Wikipedia",
      "action_input": "What is Bocchi the Rock?"
    }
    ```[0m
    Observation: [36;1m[1;3mPage: Bocchi the Rock!
    Summary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Botchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.
    An anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.
    
    Page: Hitori Bocchi no Marumaru Seikatsu
    Summary: Hitori Bocchi no Marumaru Seikatsu (Japanese: ひとりぼっちの○○生活, lit. "Bocchi Hitori's ____ Life" or "The ____ Life of Being Alone") is a Japanese yonkoma manga series written and illustrated by Katsuwo. It was serialized in ASCII Media Works' Comic Dengeki Daioh "g" magazine from September 2013 to April 2021. Eight tankōbon volumes have been released. An anime television series adaptation by C2C aired from April to June 2019.
    
    Page: Kessoku Band (album)
    Summary: Kessoku Band (Japanese: 結束バンド, Hepburn: Kessoku Bando) is the debut studio album by Kessoku Band, a fictional musical group from the anime television series Bocchi the Rock!, released digitally on December 25, 2022, and physically on CD on December 28 by Aniplex. Featuring vocals from voice actresses Yoshino Aoyama, Sayumi Suzushiro, Saku Mizuno, and Ikumi Hasegawa, the album consists of 14 tracks previously heard in the anime, including a cover of Asian Kung-Fu Generation's "Rockn' Roll, Morning Light Falls on You", as well as newly recorded songs; nine singles preceded the album's physical release. Commercially, Kessoku Band peaked at number one on the Billboard Japan Hot Albums Chart and Oricon Albums Chart, and was certified gold by the Recording Industry Association of Japan.
    
    [0m
    Thought:
     ======= start of message ======= 
    
    
    type: system
    data:
      content: "Answer the following questions as best you can. You have access to the following tools:\n\nWikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.\n\nThe way you use the tools is by specifying a json blob.\nSpecifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).\n\nThe only values that should be in the \"action\" field are: Wikipedia\n\nThe $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:\n\n```\n{\n  \"action\": $TOOL_NAME,\n  \"action_input\": $INPUT\n}\n```\n\nALWAYS use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction:\n```\n$JSON_BLOB\n```\nObservation: the result of the action\n... (this Thought/Action/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Reminder to always use the exact characters `Final Answer` when responding."
      additional_kwargs: {}
    
    ======= end of message ======= 
    
    
    
     ======= start of message ======= 
    
    
    type: human
    data:
      content: "What is Bocchi the Rock?\n\nThis was your previous work (but I haven't seen any of it! I only see what you return as final answer):\nAction:\n```\n{\n  \"action\": \"Wikipedia\",\n  \"action_input\": \"What is Bocchi the Rock?\"\n}\n```\nObservation: Page: Bocchi the Rock!\nSummary: Bocchi the Rock! (ぼっち・ざ・ろっく!, Botchi Za Rokku!) is a Japanese four-panel manga series written and illustrated by Aki Hamaji. It has been serialized in Houbunsha's seinen manga magazine Manga Time Kirara Max since December 2017. Its chapters have been collected in five tankōbon volumes as of November 2022.\nAn anime television series adaptation produced by CloverWorks aired from October to December 2022. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.\n\nPage: Hitori Bocchi no Marumaru Seikatsu\nSummary: Hitori Bocchi no Marumaru Seikatsu (Japanese: ひとりぼっちの○○生活, lit. \"Bocchi Hitori's ____ Life\" or \"The ____ Life of Being Alone\") is a Japanese yonkoma manga series written and illustrated by Katsuwo. It was serialized in ASCII Media Works' Comic Dengeki Daioh \"g\" magazine from September 2013 to April 2021. Eight tankōbon volumes have been released. An anime television series adaptation by C2C aired from April to June 2019.\n\nPage: Kessoku Band (album)\nSummary: Kessoku Band (Japanese: 結束バンド, Hepburn: Kessoku Bando) is the debut studio album by Kessoku Band, a fictional musical group from the anime television series Bocchi the Rock!, released digitally on December 25, 2022, and physically on CD on December 28 by Aniplex. Featuring vocals from voice actresses Yoshino Aoyama, Sayumi Suzushiro, Saku Mizuno, and Ikumi Hasegawa, the album consists of 14 tracks previously heard in the anime, including a cover of Asian Kung-Fu Generation's \"Rockn' Roll, Morning Light Falls on You\", as well as newly recorded songs; nine singles preceded the album's physical release. Commercially, Kessoku Band peaked at number one on the Billboard Japan Hot Albums Chart and Oricon Albums Chart, and was certified gold by the Recording Industry Association of Japan.\n\n\nThought:"
      additional_kwargs: {}
      example: false
    
    ======= end of message ======= 
    
    
    [32;1m[1;3mThis finally works.
    Final Answer: Bocchi the Rock! is a four-panel manga series and anime television series. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim.[0m
    
    [1m> Finished chain.[0m
    




    {'input': 'What is Bocchi the Rock?',
     'output': "Bocchi the Rock! is a four-panel manga series and anime television series. The series has been praised for its writing, comedy, characters, and depiction of social anxiety, with the anime's visual creativity receiving acclaim."}


