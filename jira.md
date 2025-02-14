# Jira Toolkit

This notebook goes over how to use the `Jira` toolkit.

The `Jira` toolkit allows agents to interact with a given Jira instance, performing actions such as searching for issues and creating issues, the tool wraps the atlassian-python-api library, for more see: https://atlassian-python-api.readthedocs.io/jira.html

## Installation and setup

To use this tool, you must first set as environment variables:
    JIRA_API_TOKEN
    JIRA_USERNAME
    JIRA_INSTANCE_URL
    JIRA_CLOUD


```python
%pip install --upgrade --quiet  atlassian-python-api
```


```python
%pip install -qU langchain-community langchain_openai
```


```python
import os

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from langchain_openai import OpenAI
```


```python
os.environ["JIRA_API_TOKEN"] = "abc"
os.environ["JIRA_USERNAME"] = "123"
os.environ["JIRA_INSTANCE_URL"] = "https://jira.atlassian.com"
os.environ["OPENAI_API_KEY"] = "xyz"
os.environ["JIRA_CLOUD"] = "True"
```


```python
llm = OpenAI(temperature=0)
jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
```

## Tool usage

Let's see what individual tools are in the Jira toolkit:


```python
[(tool.name, tool.description) for tool in toolkit.get_tools()]
```




    [('JQL Query',
      '\n    This tool is a wrapper around atlassian-python-api\'s Jira jql API, useful when you need to search for Jira issues.\n    The input to this tool is a JQL query string, and will be passed into atlassian-python-api\'s Jira `jql` function,\n    For example, to find all the issues in project "Test" assigned to the me, you would pass in the following string:\n    project = Test AND assignee = currentUser()\n    or to find issues with summaries that contain the word "test", you would pass in the following string:\n    summary ~ \'test\'\n    '),
     ('Get Projects',
      "\n    This tool is a wrapper around atlassian-python-api's Jira project API, \n    useful when you need to fetch all the projects the user has access to, find out how many projects there are, or as an intermediary step that involv searching by projects. \n    there is no input to this tool.\n    "),
     ('Create Issue',
      '\n    This tool is a wrapper around atlassian-python-api\'s Jira issue_create API, useful when you need to create a Jira issue. \n    The input to this tool is a dictionary specifying the fields of the Jira issue, and will be passed into atlassian-python-api\'s Jira `issue_create` function.\n    For example, to create a low priority task called "test issue" with description "test description", you would pass in the following dictionary: \n    {{"summary": "test issue", "description": "test description", "issuetype": {{"name": "Task"}}, "priority": {{"name": "Low"}}}}\n    '),
     ('Catch all Jira API call',
      '\n    This tool is a wrapper around atlassian-python-api\'s Jira API.\n    There are other dedicated tools for fetching all projects, and creating and searching for issues, \n    use this tool if you need to perform any other actions allowed by the atlassian-python-api Jira API.\n    The input to this tool is a dictionary specifying a function from atlassian-python-api\'s Jira API, \n    as well as a list of arguments and dictionary of keyword arguments to pass into the function.\n    For example, to get all the users in a group, while increasing the max number of results to 100, you would\n    pass in the following dictionary: {{"function": "get_all_users_from_group", "args": ["group"], "kwargs": {{"limit":100}} }}\n    or to find out how many projects are in the Jira instance, you would pass in the following string:\n    {{"function": "projects"}}\n    For more information on the Jira API, refer to https://atlassian-python-api.readthedocs.io/jira.html\n    '),
     ('Create confluence page',
      'This tool is a wrapper around atlassian-python-api\'s Confluence \natlassian-python-api API, useful when you need to create a Confluence page. The input to this tool is a dictionary \nspecifying the fields of the Confluence page, and will be passed into atlassian-python-api\'s Confluence `create_page` \nfunction. For example, to create a page in the DEMO space titled "This is the title" with body "This is the body. You can use \n<strong>HTML tags</strong>!", you would pass in the following dictionary: {{"space": "DEMO", "title":"This is the \ntitle","body":"This is the body. You can use <strong>HTML tags</strong>!"}} ')]




```python
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```


```python
agent.run("make a new issue in project PW to remind me to make more fried rice")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to create an issue in project PW
    Action: Create Issue
    Action Input: {"summary": "Make more fried rice", "description": "Reminder to make more fried rice", "issuetype": {"name": "Task"}, "priority": {"name": "Low"}, "project": {"key": "PW"}}[0m
    Observation: [38;5;200m[1;3mNone[0m
    Thought:[32;1m[1;3m I now know the final answer
    Final Answer: A new issue has been created in project PW with the summary "Make more fried rice" and description "Reminder to make more fried rice".[0m
    
    [1m> Finished chain.[0m
    




    'A new issue has been created in project PW with the summary "Make more fried rice" and description "Reminder to make more fried rice".'


