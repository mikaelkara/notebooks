# File System

LangChain provides tools for interacting with a local file system out of the box. This notebook walks through some of them.

**Note:** these tools are not recommended for use outside a sandboxed environment! 


```python
%pip install -qU langchain-community
```

First, we'll import the tools.


```python
from tempfile import TemporaryDirectory

from langchain_community.agent_toolkits import FileManagementToolkit

# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory()
```

## The FileManagementToolkit

If you want to provide all the file tooling to your agent, it's easy to do so with the toolkit. We'll pass the temporary directory in as a root directory as a workspace for the LLM.

It's recommended to always pass in a root directory, since without one, it's easy for the LLM to pollute the working directory, and without one, there isn't any validation against
straightforward prompt injection.


```python
toolkit = FileManagementToolkit(
    root_dir=str(working_directory.name)
)  # If you don't provide a root_dir, operations will default to the current working directory
toolkit.get_tools()
```




    [CopyFileTool(root_dir='/tmp/tmprdvsw3tg'),
     DeleteFileTool(root_dir='/tmp/tmprdvsw3tg'),
     FileSearchTool(root_dir='/tmp/tmprdvsw3tg'),
     MoveFileTool(root_dir='/tmp/tmprdvsw3tg'),
     ReadFileTool(root_dir='/tmp/tmprdvsw3tg'),
     WriteFileTool(root_dir='/tmp/tmprdvsw3tg'),
     ListDirectoryTool(root_dir='/tmp/tmprdvsw3tg')]



### Selecting File System Tools

If you only want to select certain tools, you can pass them in as arguments when initializing the toolkit, or you can individually initialize the desired tools.


```python
tools = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
tools
```




    [ReadFileTool(root_dir='/tmp/tmprdvsw3tg'),
     WriteFileTool(root_dir='/tmp/tmprdvsw3tg'),
     ListDirectoryTool(root_dir='/tmp/tmprdvsw3tg')]




```python
read_tool, write_tool, list_tool = tools
write_tool.invoke({"file_path": "example.txt", "text": "Hello World!"})
```




    'File written successfully to example.txt.'




```python
# List files in the working directory
list_tool.invoke({})
```




    'example.txt'




```python

```
