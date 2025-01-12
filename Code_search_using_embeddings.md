## Code search using embeddings

This notebook shows how Ada embeddings can be used to implement semantic code search. For this demonstration, we use our own [openai-python code repository](https://github.com/openai/openai-python). We implement a simple version of file parsing and extracting of functions from python files, which can be embedded, indexed, and queried.

### Helper Functions

We first setup some simple parsing functions that allow us to extract important information from our codebase.


```python
import pandas as pd
from pathlib import Path

DEF_PREFIXES = ['def ', 'async def ']
NEWLINE = '\n'

def get_function_name(code):
    """
    Extract function name from a line beginning with 'def' or 'async def'.
    """
    for prefix in DEF_PREFIXES:
        if code.startswith(prefix):
            return code[len(prefix): code.index('(')]


def get_until_no_space(all_lines, i):
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, len(all_lines)):
        if len(all_lines[j]) == 0 or all_lines[j][0] in [' ', '\t', ')']:
            ret.append(all_lines[j])
        else:
            break
    return NEWLINE.join(ret)


def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    with open(filepath, 'r') as file:
        all_lines = file.read().replace('\r', NEWLINE).split(NEWLINE)
        for i, l in enumerate(all_lines):
            for prefix in DEF_PREFIXES:
                if l.startswith(prefix):
                    code = get_until_no_space(all_lines, i)
                    function_name = get_function_name(code)
                    yield {
                        'code': code,
                        'function_name': function_name,
                        'filepath': filepath,
                    }
                    break


def extract_functions_from_repo(code_root):
    """
    Extract all .py functions from the repository.
    """
    code_files = list(code_root.glob('**/*.py'))

    num_files = len(code_files)
    print(f'Total number of .py files: {num_files}')

    if num_files == 0:
        print('Verify openai-python repo exists and code_root is set correctly.')
        return None

    all_funcs = [
        func
        for code_file in code_files
        for func in get_functions(str(code_file))
    ]

    num_funcs = len(all_funcs)
    print(f'Total number of functions extracted: {num_funcs}')

    return all_funcs
```

# Data Loading

We'll first load the openai-python folder and extract the needed information using the functions we defined above.


```python
# Set user root directory to the 'openai-python' repository
root_dir = Path.home()

# Assumes the 'openai-python' repository exists in the user's root directory
code_root = root_dir / 'openai-python'

# Extract all functions from the repository
all_funcs = extract_functions_from_repo(code_root)
```

    Total number of .py files: 51
    Total number of functions extracted: 97
    

Now that we have our content, we can pass the data to the `text-embedding-3-small` model and get back our vector embeddings.


```python
from utils.embeddings_utils import get_embedding

df = pd.DataFrame(all_funcs)
df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df['filepath'] = df['filepath'].map(lambda x: Path(x).relative_to(code_root))
df.to_csv("data/code_search_openai-python.csv", index=False)
df.head()
```




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
      <th></th>
      <th>code</th>
      <th>function_name</th>
      <th>filepath</th>
      <th>code_embedding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>def _console_log_level():\n    if openai.log i...</td>
      <td>_console_log_level</td>
      <td>openai/util.py</td>
      <td>[0.005937571171671152, 0.05450401455163956, 0....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>def log_debug(message, **params):\n    msg = l...</td>
      <td>log_debug</td>
      <td>openai/util.py</td>
      <td>[0.017557814717292786, 0.05647840350866318, -0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>def log_info(message, **params):\n    msg = lo...</td>
      <td>log_info</td>
      <td>openai/util.py</td>
      <td>[0.022524144500494003, 0.06219055876135826, -0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>def log_warn(message, **params):\n    msg = lo...</td>
      <td>log_warn</td>
      <td>openai/util.py</td>
      <td>[0.030524108558893204, 0.0667714849114418, -0....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>def logfmt(props):\n    def fmt(key, val):\n  ...</td>
      <td>logfmt</td>
      <td>openai/util.py</td>
      <td>[0.05337328091263771, 0.03697286546230316, -0....</td>
    </tr>
  </tbody>
</table>
</div>



### Testing

Let's test our endpoint with some simple queries. If you're familiar with the `openai-python` repository, you'll see that we're able to easily find functions we're looking for only a simple English description.

We define a search_functions method that takes our data that contains our embeddings, a query string, and some other configuration options. The process of searching our database works like such:

1. We first embed our query string (code_query) with `text-embedding-3-small`. The reasoning here is that a query string like 'a function that reverses a string' and a function like 'def reverse(string): return string[::-1]' will be very similar when embedded.
2. We then calculate the cosine similarity between our query string embedding and all data points in our database. This gives a distance between each point and our query.
3. We finally sort all of our data points by their distance to our query string and return the number of results requested in the function parameters. 


```python
from utils.embeddings_utils import cosine_similarity

def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
    embedding = get_embedding(code_query, model='text-embedding-3-small')
    df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)

    if pprint:
        for r in res.iterrows():
            print(f"{r[1].filepath}:{r[1].function_name}  score={round(r[1].similarities, 3)}")
            print("\n".join(r[1].code.split("\n")[:n_lines]))
            print('-' * 70)

    return res
```


```python
res = search_functions(df, 'fine-tuning input data validation logic', n=3)
```

    openai/validators.py:format_inferrer_validator  score=0.453
    def format_inferrer_validator(df):
        """
        This validator will infer the likely fine-tuning format of the data, and display it to the user if it is classification.
        It will also suggest to use ada and explain train/validation split benefits.
        """
        ft_type = infer_task_type(df)
        immediate_msg = None
    ----------------------------------------------------------------------
    openai/validators.py:infer_task_type  score=0.37
    def infer_task_type(df):
        """
        Infer the likely fine-tuning task type from the data
        """
        CLASSIFICATION_THRESHOLD = 3  # min_average instances of each class
        if sum(df.prompt.str.len()) == 0:
            return "open-ended generation"
    ----------------------------------------------------------------------
    openai/validators.py:apply_validators  score=0.369
    def apply_validators(
        df,
        fname,
        remediation,
        validators,
        auto_accept,
        write_out_file_func,
    ----------------------------------------------------------------------
    


```python
res = search_functions(df, 'find common suffix', n=2, n_lines=10)
```

    openai/validators.py:get_common_xfix  score=0.487
    def get_common_xfix(series, xfix="suffix"):
        """
        Finds the longest common suffix or prefix of all the values in a series
        """
        common_xfix = ""
        while True:
            common_xfixes = (
                series.str[-(len(common_xfix) + 1) :]
                if xfix == "suffix"
                else series.str[: len(common_xfix) + 1]
    ----------------------------------------------------------------------
    openai/validators.py:common_completion_suffix_validator  score=0.449
    def common_completion_suffix_validator(df):
        """
        This validator will suggest to add a common suffix to the completion if one doesn't already exist in case of classification or conditional generation.
        """
        error_msg = None
        immediate_msg = None
        optional_msg = None
        optional_fn = None
    
        ft_type = infer_task_type(df)
    ----------------------------------------------------------------------
    


```python
res = search_functions(df, 'Command line interface for fine-tuning', n=1, n_lines=20)
```

    openai/cli.py:tools_register  score=0.391
    def tools_register(parser):
        subparsers = parser.add_subparsers(
            title="Tools", help="Convenience client side tools"
        )
    
        def help(args):
            parser.print_help()
    
        parser.set_defaults(func=help)
    
        sub = subparsers.add_parser("fine_tunes.prepare_data")
        sub.add_argument(
            "-f",
            "--file",
            required=True,
            help="JSONL, JSON, CSV, TSV, TXT or XLSX file containing prompt-completion examples to be analyzed."
            "This should be the local file path.",
        )
        sub.add_argument(
            "-q",
    ----------------------------------------------------------------------
    
