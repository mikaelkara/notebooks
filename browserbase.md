# Browserbase

[Browserbase](https://browserbase.com) is a developer platform to reliably run, manage, and monitor headless browsers.

Power your AI data retrievals with:
- [Serverless Infrastructure](https://docs.browserbase.com/under-the-hood) providing reliable browsers to extract data from complex UIs
- [Stealth Mode](https://docs.browserbase.com/features/stealth-mode) with included fingerprinting tactics and automatic captcha solving
- [Session Debugger](https://docs.browserbase.com/features/sessions) to inspect your Browser Session with networks timeline and logs
- [Live Debug](https://docs.browserbase.com/guides/session-debug-connection/browser-remote-control) to quickly debug your automation

## Installation and Setup

- Get an API key and Project ID from [browserbase.com](https://browserbase.com) and set it in environment variables (`BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID`).
- Install the [Browserbase SDK](http://github.com/browserbase/python-sdk):


```python
%pip install browserbase
```

## Loading documents

You can load webpages into LangChain using `BrowserbaseLoader`. Optionally, you can set `text_content` parameter to convert the pages to text-only representation.


```python
from langchain_community.document_loaders import BrowserbaseLoader
```


```python
loader = BrowserbaseLoader(
    urls=[
        "https://example.com",
    ],
    # Text mode
    text_content=False,
)

docs = loader.load()
print(docs[0].page_content[:61])
```

### Loader Options

- `urls` Required. A list of URLs to fetch.
- `text_content` Retrieve only text content. Default is `False`.
- `api_key` Optional. Browserbase API key. Default is `BROWSERBASE_API_KEY` env variable.
- `project_id` Optional. Browserbase Project ID. Default is `BROWSERBASE_PROJECT_ID` env variable.
- `session_id` Optional. Provide an existing Session ID.
- `proxy` Optional. Enable/Disable Proxies.

## Loading images

You can also load screenshots of webpages (as bytes) for multi-modal models.

Full example using GPT-4V:


```python
from browserbase import Browserbase
from browserbase.helpers.gpt4 import GPT4VImage, GPT4VImageDetail
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=256)
browser = Browserbase()

screenshot = browser.screenshot("https://browserbase.com")

result = chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "What color is the logo?"},
                GPT4VImage(screenshot, GPT4VImageDetail.auto),
            ]
        )
    ]
)

print(result.content)
```
