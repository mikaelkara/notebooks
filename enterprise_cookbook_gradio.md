# Creating Demos with Spaces and Gradio
_Authored by: [Diego Maniloff](https://huggingface.co/dmaniloff)_

## Introduction
In this notebook we will demonstrate how to bring any machine learning model to life using [Gradio](https://www.gradio.app/), a library that allows you to create a web demo from any Python function and share it with the world 🌎!

📚 This notebook covers:
- Building a `Hello, World!` demo: The basics of Gradio
- Moving your demo to Hugging Face Spaces
- Making it interesting: a real-world example that leverages the 🤗 Hub
- Some of the cool "batteries included" features that come with Gradio

⏭️ At the end of this notebook you will find a `Further Reading` list with links to keep going on your own.

## Setup
To get started install the `gradio` library along with `transformers`.


```python
!pip -q install gradio==4.36.1
!pip -q install transformers==4.41.2
```


```python
# the usual shorthand is to import gradio as gr
import gradio as gr
```

## Your first demo: the basics of Gradio
At its core, Gradio turns any Python function into a web interface.

Say we have a simple function that takes `name` and `intensity` as parameters, and returns a string like so:



```python
def greet(name: str, intensity: int) -> str:
    return "Hello, " + name + "!" * int(intensity)
```

If you run this function for the name 'Diego' you will get an output string that looks like this:



```python
print(greet("Diego", 3))
```

    Hello, Diego!!!
    

With Gradio, we can build an interface for this function via the `gr.Interface` class. All we need to do is pass in the `greet` function we created above, and the kinds of inputs and outputs that our function expects:



```python
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"], # the inputs are a text box and a slider ("text" and "slider" are components in Gradio)
    outputs=["text"],          # the output is a text box
)
```

Notice how we passed in `["text", "slider"]` as inputs and `["text"]` as outputs -- these are called [Components](https://www.gradio.app/docs/gradio/introduction) in Gradio.

That's all we need for our first demo. Go ahead and try it out 👇🏼! Type your name into the `name` textbox, slide the intensity that you want, and click `Submit`.


```python
# the launch method will fire up the interface we just created
demo.launch()
```

## Let's make it interesting: a meeting transcription tool
At this point you understand how to take a basic Python function and turn it into
a web-ready demo. However, we only did this for a function that is very simple, a bit boring even!

Let's consider a more interesting example that highlights the very thing that Gradio was built for: demoing cutting-edge machine learning models. A good friend of mine recently asked me for help with an audio recording of an interview she had done. She needed to convert the audio file into a well-organized text summary. How did I help her? I built a Gradio app!

Let's walk through the steps to build the meeting transcription tool. We can think of the process as two parts:


1. Transcribe the audio file into text
2. Organize the text into sections, paragraphs, lists, etc. We could include summarization here too.



### Audio-to-text
In this part we will build a demo that handles the first step of the meeting transcription tool: converting audio into text.

As we learned, the key ingredient to building a Gradio demo is to have a Python function that executes the logic we are trying to showcase. For the audio-to-text conversion, we will build our function using the awesome `transformers` library and its `pipeline` utility to use a popular audio-to-text model called `distil-whisper/distil-large-v3`.

The result is the following `transcribe` function, which takes as input the audio that we want to convert:




```python
import os
import tempfile

import torch
import gradio as gr
from transformers import pipeline

device = 0 if torch.cuda.is_available() else "cpu"

AUDIO_MODEL_NAME = "distil-whisper/distil-large-v3" # faster and very close in performance to the full-size "openai/whisper-large-v3"
BATCH_SIZE = 8


pipe = pipeline(
    task="automatic-speech-recognition",
    model=AUDIO_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


def transcribe(audio_input):
    """Function to convert audio to text."""
    if audio_input is None:
        raise gr.Error("No audio file submitted!")

    output = pipe(
        audio_input,
        batch_size=BATCH_SIZE,
        generate_kwargs={"task": "transcribe"},
        return_timestamps=True
    )
    return output["text"]
```

Now that we have our Python function, we can demo that by passing it into `gr.Interface`. Notice how in this case the input that the function expects is the audio that we want to convert. Gradio includes a ton useful components, one of which is [Audio](https://www.gradio.app/docs/gradio/audio), exactly what we need for our demo 🎶 😎.



```python
part_1_demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"), # "filepath" passes a str path to a temporary file containing the audio
    outputs=gr.Textbox(show_copy_button=True), # give users the option to copy the results
    title="Transcribe Audio to Text", # give our demo a title :)
)

part_1_demo.launch()
```

Go ahead and try it out 👆! You can upload an `.mp3` file or hit the 🎤 button to record your own voice.

For a sample file with an actual meeting recording, you can check out the [MeetingBank_Audio dataset](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio) which is a dataset of meetings from city councils of 6 major U.S. cities. For my own testing, I tried out a couple of the [Denver meetings](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/blob/main/Denver/mp3/Denver-21.zip).

> [!TIP]
> Also check out `Interface`'s [from_pipeline](https://www.gradio.app/docs/gradio/interface#interface-from_pipeline) constructor which will directly build the `Interface` from a `pipeline`.

### Organize & summarize text
For part 2 of the meeting transcription tool, we need to organize the transcribed text from the previous step.

Once again, to build a Gradio demo we need the Python function with the logic that we care about. For text organization and summarization, we will use an "instruction-tuned" model that is trained to follow a broad range of tasks. There are many options to pick from such as [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) or [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). For our example we are going to use [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).


Just like for part 1, we could leverage the `pipeline` utility within `transformers` to do this, but instead we will take this opportunity to showcase the [Serverless Inference API](https://huggingface.co/docs/api-inference/index), which is an API within the Hugging Face Hub that allows us to use thousands of publicly accessible (or your own privately permissioned) machine learning models ***for free***! Check out the cookbook section of the Serverless Inferfence API [here](https://huggingface.co/learn/cookbook/en/enterprise_hub_serverless_inference_api).



Using the Serverless Inferfence API means that instead of calling a model via a pipeline (like we did for the audio conversion part), we will call it from the `InferenceClient`, which is part of the `huggingface_hub` library ([Hub Python Library](https://huggingface.co/docs/huggingface_hub/en/package_reference/login)). And in turn, to use the `InferenceClient`, we need to log into the 🤗 Hub using `notebook_login()`, which will produce a dialog box asking for your User Access Token to authenticate with the Hub.

You can manage your tokens from your [personal settings page](https://huggingface.co/settings/tokens), and please remember to use [fine-grained](https://huggingface.co/docs/hub/security-tokens) tokens as much as possible for enhanced security.


```python
from huggingface_hub import notebook_login, InferenceClient

# running this will prompt you to enter your Hugging Face credentials
notebook_login()
```

Now that we are logged into the Hub, we can write our text processing function using the Serverless Inference API via `InferenceClient`.

The code for this part will be structured into two functions:

- `build_messages`, to format the message prompt into the LLM;
- `organize_text`, to actually pass the raw meeting text into the LLM for organization (and summarization, depending on the prompt we provide).



```python
# sample meeting transcript from huuuyeah/MeetingBank_Audio
# this is just a copy-paste from the output of part 1 using one of the Denver meetings
sample_transcript = """
 Good evening. Welcome to the Denver City Council meeting of Monday, May 8, 2017. My name is Kelly Velez. I'm your Council Secretary. According to our rules of procedure, when our Council President, Albus Brooks, and Council President Pro Tem, JoLynn Clark, are both absent, the Council Secretary calls the meeting to order. Please rise and join Councilman Herndon in the Pledge of Allegiance. Madam Secretary, roll call. Roll call. Here. Mark. Espinosa. Here. Platt. Delmar. Here. Here. Here. Here. We have five members present. There is not a quorum this evening. Many of the council members are participating in an urban exploration trip in Portland, Oregon, pursuant to Section 3.3.4 of the city charter. Because there is not a quorum of seven council members present, all of tonight's business will move to next week, to Monday, May 15th. Seeing no other business before this body except to wish Councilwoman Keniche a very happy birthday this meeting is adjourned Thank you. A standard model and an energy efficient model likely will be returned to you in energy savings many times during its lifespan. Now, what size do you need? Air conditioners are not a one-size-or-type fits all. Before you buy an air conditioner, you need to consider the size of your home and the cost to operate the unit per hour. Do you want a room air conditioner, which costs less but cools a smaller area, or do you want a central air conditioner, which cools your entire house but costs more? Do your homework. Now, let's discuss evaporative coolers. In low humidity areas, evaporating water into the air provides a natural and energy efficient means of cooling. Evaporative coolers, also called swamp coolers, cool outdoor air by passing it over water saturated pads, causing the water to evaporate into it. Evaporative coolers cost about one half as much to install as central air conditioners and use about one-quarter as much energy. However, they require more frequent maintenance than refrigerated air conditioners, and they're suitable only for areas with low humidity. Watch the maintenance tips at the end of this segment to learn more. And finally, fans. When air moves around in your home, it creates a wind chill effect. A mere two-mile-an-hour breeze will make your home feel four degrees cooler and therefore you can set your thermostat a bit higher. Ceiling fans and portable oscillating fans are cheap to run and they make your house feel cooler. You can also install a whole house fan to draw the hot air out of your home. A whole house fan draws cool outdoor air inside through open windows and exhausts hot room air through the attic to the outside. The result is excellent ventilation, lower indoor temperatures, and improved evaporative cooling. But remember, there are many low-cost, no-cost ways that you can keep your home cool. You should focus on these long before you turn on your AC or even before you purchase an AC. But if you are going to purchase a new cooling system, remember to get one that's energy efficient and the correct size for your home. Wait, wait, don't go away, there's more. After this segment of the presentation is over, you're going to be given the option to view maintenance tips about air conditioners and evaporative coolers. Now all of these tips are brought to you by the people at Xcel Energy. Thanks for watching.
"""
```


```python
from huggingface_hub import InferenceClient

TEXT_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

client = InferenceClient()

def organize_text(meeting_transcript):
    messages = build_messages(meeting_transcript)
    response = client.chat_completion(
        messages, model=TEXT_MODEL_NAME, max_tokens=250, seed=430
    )
    return response.choices[0].message.content


def build_messages(meeting_transcript) -> list:
    system_input = "You are an assitant that organizes meeting minutes."
    user_input = """Take this raw meeting transcript and return an organized version.
    Here is the transcript:
    {meeting_transcript}
    """.format(meeting_transcript=meeting_transcript)

    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input},
    ]
    return messages



```

And now that we have our text organization function `organize_text`, we can build a demo for it as well:


```python
part_2_demo = gr.Interface(
    fn=organize_text,
    inputs=gr.Textbox(value=sample_transcript),
    outputs=gr.Textbox(show_copy_button=True),
    title="Clean Up Transcript Text",
)
part_2_demo.launch()
```

Go ahead and try it out 👆! If you hit "Submit" in the demo above, you will see that the output text is a much clearer and organized version of the transcript, with a title and sections for the different parts of the meeting.

See if you can get a summary by playing around with the `user_input` variable that controls the LLM prompt.

### Putting it all together
At this point we have a function for each of the two steps we want out meeting transcription tool to do: 
1. convert the audio into a text file, and 
2. organize that text file into a nicely-formatted meeting document.

All we have to do next is stitch these two functions together and build a demo for the combined steps. In other words, our complete meeting transcription tool is just a new function (which we'll creatively call `meeting_transcript_tool` 😀) that takes the output of `transcribe` and passes it into `organize_text`:



```python
def meeting_transcript_tool(audio_input):
    meeting_text = transcribe(audio_input)
    organized_text = organize_text(meeting_text)
    return organized_text


full_demo = gr.Interface(
    fn=meeting_transcript_tool,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(show_copy_button=True),
    title="The Complete Meeting Transcription Tool",
)
full_demo.launch()

```

Go ahead and try it out 👆! This is now the full demo of our transcript tool. If you give it an audio file, the output will be the already-organized (and potentially summarized) version of the meeting. Super cool 😎.

## Move your demo into 🤗 Spaces
If you made it this far, now you know the basics of how to create a demo of your machine learning model using Gradio 👏!

Up next we are going to show you how to take your brand new demo to Hugging Face Spaces. On top of the ease of use and powerful features of Gradio, moving your demo to 🤗 Spaces gives you the benefit of permanent hosting, ease of deployment each time you update your app, and the ability to share your work with anyone! Do keep in mind that your Space will go to sleep after a while unless you are using it or making changes to it.


The first step is to head over to [https://huggingface.co/new-space](https://huggingface.co/new-space), select "Gradio" from the templates, and leave the rest of the options as default for now (you can change these later):

<img src="https://github.com/dmaniloff/public-screenshots/blob/main/create-new-space.png?raw=true" width="350" alt="image description">

This will result in a newly created Space that you can populate with your demo code. As an example for you to follow, I created the 🤗 Space `dmaniloff/meeting-transcript-tool`, which you can access [here](https://huggingface.co/spaces/dmaniloff/meeting-transcript-tool).

There are two files we need to edit:

*   `app.py` -- This is where the demo code lives. It should look something like this:
      ```python
      # outline of app.py:

      def meeting_transcript_tool(...):
         ...

      def transcribe(...):
         ...

      def organize_text(...):
         ...

      ```

*   `requirements.txt` -- This is where we tell our Space about the libraries it will need. It should look something like this:
      ```
      # contents of requirements.txt:
      torch
      transformers
      ```


## Gradio comes with batteries included 🔋

Gradio comes with lots of cool functionality right out of the box. We won't be able to cover all of it in this notebook, but here's 3 that we will check out:

- Access as an API
- Sharing via public URL
- Flagging



### Access as an API
One of the benefits of building your web demos with Gradio is that you automatically get an API 🙌! This means that you can access the functionality of your Python function using a standard HTTP client like `curl` or the Python `requests` library.

If you look closely at the demos we created above, you will see at the bottom there is a link that says "Use via API". If you click on it in the Space I created ([dmaniloff/meeting-transcript-tool](https://huggingface.co/spaces/dmaniloff/meeting-transcript-tool/blob/main/app.py)), you will see the following:

<img src="https://github.com/dmaniloff/public-screenshots/blob/main/gradio-as-api.png?raw=true" width="750" alt="image description">

Let's go ahead and copy-paste that code below to use our Space as an API:


```python
!pip install gradio_client
```


```python
from gradio_client import Client, handle_file

client = Client("dmaniloff/meeting-transcript-tool")
result = client.predict(
		audio_input=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
		api_name="/predict"
)
print(result)
```

    Loaded as API: https://dmaniloff-meeting-transcript-tool.hf.space ✔
    Certainly! Below is an organized version of a hypothetical meeting transcript. Since the original transcript you've provided is quite minimal, I'll create a more detailed and structured example featuring a meeting summary.
    
    ---
    
    # Meeting Transcript: Project Alpha Kickoff
    
    **Date:** April 7, 2023
    
    **Location:** Conference Room B, TechCorp Headquarters
    
    
    **Attendees:**
    
    - John Smith (Project Manager)
    
    - Emily Johnson (Lead Developer)
    
    - Michael Brown (Marketing Lead)
    
    - Lisa Green (Design Lead)
    
    
    **Meeting Duration:** 1 hour 30 minutes
    
    
    ## Opening Remarks
    
    **John Smith:**
    
    Good morning everyone, and thank you for joining this kickoff meeting for Project Alpha. Today, we'll discuss our project vision, milestones, and roles. Let's get started.
    
    
    ## Vision and Goals
    
    **Emily Johnson:**
    
    The main goal of Project Alpha is to
    

Wow! What happened there? Let's break it down:

- We installed the `gradio_client`, which is a package that is specifically designed to interact with APIs built with Gradio.
- We instantiated the client by providing the name of the 🤗 Space that we want to query.
- We called the `predict` method of the client and passed in a sample audio file to it.

The Gradio client takes care of making the HTTP POST for us, and it also provides functionality like reading the input audio file that our meeting transcript tool will process (via the function `handle_file`).

Again, using this client is a choice, and you can just as well run a `curl -X POST https://dmaniloff-meeting-transcript-tool.hf.space/call/predict [...]` and pass in all the parameters needed in the request.

> [!TIP]
> The output that we get from the call above is a made-up meeting that was generated by the LLM that we are using for text organization. This is because the sample input file isn't an actual meeting recording. You can tweak the LLM's prompt to handle this case.

### Share via public URL
Another cool feature built into Gradio is that even if you build your demo on your local computer (before you move it into a 🤗 Space) you can still share this with anyone in the world by passing in `share=True` into `launch` like so:

```python
 demo.launch(share=True)
 ```

 You might have noticed that in this Google Colab environment that behaviour is enabled by default, and so the previous demos that we created already had a public URL that you can share 🌎. Go back ⬆ and look at the logs for `Running on public URL:` to find it 🔎!

### Flagging
[Flagging](https://www.gradio.app/guides/using-flagging) is a feature built into Gradio that allows the users of your demo to provide feedback. You might have noticed that the first demo we created had a `Flag` button at the bottom.

Under the default options, if a user clicks that button then the input and output samples are saved into a CSV log file that you can review later. If the demo involves audio (like in our case), these are saved separately in a parallel directory and the paths to these files are saved in the CSV file.

Go back and play with our first demo once more, and then click the `Flag` button. You will see that a new log file is created in the `flagged` directory:


```python
!cat flagged/log.csv
```

    name,intensity,output,flag,username,timestamp
    Diego,4,"Hello, Diego!!!!",,,2024-06-29 22:07:50.242707


In this case I set inputs to `name=diego` and `intensity=29`, which I then flagged. You can see that the log file includes the inputs to the function, the output `"Hello, diego!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"`, and also a timestamp.

While a list of inputs and outputs that your users found problematic is better than nothing, Gradio's flagging feature allows you to do much more. For example, you can provide a `flagging_options` parameter that lets you customize the kind of feedback or errors that you can receive, such as `["Incorrect", "Ambiguous"]`. Note that this requires that `allow_flagging` is set to `"manual"`:


```python
demo_with_custom_flagging = gr.Interface(
    fn=greet,
    inputs=["text", "slider"], # the inputs are a text box and a slider ("text" and "slider" are components in Gradio)
    outputs=["text"],          # the output is a text box
    allow_flagging="manual",
    flagging_options=["Incorrect", "Ambiguous"],
)
demo_with_custom_flagging.launch()
```

Go ahead and try it out 👆! You can see that the flagging buttons now are `Flag as Incorrect` and `Flag as Ambiguous`, and the new log file will reflect those options:


```python
!cat flagged/log.csv
```

    name,intensity,output,flag,username,timestamp
    Diego,4,"Hello, Diego!!!!",,,2024-06-29 22:07:50.242707
    Diego,5,"Hello, Diego!!!!!",Ambiguous,,2024-06-29 22:08:04.281030


## Wrap up & Next Steps
In this notebook we learned how to demo any machine learning model using Gradio.

First, we learned the basics of setting up an interface for a simple Python function; and second, we covered Gradio's true strength: building demos for machine learning models.

For this, we learned how easy it is to leverage models in the 🤗 Hub via the `transformers` library and its `pipeline` function, and how to use multimedia inputs like `gr.Audio`.

Third, we covered how to host your Gradio demo on 🤗 Spaces, which lets you keep your demo running in the cloud and gives you flexibility in terms of the compute requirements for your demo.

Finally, we showcased a few of the super cool batteries included that come with Gradio such as API access, public URLs, and Flagging.

For next steps, check out the `Further Reading` links at the end of each section.

## ⏭️ Further reading
- [Your first demo with gradio](https://www.gradio.app/guides/quickstart#building-your-first-demo)
- [Gradio Components](https://www.gradio.app/docs/gradio/introduction)
- [The transformers library](https://huggingface.co/docs/transformers/en/index)
- [The pipeline function](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
- [Hub Python Library](https://huggingface.co/docs/huggingface_hub/en/package_reference/login)
- [Serverless Inference API](https://huggingface.co/docs/api-inference/index)
- [🤗 Spaces](https://huggingface.co/spaces)
- [Spaces documentation](https://huggingface.co/docs/hub/spaces)

