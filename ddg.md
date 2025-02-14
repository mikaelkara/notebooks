# DuckDuckGo Search

This guide shows over how to use the DuckDuckGo search component.

## Usage


```python
%pip install -qU duckduckgo-search langchain-community
```


```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search.invoke("Obama's first name?")
```




    "The White House, official residence of the president of the United States, in July 2008. The president of the United States is the head of state and head of government of the United States, [1] indirectly elected to a four-year term via the Electoral College. [2] The officeholder leads the executive branch of the federal government and is the commander-in-chief of the United States Armed ... Here is a list of the presidents and vice presidents of the United States along with their parties and dates in office. ... Chester A Arthur: Twenty-First President of the United States. 10 Interesting Facts About James Buchanan. Martin Van Buren - Eighth President of the United States. Quotes From Harry S. Truman. 2 of 2. Barack Obama: timeline Key events in the life of Barack Obama. Barack Obama (born August 4, 1961, Honolulu, Hawaii, U.S.) is the 44th president of the United States (2009-17) and the first African American to hold the office. Before winning the presidency, Obama represented Illinois in the U.S. Senate (2005-08). Table of Contents As the head of the government of the United States, the president is arguably the most powerful government official in the world. The president is elected to a four-year term via an electoral college system. Since the Twenty-second Amendment was adopted in 1951, the American presidency has been limited to a maximum of two terms.. Click on a president below to learn more about ... Most common names of U.S. presidents 1789-2021. Published by. Aaron O'Neill, Aug 9, 2024. The most common first name for a U.S. president is James, followed by John and then William. Six U.S ..."



To get more additional information (e.g. link, source) use `DuckDuckGoSearchResults()`


```python
from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()

search.invoke("Obama")
```




    "snippet: He maintains a close friendship with Mr. Obama. He first weighed in on presidential politics to criticize President Reagan's re-election campaign, and has since supported Mr. Obama, Hillary ..., title: Bruce Springsteen to Appear With Harris and Obama at Atlanta and ..., link: https://www.nytimes.com/2024/10/22/us/politics/springsteen-harris-atlanta-obama-philly.html, snippet: Learn about the life and achievements of Barack Obama, the 44th president of the United States and the first African American to hold the office. Explore his early years, education, political career, books, awards, and more., title: Barack Obama | Biography, Parents, Education, Presidency, Books ..., link: https://www.britannica.com/biography/Barack-Obama, snippet: Obama's personal charisma, stirring oratory, and his campaign promise to bring change to the established political system resonated with many Democrats, especially young and minority voters. On January 3, 2008, Obama won a surprise victory in the first major nominating contest, the Iowa caucus, over Sen. Hillary Clinton, who was the overwhelming favorite to win the nomination., title: Barack Obama - 44th President, Political Career, Legacy | Britannica, link: https://www.britannica.com/biography/Barack-Obama/Politics-and-ascent-to-the-presidency, snippet: Former President Barack Obama endorsed Kamala Harris and Tim Walz for president and vice president in his keynote address at the 2024 Democratic National Convention. He criticized Donald Trump's rhetoric and policies, and praised Harris' empathy and decency., title: Watch: Barack Obama's full speech at the 2024 DNC, link: https://www.cbsnews.com/news/watch-barack-obamas-full-speech-2024-dnc-transcript/"



By default the results are returned as a comma-separated string of key-value pairs from the original search results. You can also choose to return the search results as a list by setting `output_format="list"` or as a JSON string by setting `output_format="json"`.


```python
search = DuckDuckGoSearchResults(output_format="list")

search.invoke("Obama")
```




    [{'snippet': 'Obama was headed to neighboring Michigan later Tuesday, among the several stops the former president is making in battleground states to encourage early voting. Harris has been spending a lot of time in the " blue wall " states of Wisconsin, Michigan and Pennsylvania in the final weeks of the campaign, including stops in Michigan and ...',
      'title': 'Obama and Walz host rally in Wisconsin as early voting kicks off | AP News',
      'link': 'https://apnews.com/article/wisconsin-voting-trump-harris-obama-walz-aeeff20ab17a54172263ee4778bed3dc'},
     {'snippet': "Obama has directed plenty of criticisms at Trump over the years, so some might perceive this as little more than the latest installment in a larger pattern. But let's not be too quick to rush ...",
      'title': 'Why Obama slamming Trump on his response to Covid matters - MSNBC',
      'link': 'https://www.msnbc.com/rachel-maddow-show/maddowblog/obama-slamming-trump-response-covid-matters-rcna176624'},
     {'snippet': 'Learn about the life and achievements of Barack Obama, the 44th president of the United States and the first African American to hold the office. Explore his early years, education, political career, books, awards, and more.',
      'title': 'Barack Obama | Biography, Parents, Education, Presidency, Books ...',
      'link': 'https://www.britannica.com/biography/Barack-Obama'},
     {'snippet': "He maintains a close friendship with Mr. Obama. He first weighed in on presidential politics to criticize President Reagan's re-election campaign, and has since supported Mr. Obama, Hillary ...",
      'title': 'Bruce Springsteen to Appear With Harris and Obama at Atlanta and ...',
      'link': 'https://www.nytimes.com/2024/10/22/us/politics/springsteen-harris-atlanta-obama-philly.html'}]



You can also just search for news articles. Use the keyword `backend="news"`


```python
search = DuckDuckGoSearchResults(backend="news")

search.invoke("Obama")
```




    "snippet: Springsteen, a longtime Democratic activist, will be joined by former President Barack Obama at both shows, with Harris scheduled to attend the Atlanta concert. Springsteen, a New Jersey native, has maintained a deep connection with Philadelphia throughout his career., title: Bruce Springsteen to hold battleground concerts with Kamala Harris, Barack Obama, link: https://www.statesman.com/story/news/politics/elections/2024/10/22/springsteen-obama-2024-concerts-harris/75791934007/, date: 2024-10-22T20:45:00+00:00, source: Austin American-Statesman, snippet: Obama roasts Trump's bible: 'He's Mr. Tough Guy on China except when it comes to making a few bucks', title: Obama roasts Trump's bible: 'He's Mr. Tough Guy on China except when it comes to making a few bucks', link: https://www.msnbc.com/deadline-white-house/watch/obama-roasts-trump-he-s-mr-tough-guy-on-china-except-when-it-comes-to-making-a-few-bucks-222399557750, date: 2024-10-22T20:28:00+00:00, source: MSNBC, snippet: Democratic nominee and Vice President Kamala Harris is sending a potentially powerful closer to Detroit today to shore up her standing in Michigan: Former President Barack Obama.             The campaign has been notably mum on exactly when and where Obama is speaking but as he hits the campaign trail on behalf of his fellow Democrat,, title: Watch live tonight: Former President Obama to rally for Kamala Harris in Detroit, link: https://www.msn.com/en-us/news/politics/watch-live-tonight-former-president-obama-to-rally-for-kamala-harris-in-detroit/ar-AA1sJpiB, date: 2024-10-22T18:32:26+00:00, source: Detroit Free Press on MSN.com, snippet: During a campaign rally in Wisconsin, former President Obama called into question former President Trump's mental fitness and joked about him ending a town hall to listen to music with attendees., title: 'Our playlist would be better': Obama jokes about Trump listening to music at town hall, link: https://www.msnbc.com/msnbc/watch/obama-jokes-about-trump-ending-town-hall-to-listen-to-music-222399045983, date: 2024-10-22T20:31:00+00:00, source: MSNBC"



You can also directly pass a custom `DuckDuckGoSearchAPIWrapper` to `DuckDuckGoSearchResults` to provide more control over the search results.


```python
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)

search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

search.invoke("Obama")
```




    'snippet: 22.10.2024, 18.21 Uhr. 1. Macht der Worte. Der beste Redner der US-Politik: Obama in Tucson. Foto: Mamta Popat / Arizona Daily Star / AP / dpa. Politik ist Sprache. Das wussten die Jakobiner im ..., title: News des Tages: Barack Obama und Donald Trump, Elon Musk, Herbert Kickl, link: https://www.spiegel.de/politik/deutschland/news-des-tages-barack-obama-und-donald-trump-elon-musk-herbert-kickl-a-c1a76de5-f9aa-4038-a1ad-f426b85267f8, snippet: Oct. 22, 2024, 12:38 p.m. ET. The rocker Bruce Springsteen will perform at a rally on Thursday in Atlanta, appearing alongside Vice President Kamala Harris and former President Barack Obama, as ..., title: Bruce Springsteen to Appear With Harris and Obama at Atlanta and ..., link: https://www.nytimes.com/2024/10/22/us/politics/springsteen-harris-atlanta-obama-philly.html, snippet: In-person early voting has kicked off across battleground Wisconsin, with former President Barack Obama and Democratic vice presidential nominee Tim Walz hosting a rally in liberal Madison, title: Watch live: Barack Obama and VP nominee Tim Walz hold rally in ..., link: https://news.sky.com/video/watch-live-barack-obama-and-and-vp-nominee-tim-walz-hold-rally-in-wisconsin-13239026, snippet: Barack Obama will speak to a campaign rally for Kamala Harris in Madison, Wisconsin this afternoon. The former US president was expected to be joined by Tim Walz, the Democratic vice presidential ..., title: Obama-Walz rally live: Former president campaigns for Harris in Madison, link: https://www.telegraph.co.uk/us/politics/2024/10/22/obama-walz-rally-campaigns-kamala-harris-madison-wisconsin/'



## Related

- [How to use a chat model to call tools](https://python.langchain.com/docs/how_to/tool_calling/)
