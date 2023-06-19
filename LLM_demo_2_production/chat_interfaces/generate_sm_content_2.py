import json

import chainlit as cl
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain import OpenAI, LLMChain, PromptTemplate


template = """
Perform the following actions for a research paper text: 
1 - Summarize the following text delimited by triple backticks with 1 sentence.
2 - Generate a title for this summary with emojis.
3 - Prepare a tweet based on summary.
4 - Prepare a longer linkedin post based on paper abstract. 

Return a Markdown table with the following strcuture: summary, title, tweet, linkedin_post.

Paper description: :
```{paper_description}```
"""


@cl.on_chat_start
async def start():
    content = "🙋Hi there! I will assist you 💁 in creating a social media post for an academic paper 📝 \n\n Paste paper information."
    await cl.Message(content=content).send()


@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["paper_description"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, streaming=False, max_tokens=512), verbose=True)

    return llm_chain
