import chainlit as cl
from langchain import PromptTemplate, LLMChain, OpenAI

from langchain.document_loaders import ArxivLoader

template = """Please create a social media post on LinkedIn with emojis for the academic paper described below.

Paper description: {paper_description}
"""

@cl.on_chat_start
async def start():
    # Send the first message without the elements
    content = "Here is image1, a nice image of a cat! As well as text1 and text2!"

    await cl.Message(
        content=content,
    ).send()

    elements = [
        # cl.LocalImage(path="images/cat.jpeg", name="image1", display="inline"),
        cl.Text(content="Here is a side text document", name="text1", display="side"),
        cl.Text(content="Here is a page text document", name="text2", display="page"),
    ]

    # Send the second message with the elements
    await cl.Message(
        content=content,
        elements=elements,
    ).send()

@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["paper_description"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, streaming=True), verbose=True)

    return llm_chain
