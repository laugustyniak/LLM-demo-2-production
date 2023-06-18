import chainlit as cl
from langchain import PromptTemplate, LLMChain, OpenAI

from langchain.document_loaders import ArxivLoader

template = """Please create a social media post on LinkedIn with emojis for the academic paper described below.

Paper description: {paper_description}
"""

@cl.on_chat_start
async def start():
    # Send the first message without the elements
    content = "🙋Hi there! I will assist you 💁 in creating a social media post for an academic paper 📝 \n\n Please send me a an arxiv link to an academic paper."

    # await cl.Message(
    #     content=content,
    # ).send()

    # elements = [
    #     cl.LocalImage(path="images/cat.jpeg", name="image1", display="inline"),
    #     cl.Text(content="Here is a side text document", name="text1", display="side"),
    #     cl.Text(content="Here is a page text document", name="text2", display="page"),
    # ]

    # Send the second message with the elements
    await cl.Message(content=content).send()

@cl.on_message
async def main(message: str):
    # Your custom logic goes here...

    # Send a response back to the user
    await cl.Message(
        content=f"You sent: {message}",
    ).send()

@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["paper_description"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, streaming=True), verbose=True)

    return llm_chain
