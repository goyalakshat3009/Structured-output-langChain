
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Models
model1 = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

model2 = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.7
)

parser = StrOutputParser()


# Pydantic Model
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)


# Prompt 1: Sentiment Classification
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback
into positive or negative.

Feedback: {feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser2.get_format_instructions()
    }
)

classifier_chain = prompt1 | model1 | parser2


# Prompt 2: Positive Feedback
prompt2 = PromptTemplate(
    template="""
Write an appropriate response to this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)


# Prompt 3: Negative Feedback
prompt3 = PromptTemplate(
    template="""
Write an appropriate response to this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)


# Conditional Branch
branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        prompt2 | model1 | parser
    ),
    (
        lambda x: x.sentiment == "negative",
        prompt3 | model2 | parser
    ),
    RunnableLambda(lambda x: "Could not find sentiment")
)


# Complete Chain
chain = classifier_chain | branch_chain


# Test
result = chain.invoke({
    "feedback": "This is a beautiful phone. I really love its camera!"
})

print(result)

# Graph
chain.get_graph().print_ascii()
