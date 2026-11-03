from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    format="json"
)

# 1. Define the schema first
class Review(BaseModel):
    key_themes: list[str]
    summary: str
    sentiment: Literal["pos", "neg", "neutral"]
    pros: Optional[list[str]] = None
    cons: Optional[list[str]] = None
    name: Optional[str] = None

# 2. Create parser after defining Review
parser = PydanticOutputParser(pydantic_object=Review)

# 3. Create prompt
prompt = PromptTemplate(
    template="""
Extract information from this product review.

Return a JSON object with exactly these fields:
- key_themes: list of strings
- summary: string
- sentiment: pos, neg, or neutral
- pros: list of strings
- cons: list of strings
- name: string or null

Do not explain anything. Return only JSON.

Review:
{review}
""",
    input_variables=["review"]
)

chain = prompt | model | parser

# 5. Invoke chain
result = chain.invoke({
    "review": """
    The Samsung Galaxy S24 Ultra has an excellent camera
    and battery, but it is expensive.
    Review by Akshat Goyal.
    """
})

print(result)
print(result.name)
print(result.sentiment)