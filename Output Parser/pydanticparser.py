
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Define model
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    format="json"
)

# Define schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt
template = PromptTemplate(
    template="""
Generate details of ONE fictional Sri Lankan person.

Return only JSON with these exact keys:
name, age, city.

Rules:
- name must be a string.
- age must be an integer greater than 18.
- city must be a string.
- Do not generate Python code.
- Do not explain anything.

{format_instruction}

""",
    input_variables=["place"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# Create chain
chain = template | model | parser

# Execute
final_result = chain.invoke({"place": "Sri Lankan"})

print(final_result)
print(final_result.name)
print(final_result.age)
print(final_result.city)
