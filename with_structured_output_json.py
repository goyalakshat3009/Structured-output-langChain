from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    timeout=60,
    max_retries=2
)

# JSON schema
json_schema = {
    "title": "Review",
    "description": "Structured representation of a review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {
            "type": "string"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neutral"]
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"}
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"}
        },
        "name": {
            "type": ["string", "null"]
        }
    },
    "required": [
        "key_themes",
        "summary",
        "sentiment",
        "pros",
        "cons",
        "name"
    ],
    "additionalProperties": False
}

structured_model = model.with_structured_output(
    json_schema,
    method="json_schema"
)

result = structured_model.invoke(
    "Review by Akshat Goyal. The Samsung phone has an excellent "
    "camera and battery, but it is expensive and heavy."
)

print(result)