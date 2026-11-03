from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):
    name: str
    email: EmailStr
    age: int = Field(gt=0, lt=100)
    cgpa: float = Field(gt=0, lt=10, default=5)

student = Student(
    name="Akshat",
    email="akshat@gmail.com",
    age=20,
    cgpa=8.5
)

print(student)