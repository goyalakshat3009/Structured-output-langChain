from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person: Person={'name':'Akshat','age':'19'}
print(new_person)