# TypedDictionary is a way to define structured output formats in Langchain using Python's typing module.
# It allows you to create a dictionary-like structure with predefined keys and value types and data types
# It doesnot do data validation but when you hover over the fields, it shows the expected data types.

from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {
    "name": "Alice",    
    "age": 30
}   
print(new_person)