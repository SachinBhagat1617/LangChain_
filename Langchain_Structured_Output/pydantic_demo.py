from pydantic import BaseModel, Field, EmailStr
from typing import Optional

# Pydantic is a data validation and settings management library for Python.
# It uses Python type annotations to define data models and validate data against those models.
# features default values, Optional fields, Coercion of data types(e.g., str to int), Field metadata for additional information about fields like description, constraints, regex   etc.
# it also return pydantic obj -> convert to json/dict easily

class Student(BaseModel):
    name: str = Field(..., description="Full name of the student")  # ... means required field
    age: int = Field(..., ge=0, le=120, description="Age of the student in years")
    email: EmailStr = Field(..., description="Email address of the student")
    gpa: Optional[float] = Field(None, ge=0.0, le=10.0, description="Grade Point Average of the student")
new_student = Student(
    name="John Doe",        
    age="21",  # Coerced from str to int ( it handles type coercion)
    email="john@gmail.com",
    gpa=8.5     
)
print(new_student)
print(new_student.model_dump_json())  # Convert to JSON format
print(new_student.model_dump())       # Convert to dictionary format