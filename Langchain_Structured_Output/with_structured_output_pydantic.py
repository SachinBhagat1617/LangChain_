from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Optional,Literal,Annotated

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Annotated is used to add metadata ("Prompt to model") to the fields in the TypedDict
#Literal is used to restrict the field to specific string values
class Review(BaseModel):
    key_themes: list[str]= Field(...,description="Write down all the key themes mentioned in the review in a list format.")
    summary: str= Field(...,description="A brief summary of the review")
    sentiment: Literal["pos","neg","neutral"]= Field(...,description="Overall sentiment of the review, choose from 'pos', 'neg', 'neutral'")
    pros: Optional[list[str]]= Field(None,description="List of pros mentioned in the review in a list format. If none, return an empty list.")
    cons: Optional[list[str]]= Field(None,description="List of cons mentioned in the review in a list format. If none, return an empty list.")
    name: str= Field(...,description="Name of the product being reviewed, extract from the review if mentioned. If not mentioned, return 'Unknown'.")

structured_model= model.with_structured_output(Review)
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
""")

print(result)

