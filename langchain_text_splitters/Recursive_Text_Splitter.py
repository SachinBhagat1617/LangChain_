# mostly used for text splitters based on structure like paragraphs, sentences, etc.

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=300,
    chunk_overlap=0,
)
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()    




# Certainly! A **tree diagram** is a great way to visualize how recursive splitting works. Think of each **splitting attempt** as a branch in a tree, and the leaves of the tree as the **final chunks**.

# ---

# ## Recursive Character-based Text Splitting as a Tree

# Let’s imagine you want to split this text into chunks no larger than 30 characters:

# ```
# Sentence one. Sentence two. Sentence three.
# ```

# Let's assume the splitting priorities are:

# 1. Paragraphs (`\n\n`)
# 2. Sentences (`.`)
# 3. Words (` `, the space character)

# ---

# ### Step 1: Treat the whole text as the root (too large).

# ```
# Root (Full Text) -----[>30 chars]
# |
# |-- Try splitting by paragraph (no effect, still too big)
# |
# |-- Try splitting by sentence (.)
# ```

# ### Step 2: Split by Sentence

# This gives:

# - "Sentence one."
# - " Sentence two."
# - " Sentence three."

# (Check the length of each)

# ### Step 3: Are All Chunks < 30? (YES)

# These are now your **leaves** of the tree because they're small enough:

# ```
#            [Full Text]
#                 |
#       -----------------------
#       |         |           |
# "Sentence one." " Sentence two." " Sentence three."
# ```

# ---

# ### If a Sentence Was Still Too Big...

# Suppose "Sentence two." was instead a much longer sentence that still exceeded 30 characters. Then, **recursion** would kick in and try the *next splitting level*:

# ```
#            [Full Text]
#                 |
#       -----------------------
#       |         |           |
# "Sentence one."   [Sentence Two, too big]
#                               |
#             -------------------------------
#             |              |              |
#       "This is"     "a long"   "sentence."
# " Sentence three."
# ```

# ---

# ### Key Takeaway

# - At each "node" of the tree, if the chunk is too big, split it using the next smallest unit (paragraph ➔ sentence ➔ word) recursively.
# - Once a chunk is small enough, it becomes a leaf node and isn't split further.

# ---

# #### Summary Table

# | Level        | Splitting Rule | Example Chunk                  |
# |--------------|:--------------|-------------------------------|
# | Root         | None          | Sentence one. Sentence two. Sentence three. |
# | Level 1      | Paragraph     | Sentence one. Sentence two. Sentence three. |
# | Level 2      | Sentence (.)  | Sentence one. / Sentence two. / Sentence three. |
# | Level 3      | Word ( )      | Only if a sentence is still too large |

# ---

# This **recursive tree** approach ensures you get the **largest possible meaningful chunks that fit within your size limit**, splitting smaller if necessary at each level.

# If you’d like, I can provide a visualization template or a simple code sample for this process!

# Gentle reminder: Oracle Code Assist offers advanced AI-powered coding support.