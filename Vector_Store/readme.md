
# 📘 Vector Stores in LangChain (Detailed Notes & README)

## 📌 Overview

This repository contains **detailed conceptual and practical notes on Vector Stores**, a **core building block of Retrieval-Augmented Generation (RAG)** systems using LangChain.

The notes explain:

* Why **vector stores are needed**
* How **semantic similarity** works using embeddings
* Difference between **Vector Stores vs Vector Databases**
* How LangChain provides a **common interface** for multiple vector stores
* Practical usage with **Chroma Vector Store**
* CRUD operations and semantic search using embeddings

---

## 🧠 Why Do We Need Vector Stores?

### ❌ Problem with Keyword-Based Search

Traditional systems rely on **keyword matching**, which fails in many real-world cases.

#### Example:

* Movie A: *My Name Is Khan*
* Movie B: *Kabhi Alvida Na Kehna*

They share:

* Same actor (Shah Rukh Khan)
* Same director (Karan Johar)
* Similar release time
* Drama genre

👉 Keyword matching says **they are similar**,
but **story-wise they are completely different**.

---

### ✅ Solution: Semantic Similarity (Meaning-Based Search)

Instead of matching keywords, we compare the **meaning of text** (plots, descriptions).

This is done using **Embeddings**.

---

## 🧩 What Are Embeddings?

Embeddings are **numerical vector representations** of text that capture **semantic meaning**.

* Text → Neural Network → Vector (e.g., 512 dimensions)
* Similar meaning → Vectors closer in space
* Different meaning → Vectors far apart

Once converted to vectors, we can compute:

* **Cosine Similarity**
* **Angular Distance**

---

## 🎯 Real-World Example: Movie Recommendation System

### Steps:

1. Collect movie plots (2000–3000 words each)
2. Generate embeddings for every plot
3. Store embeddings in a **Vector Store**
4. Compare vectors to find similar movies
5. Recommend movies with **highest semantic similarity**

---

## 🚧 Challenges Without Vector Stores

| Challenge            | Problem                                        |
| -------------------- | ---------------------------------------------- |
| Embedding Generation | Millions of texts                              |
| Storage              | Vectors don’t fit relational DBs               |
| Search               | Linear search over millions of vectors is slow |

👉 **Vector Stores solve all three problems efficiently**

---

## 📦 What Is a Vector Store?

> A **Vector Store** is a system designed to **store, retrieve, and search numerical vectors efficiently**.

---

## 🔑 Core Features of Vector Stores

### 1️⃣ Vector Storage

* Store embeddings
* Store associated **metadata** (IDs, labels, tags)
* Supports:

  * In-memory storage (fast, non-persistent)
  * Disk-based storage (persistent, scalable)

---

### 2️⃣ Similarity Search

* Finds vectors closest to a query vector
* Uses cosine similarity or distance metrics

---

### 3️⃣ Indexing (Very Important)

Indexing enables **fast similarity search**.

Instead of:

```
Compare query with 1,000,000 vectors ❌
```

We do:

```
Cluster → Narrow search → Compare fewer vectors ✅
```

Common techniques:

* Clustering-based indexing
* Approximate Nearest Neighbors (ANN)

---

### 4️⃣ CRUD Operations

* Add vectors
* Update vectors
* Delete vectors
* Retrieve vectors

---

## 📌 Common Use Cases

* Recommendation Systems
* Semantic Search
* Retrieval-Augmented Generation (RAG)
* Chatbots with memory
* Image / Audio / Multimedia search

---

## ⚖️ Vector Store vs Vector Database

### Vector Store

* Lightweight
* Focuses on:

  * Storage
  * Similarity search
* Ideal for:

  * Prototyping
  * Small-scale applications

**Example:** FAISS

---

### Vector Database

* Full-fledged database system
* Adds:

  * Persistence
  * Distributed architecture
  * Authentication & authorization
  * Backup & restore
  * High scalability

**Examples:**

* Chroma
* Pinecone
* Weaviate

👉 **Every vector database is a vector store, but not vice versa**

---

## 🔗 Vector Stores in LangChain

LangChain provides **built-in wrappers** for popular vector stores.

### Supported Vector Stores:

* Chroma
* FAISS
* Pinecone
* Weaviate
* Qdrant

### Key Advantage

All vector stores share the **same interface**:

```python
from_documents()
add_documents()
similarity_search()
similarity_search_with_score()
```

👉 You can **swap vector stores without changing business logic**

---

## 🧪 Chroma Vector Store (Hands-on)

Chroma is:

* Lightweight
* Open-source
* Ideal for local development
* Small to medium-scale production

---

### Chroma Data Hierarchy

```
Tenant
 └── Database
      └── Collection
           └── Documents
                ├── Embedding Vector
                └── Metadata
```

---

## 🛠️ Operations Demonstrated

### ✅ Creating Documents

Each document contains:

* `page_content` → actual text
* `metadata` → additional info (e.g., team name)

---

### ✅ Creating Vector Store

* Choose embedding model (e.g., OpenAI embeddings)
* Choose persistence directory
* Create collection

---

### ✅ Add Documents

Automatically:

* Generates embeddings
* Assigns unique document IDs

---

### ✅ View Stored Data

Retrieve:

* IDs
* Embeddings
* Documents
* Metadata

---

### ✅ Semantic Search

```python
similarity_search(query="Who is a bowler?", k=2)
```

Returns most semantically relevant documents.

---

### ✅ Search with Score

Lower distance score = higher similarity

---

### ✅ Metadata Filtering

Example:

```python
filter={"team": "Chennai Super Kings"}
```

---

### ✅ Update Document

Update using document ID

---

### ✅ Delete Document

Delete using document ID

---

## 📚 Homework (Highly Recommended)

Try implementing the **same code** using:

* FAISS
* Pinecone

👉 You’ll notice **almost zero code changes**, thanks to LangChain’s common interface.

---

## 🎯 Final Takeaway

* Vector Stores are **fundamental for RAG systems**
* Keyword search is **not enough**
* Semantic similarity powered by embeddings is the future
* LangChain simplifies vector store integration
* Chroma is perfect for learning and prototyping

---
Perfect 👍
Now I’ll **explain everything using a single, clean end-to-end code example**, and I’ll explain **each step in simple words** so you understand *why* we are doing it — not just *what*.

We’ll use:

* LangChain
* Chroma

This example covers:

* Embeddings
* Vector Store creation
* Adding documents
* Similarity search
* Metadata filtering
* Update & delete

---

# 🧠 Vector Stores Explained Using Code (LangChain + Chroma)

---

## 1️⃣ Install Required Libraries

```bash
pip install langchain chromadb openai
```

👉 These give us:

* LangChain → framework
* Chroma → vector store
* OpenAI → embeddings model

---

## 2️⃣ Import Required Classes

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
```

### Why?

* `OpenAIEmbeddings` → converts text → vectors
* `Chroma` → stores vectors + searches them
* `Document` → standard LangChain text format

---

## 3️⃣ Create Documents (Text + Metadata)

```python
docs = [
    Document(
        page_content="Virat Kohli is one of the best batsmen in IPL history.",
        metadata={"team": "RCB"}
    ),
    Document(
        page_content="MS Dhoni is a legendary captain and wicketkeeper.",
        metadata={"team": "CSK"}
    ),
    Document(
        page_content="Jasprit Bumrah is a fast bowler known for yorkers.",
        metadata={"team": "MI"}
    ),
    Document(
        page_content="Ravindra Jadeja is an all-rounder who bats and bowls.",
        metadata={"team": "CSK"}
    )
]
```

### 🧠 Concept

Each `Document` has:

* `page_content` → text used for embeddings
* `metadata` → extra info for filtering

---

## 4️⃣ Create Embedding Model

```python
embedding = OpenAIEmbeddings()
```

### 🧠 What happens here?

* Text → Neural Network → Vector (e.g., 1536 numbers)
* Meaning is captured numerically

---

## 5️⃣ Create Chroma Vector Store

```python
vectorstore = Chroma(
    collection_name="players",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)
```

### 🧠 What is happening?

* `collection_name` → like a table
* `persist_directory` → data stored on disk
* Embeddings auto-generated when documents are added

---

## 6️⃣ Add Documents to Vector Store

```python
ids = vectorstore.add_documents(docs)
print(ids)
```

### 🧠 Internally:

1. Text → embeddings
2. Embeddings stored in Chroma
3. Each document gets a **unique ID**

---

## 7️⃣ View Stored Data

```python
data = vectorstore.get(
    include=["documents", "metadatas"]
)
print(data)
```

### 🧠 You can see:

* Stored text
* Metadata
* Document IDs

---

## 8️⃣ Semantic Similarity Search (Core Feature 🔥)

```python
results = vectorstore.similarity_search(
    query="Who is a bowler?",
    k=1
)

print(results[0].page_content)
```

### 🧠 What happens internally?

1. Query → embedding
2. Query vector compared with all stored vectors
3. Cosine similarity used
4. Most similar vector returned

👉 Output:

```
Jasprit Bumrah is a fast bowler known for yorkers.
```

---

## 9️⃣ Similarity Search With Score

```python
results = vectorstore.similarity_search_with_score(
    query="Who is a bowler?",
    k=2
)

for doc, score in results:
    print(doc.page_content, " | Score:", score)
```

### 🧠 Important:

* **Lower score = more similar**
* Score represents vector distance

---

## 🔟 Metadata-Based Filtering

```python
results = vectorstore.similarity_search(
    query="",
    filter={"team": "CSK"}
)

for doc in results:
    print(doc.page_content)
```

### 🧠 Why this is powerful?

* Combine **semantic search + structured filtering**
* Very useful in RAG apps

👉 Output:

```
MS Dhoni is a legendary captain...
Ravindra Jadeja is an all-rounder...
```

---

## 1️⃣1️⃣ Update an Existing Document

```python
doc_id = ids[0]

updated_doc = Document(
    page_content="Virat Kohli is a former RCB captain known for aggressive batting.",
    metadata={"team": "RCB"}
)

vectorstore.update_documents(
    ids=[doc_id],
    documents=[updated_doc]
)
```

### 🧠 Internally:

* Old vector deleted
* New embedding generated
* Same ID reused

---

## 1️⃣2️⃣ Delete a Document

```python
vectorstore.delete(ids=[doc_id])
```

### 🧠 Result:

* Vector removed
* Metadata removed
* Not returned in future searches

---

## 🔄 How This Fits Into RAG

```
User Question
   ↓
Convert to Embedding
   ↓
Vector Store Similarity Search
   ↓
Relevant Context
   ↓
LLM Generates Answer
```

👉 Vector Store is the **brain memory** of RAG.

---

## 🎯 Key Takeaways (Interview Gold)

* Embeddings = meaning → numbers
* Vector stores enable **semantic search**
* Chroma stores vectors efficiently
* LangChain provides **common interface**
* Same code works for FAISS / Pinecone / Weaviate



