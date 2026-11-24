
import chromadb
from embedding_generator import generate_embeddings


# Step 1: Generate embeddings
model, chunks, embeddings = generate_embeddings()


# Step 2: Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")


# Step 3: Create a collection
collection = client.get_or_create_collection(
    name="python_documents"
)


# Step 4: Generate unique IDs
ids = [f"chunk_{i}" for i in range(len(chunks))]


# Step 5: Convert embeddings into Python lists
embeddings = embeddings.tolist()


# Step 6: Store data in ChromaDB
if chunks:

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    print("Documents stored successfully!")

else:
    print("No chunks available to store.")


# Step 7: Display database information
print("Total stored chunks:", collection.count())

print("Collection name:", collection.name)
