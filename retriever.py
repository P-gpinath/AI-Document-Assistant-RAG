
import chromadb


# Step 1: Connect to the existing ChromaDB database
client = chromadb.PersistentClient(path="./chroma_db")


# Step 2: Access the existing collection
collection = client.get_collection(
    name="python_documents"
)


# Step 3: Check stored documents
print("Collection name:", collection.name)

print("Total stored chunks:", collection.count())



from sentence_transformers import SentenceTransformer


# Step 4: Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 5: Create a user question
question = input("\nEnter your question: ")


# Step 6: Convert the question into an embedding
question_embedding = model.encode(question)


# Step 7: Search for the 3 most relevant chunks
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=3
)



# Step 8: Display retrieved chunks and distances
print("\n========== SEARCH RESULTS ==========")

for index, document in enumerate(results["documents"][0], start=1):

    distance = results["distances"][0][index - 1]

    print(f"\nResult {index}:")
    print("Distance:", distance)
    print("Content:", document)

