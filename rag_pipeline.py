
import chromadb
import ollama
from sentence_transformers import SentenceTransformer


# Step 1: Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Step 2: Access the existing collection
collection = client.get_collection(
    name="python_documents"
)

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question):

    # Step 1: Convert the question into an embedding
    question_embedding = model.encode(question)

    # Step 2: Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=3
    )

    # Step 3: Get the retrieved document chunks
    documents = results["documents"][0]

    # Step 4: Combine the chunks into one context
    context = "\n\n".join(documents)

    return context
def generate_answer(question, context):

    # Step 1: Create the prompt
    
    prompt = f"""
    You are a helpful Python tutor.

    Use the following document content to answer the question.

    DOCUMENT CONTENT:
    {context}

    QUESTION:
    {question}

    Give a simple and clear answer.
    Do not say the information is missing if the document contains
    relevant information.
    """


    # Step 2: Send the prompt to Ollama
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Step 3: Return the AI-generated answer
    return response["message"]["content"]


if __name__ == "__main__":

    # Step 1: Get the user's question
    question = input("\nAsk a question about your PDF: ")

    # Step 2: Retrieve relevant context
    context = retrieve_context(question)

    # Display retrieved context
    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)

    # Step 3: Generate an AI answer
    answer = generate_answer(question, context)

    # Step 4: Display the answer
    print("\n========== RAG ANSWER ==========")
    print(answer)
