
import streamlit as st
import chromadb
import ollama
import hashlib

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from src.text_processor import clean_text, create_chunks


# Step 1: Configure the application
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered"
)


# Step 2: Load the embedding model
@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


model = load_embedding_model()


# Step 3: Initialize session state
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Step 4: Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")


# Step 5: Create or access collection
collection = client.get_or_create_collection(
    name="uploaded_documents"
)


# Step 6: Display application title
st.title("🤖 AI Document Assistant")

st.write(
    "Upload a PDF document and ask multiple questions about its content."
)


# Step 7: Upload a PDF
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# Step 8: Process the uploaded PDF
if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_signature = hashlib.md5(file_bytes).hexdigest()

    if file_signature != st.session_state.file_signature:

        st.session_state.document_ready = False

        st.session_state.chat_history = []

        with st.spinner("Processing your PDF..."):

            reader = PdfReader(uploaded_file)

            extracted_text = ""

            for page_number, page in enumerate(reader.pages, start=1):

                text = page.extract_text()

                if text:
                    extracted_text += text + "\n"


            # Step 9: Clean the extracted text
            if extracted_text.strip():

                cleaned_text = clean_text(extracted_text)


                # Step 10: Create text chunks
                chunks = create_chunks(cleaned_text)


                # Step 11: Generate embeddings
                embeddings = model.encode(chunks)


                # Step 12: Generate unique IDs
                ids = [
                    f"{file_signature}_{i}"
                    for i in range(len(chunks))
                ]


                # Step 13: Store chunks in ChromaDB
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=[
                        {"source": file_signature}
                        for _ in chunks
                    ]
                )


                # Step 14: Save document information
                st.session_state.processed_file = uploaded_file.name

                st.session_state.file_signature = file_signature

                st.session_state.document_ready = True


                # Step 15: Display processing results
                st.success("PDF processed successfully!")

                st.write("File name:", uploaded_file.name)

                st.write("Total pages:", len(reader.pages))

                st.write("Extracted characters:", len(extracted_text))

                st.write("Total chunks:", len(chunks))

                st.write("Stored chunks:", len(chunks))


            else:

                st.warning("No readable text found in this PDF.")


    # Step 16: Ask multiple questions
    if st.session_state.document_ready:

        st.divider()

        st.subheader("💬 Ask Questions About Your PDF")

        question = st.text_input(
            "Enter your question:",
            key="question_input"
        )


        if st.button("Get Answer"):

            if question.strip():

                with st.spinner("Searching your document..."):

                    # Step 17: Convert question into embedding
                    question_embedding = model.encode(question)


                    # Step 18: Retrieve relevant chunks
                    results = collection.query(
                        query_embeddings=[
                            question_embedding.tolist()
                        ],
                        n_results=3,
                        where={
                            "source": file_signature
                        }
                    )

                    documents = results["documents"][0]


                if documents:

                    context = "\n\n".join(documents)


                    # Step 19: Create the AI prompt
                    prompt = f"""
You are a helpful AI document assistant.

Answer using only the information in the document context.

If the answer is not available, clearly say that
the information was not found in the uploaded document.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Give a simple and clear answer.
"""


                    # Step 20: Generate AI answer
                    with st.spinner("Generating AI answer..."):

                        response = ollama.chat(
                            model="llama3.2:1b",
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        )


                    answer = response["message"]["content"]


                    # Step 21: Save question and answer
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )


                else:

                    st.warning("No relevant information found.")


            else:

                st.warning("Please enter a question.")


        # Step 22: Display chat history
        if st.session_state.chat_history:

            st.divider()

            st.subheader("📚 Chat History")

            for index, chat in enumerate(
                st.session_state.chat_history,
                start=1
            ):

                st.markdown(f"**Question {index}:**")

                st.write(chat["question"])

                st.markdown("**🤖 AI Answer:**")

                st.write(chat["answer"])

                st.divider()


            # Step 23: Clear chat history
            if st.button("Clear Chat History"):

                st.session_state.chat_history = []

                st.rerun()
