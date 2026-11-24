
from sentence_transformers import SentenceTransformer
from pdf_extractor import extract_text_from_pdf
from text_processor import clean_text, create_chunks


def generate_embeddings():

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Extracting PDF text...")

    pdf_text = extract_text_from_pdf("data/python_notes.pdf")

    cleaned_text = clean_text(pdf_text)

    chunks = create_chunks(cleaned_text)

    print("Total chunks:", len(chunks))

    embeddings = model.encode(chunks)

    print("Embeddings generated successfully!")

    if len(embeddings) > 0:
        print("Embedding dimensions:", len(embeddings[0]))

    return model, chunks, embeddings


if __name__ == "__main__":

    model, chunks, embeddings = generate_embeddings()
