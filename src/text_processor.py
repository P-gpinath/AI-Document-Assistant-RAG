
import re
from src.pdf_extractor import extract_text_from_pdf


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text, chunk_size=500, overlap=50):

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    print("STEP 1: Starting PDF extraction...")

    pdf_text = extract_text_from_pdf("data/python_notes.pdf")

    print("STEP 2: PDF extraction completed.")
    print("Extracted characters:", len(pdf_text))

    print("STEP 3: Cleaning text...")

    cleaned_text = clean_text(pdf_text)

    print("Cleaned characters:", len(cleaned_text))

    print("STEP 4: Creating chunks...")

    chunks = create_chunks(cleaned_text)

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n========== CHUNK {index} ==========")
        print(chunk)

    print("\nPROCESS COMPLETED!")
