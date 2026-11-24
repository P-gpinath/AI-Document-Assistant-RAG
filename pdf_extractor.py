
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):

    try:
        # Open the PDF
        reader = PdfReader(pdf_path)

        extracted_text = ""

        # Read every page
        for page_number, page in enumerate(reader.pages, start=1):

            print(f"Extracting page {page_number}...")

            text = page.extract_text()

            if text:
                extracted_text += text + "\n"

            else:
                print(f"No text found on page {page_number}")

        return extracted_text

    except Exception as error:
        print("Error reading PDF:", error)
        return ""


# Run PDF extraction
if __name__ == "__main__":

    pdf_path = "data/python_notes.pdf"

    pdf_text = extract_text_from_pdf(pdf_path)

    print("\n========== EXTRACTED TEXT ==========\n")

    print(pdf_text)

    print("\n========== EXTRACTION COMPLETED ==========")
