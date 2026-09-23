from pypdf import PdfReader


def extract_text_from_pdf(file):
    """
    Extract text from every page of a PDF.

    Each extracted page keeps:
    - document name
    - page number
    - page text
    """

    documents = []

    reader = PdfReader(file)

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        # Skip empty pages safely
        if text and text.strip():

            documents.append(
                {
                    "text": text.strip(),
                    "metadata": {
                        "source": file.name,
                        "page": page_number
                    }
                }
            )

    return documents