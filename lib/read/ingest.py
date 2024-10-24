import PyPDF2


class Ingest:

    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self.extract_text()

    def extract_text(self):
        with open(self.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def get_text(self):
        return self.text
