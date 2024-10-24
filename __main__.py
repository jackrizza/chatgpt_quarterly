from lib.read.ingest import Ingest
from lib.read.digest import Digest
from lib.read.ai import AI
from lib.main import train, main

import argparse
def cli():
    parser = argparse.ArgumentParser(description="AI Model Training and PDF Processing CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the AI model")
    train_parser.set_defaults(func=train)

    # Run command
    build_parser = subparsers.add_parser("Run", help="Run on pdf file test.pdf")
    build_parser.set_defaults(func=main)  # Assuming build is the same as train for now

    # Save PDF as text command
    save_pdf_parser = subparsers.add_parser("save_pdf_as_text", help="Save PDF content as text")
    save_pdf_parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    save_pdf_parser.set_defaults(func=save_pdf_as_text)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func()

def save_pdf_as_text():
    ingest = Ingest("./test.pdf")
    ingest_text = ingest.get_text()
    with open("output.txt", "w") as text_file:
        text_file.write(ingest_text)
if __name__ == "__main__":
    cli()