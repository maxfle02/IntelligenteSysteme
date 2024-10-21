# Load initial data
import os
from langchain_community.document_loaders import PyPDFLoader


def load_documents_from_directory(directory_path=None):
    """
    Load documents from PDFs in a directory.

    Args:
        directory_path (str, optional): The path to the directory containing the PDF files. If not provided, the default path is "data".

    Returns:
        list: A list of loaded documents.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        PermissionError: If permission is denied when accessing the directory.

    """
    documents = []
    if directory_path is None:
        directory_path = "data"

    try:
        file_list = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Error: The directory {directory_path} does not exist.")
        return documents
    except PermissionError:
        print(f"Error: Permission denied accessing the directory {directory_path}.")
        return documents

    print(f"Debug: Loading data from PDFs in {file_list}")

    for file in file_list:

        if (
            file.endswith(".pdf")
            and file not in open("data/loaded_documents.txt").read()
        ):
            # print(f"Debug: Loading file {file}")
            pdf_path = os.path.join(directory_path, file)
            try:
                with open("data/loaded_documents.txt", "a") as f:
                    f.write(f"{file}\n")
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load_and_split())

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

    return documents
