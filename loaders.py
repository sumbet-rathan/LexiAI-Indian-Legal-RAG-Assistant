
"""Step 1: Load PDF and TXT legal documents."""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from .config import DATA_DIR


def load_documents(path=DATA_DIR) -> list[Document]:
    """
    Load PDF and TXT documents from a file or directory.

    Preserves:
    - Source file path
    - Page number metadata
    - Document content
    """

    path = Path(path).resolve()

    # 1. Validate path
    if not path.exists():
        raise FileNotFoundError(
            f"Document path does not exist: {path}"
        )

    docs = []

    # 2. Load a single file
    if path.is_file():

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))

        elif path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(path),
                encoding="utf-8"
            )

        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}"
            )

        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = path.name

    # 3. Load documents from a directory
    elif path.is_dir():

        pdf_loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )

        txt_loader = DirectoryLoader(
            str(path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )

        docs.extend(pdf_loader.load())
        docs.extend(txt_loader.load())

        # 4. Preserve relative source paths
        for doc in docs:
            original_source = Path(
                doc.metadata["source"]
            ).resolve()

            doc.metadata["source"] = str(
                original_source.relative_to(path)
            )

    else:
        raise ValueError("Provide a valid file or directory.")

    # 5. Validate extracted documents
    if not docs:
        raise ValueError(
            f"No PDF or TXT documents found in: {path}"
        )

    print(f"Successfully loaded {len(docs)} pages/documents.")

    return docs


if __name__ == "__main__":

    documents = load_documents()

    print("\nFirst document metadata:")
    print(documents[0].metadata)

    print("\nExtracted text preview:")
    print(documents[0].page_content[:500])
