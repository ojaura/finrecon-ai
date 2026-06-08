# ============================================================
# load_csv.py
# Loads bank ledger and internal records CSVs, identifies
# discrepancies, converts them to documents, and adds them
# to the existing ChromaDB vector store
# ============================================================

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Paths to CSV files
BANK_LEDGER_PATH = "data/bank_ledger.csv"
INTERNAL_RECORDS_PATH = "data/internal_records.csv"

# Path to ChromaDB vector store — must match create_database.py
CHROMA_PATH = "chroma_db"


def load_and_compare_csvs():
    """
    Loads both CSVs, identifies discrepancies,
    and returns them as LangChain Document objects.
    """
    # Load both CSVs into dataframes
    bank_df = pd.read_csv(BANK_LEDGER_PATH)
    internal_df = pd.read_csv(INTERNAL_RECORDS_PATH)

    documents = []

    # Add all bank ledger transactions as documents
    for _, row in bank_df.iterrows():
        content = (
            f"BANK LEDGER - Transaction ID: {row['transaction_id']} | "
            f"Date: {row['date']} | "
            f"Description: {row['description']} | "
            f"Amount: {row['amount']} {row['currency']} | "
            f"Status: {row['status']}"
        )
        documents.append(Document(
            page_content=content,
            metadata={"source": "bank_ledger", "transaction_id": row['transaction_id']}
        ))

    # Add all internal records as documents
    for _, row in internal_df.iterrows():
        content = (
            f"INTERNAL RECORDS - Transaction ID: {row['transaction_id']} | "
            f"Date: {row['date']} | "
            f"Description: {row['description']} | "
            f"Amount: {row['amount']} {row['currency']} | "
            f"Status: {row['status']}"
        )
        documents.append(Document(
            page_content=content,
            metadata={"source": "internal_records", "transaction_id": row['transaction_id']}
        ))

    # Identify discrepancies between the two sources
    discrepancies = identify_discrepancies(bank_df, internal_df)
    documents.extend(discrepancies)

    print(f"Loaded {len(bank_df)} bank ledger transactions.")
    print(f"Loaded {len(internal_df)} internal records.")
    print(f"Found {len(discrepancies)} discrepancies.")
    return documents


def identify_discrepancies(bank_df, internal_df):
    """
    Compares bank ledger and internal records to find
    missing transactions, amount mismatches, and status mismatches.
    Returns discrepancies as LangChain Document objects.
    """
    discrepancy_docs = []

    # Merge on transaction_id to compare
    merged = pd.merge(bank_df, internal_df, on='transaction_id',
                      suffixes=('_bank', '_internal'), how='outer')

    for _, row in merged.iterrows():
        txn_id = row['transaction_id']

        # Missing from internal records
        if pd.isna(row.get('amount_internal')):
            content = (
                f"DISCREPANCY - Transaction {txn_id} exists in bank ledger "
                f"but is MISSING from internal records. "
                f"Bank amount: {row['amount_bank']} CAD."
            )
            discrepancy_docs.append(Document(
                page_content=content,
                metadata={"source": "discrepancy", "transaction_id": txn_id, "type": "missing_from_internal"}
            ))

        # Missing from bank ledger
        elif pd.isna(row.get('amount_bank')):
            content = (
                f"DISCREPANCY - Transaction {txn_id} exists in internal records "
                f"but is MISSING from bank ledger. "
                f"Internal amount: {row['amount_internal']} CAD."
            )
            discrepancy_docs.append(Document(
                page_content=content,
                metadata={"source": "discrepancy", "transaction_id": txn_id, "type": "missing_from_bank"}
            ))

        else:
            # Amount mismatch
            if row['amount_bank'] != row['amount_internal']:
                content = (
                    f"DISCREPANCY - Transaction {txn_id} has an AMOUNT MISMATCH. "
                    f"Bank ledger: {row['amount_bank']} CAD. "
                    f"Internal records: {row['amount_internal']} CAD. "
                    f"Difference: {abs(row['amount_bank'] - row['amount_internal']):.2f} CAD."
                )
                discrepancy_docs.append(Document(
                    page_content=content,
                    metadata={"source": "discrepancy", "transaction_id": txn_id, "type": "amount_mismatch"}
                ))

            # Status mismatch
            if row['status_bank'] != row['status_internal']:
                content = (
                    f"DISCREPANCY - Transaction {txn_id} has a STATUS MISMATCH. "
                    f"Bank ledger status: {row['status_bank']}. "
                    f"Internal records status: {row['status_internal']}."
                )
                discrepancy_docs.append(Document(
                    page_content=content,
                    metadata={"source": "discrepancy", "transaction_id": txn_id, "type": "status_mismatch"}
                ))

    return discrepancy_docs


def add_to_chroma(documents):
    """
    Adds CSV-derived documents to the existing ChromaDB vector store
    alongside the PDF documents.
    """
    # Initialize Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # Load existing ChromaDB and add new documents
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    db.add_documents(documents)
    print(f"Added {len(documents)} CSV documents to ChromaDB.")


def main():
    documents = load_and_compare_csvs()
    add_to_chroma(documents)
    print("CSV data successfully loaded into vector store.")


if __name__ == "__main__":
    main()