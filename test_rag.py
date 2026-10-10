import pytest
import os
import shutil
from rag import load_documents, create_chunks, initialize_vector_store, process_query, clear_vector_store

# Setup temporary data for testing
TEST_DATA_DIR = "test_data"

def setup_module():
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)
    with open(os.path.join(TEST_DATA_DIR, "test.txt"), "w") as f:
        f.write("Admissions are open. B.Tech eligibility is 60%")

def teardown_module():
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    clear_vector_store()

def test_load_documents():
    docs = load_documents(TEST_DATA_DIR)
    assert len(docs) == 1
    assert "Admissions are open" in docs[0].page_content

def test_create_chunks():
    docs = load_documents(TEST_DATA_DIR)
    chunks = create_chunks(docs)
    assert len(chunks) > 0
    assert "token" not in chunks[0].metadata # Check basic metadata

def test_rag_pipeline():
    docs = load_documents(TEST_DATA_DIR)
    chunks = create_chunks(docs)
    vs = initialize_vector_store(chunks)

    # Test retrieval
    res = process_query("What is B.Tech eligibility?", chunks, vs)
    assert "60%" in res["answer"]

    # Test fallback
    res = process_query("What is the fee?", chunks, vs)
    assert "I don't have that information" in res["answer"]