"""
Google GenAI File Search Example
Python version of the file search implementation


This tutorial will walk you through the complete lifecycle of using File Search with python

1. Create a File Search Store
2. Find a Store by Display Name
3. Upload Multiple Files Concurrently
4. Advanced Upload: Chunking & Metadata
5. Run a Standard Generation Query (RAG)
6. Find a Specific Document
7. Delete a Document
8. Update a Document
9. Cleanup: Delete the File Search Store
"""

from google import genai
from google.genai import types
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise Exception("GOOGLE_API_KEY not found in .env file")

# Initialize client with API key
client = genai.Client(api_key=api_key)

# Configuration
file_store_name = 'my-example-store'
docs_dir = "docs"

# 1. Create a File Search Store
# A File Search Store is a persistent container for your document chunks and embeddings.
# It's distinct from raw file storage and can hold gigabytes of data.
print("Step 1: Creating File Search Store...")
file_search_store = client.file_search_stores.create(
    config={'display_name': file_store_name}
)
print(f"Created store: {file_search_store.name}\n")

# 2. Retrieve Store by Display Name (useful if creation is separated from usage)
# Often the creation of a store and the use are in a different application session. 
# Since the API assigns a unique ID (fileSearchStores/xyz...), you need to look it up by the human-readable displayName.
print("Step 2: Retrieving store by display name...")
file_store = None
for store in client.file_search_stores.list(config={'page_size': 10}):
    if store.display_name == file_store_name:
        file_store = store
        print(f"Found store: {store.name}")
        break

if not file_store:
    raise Exception(f"Store with display name '{file_store_name}' not found.")

# 3. Upload Files Concurrently using threading and wait for all to finish
# Speed matters. When ingesting a folder of documents, don't process them sequentially. 
# The API supports concurrent operations, so we can use Promise.all to upload and process multiple files at once.
print("\nStep 3: Uploading files concurrently...")
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_file(file_path):
    """Upload a single file and wait for completion"""
    print(f"Uploading {file_path}...")
    operation = client.file_search_stores.upload_to_file_search_store(
        file=file_path,
        file_search_store_name=file_store.name,
        config={
            'display_name': Path(file_path).name,
        }
    )
    
    # Wait until operation is complete
    while not operation.done:
        time.sleep(1)
        operation = client.operations.get(operation=operation)
    
    print(f"Completed: {Path(file_path).name}")
    return operation

# Get all files from docs directory
files = [str(Path(docs_dir) / f) for f in os.listdir(docs_dir) if Path(docs_dir, f).is_file()]

# Upload files concurrently
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(upload_file, file_path) for file_path in files]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error uploading file: {e}")

print("\nAll files uploaded successfully!")

# 4. Run Generation with File Search
# By default, Gemini handles chunking intelligently. 
# However, for specific use cases you might want tighter control over how your data is split.
print("\nStep 4: Running generation with File Search...")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is Gemini and what is the File API?",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_store.name]
                )
            )
        ]
    )
)

print(f"Model response: {response.text}\n")

# 5. Update a File: First list all documents, get the name of the document to update, 
# delete the document and upload the updated file
# We don't need to manually retrieve chunks. 
# We just tell the Gemini model to use the fileSearch tool and point it to our store name.
# Gemini understands it needs more information, searches the store, and grounds its response automatically.
print("Step 5: Updating a document...")
updated_content = "Gemini is a sign Language developed in 2025 by Chaoran Zhou."
doc1_display_name = 'doc1.txt'
doc1_path = Path(docs_dir) / doc1_display_name

# Write updated content to file
with open(doc1_path, 'w') as f:
    f.write(updated_content)

# Find and delete the document
# Currently, the standard flow to update a document in File Search is to delete the old version and upload a new one
print(f"Finding document: {doc1_display_name}")
file_deleted = False
for document in client.file_search_stores.documents.list(parent=file_store.name):
    if document.display_name == doc1_display_name:
        client.file_search_stores.documents.delete(
            name=document.name,
            config={'force': True}
        )
        print("File deleted.")
        file_deleted = True
        break

if not file_deleted:
    raise Exception(f"Document with display name '{doc1_display_name}' not found.")

# Upload the updated file
# File Search documents are immutable once indexed. 
# To "update" a document, you must find it, delete it, and upload the new version.
# In this step, we will automate this entire loop to update doc1.txt with new information.
print("Uploading updated file...")
upload_op = client.file_search_stores.upload_to_file_search_store(
    file=str(doc1_path),
    file_search_store_name=file_store.name,
    config={
        'display_name': doc1_display_name,
    }
)

while not upload_op.done:
    time.sleep(1)
    upload_op = client.operations.get(operation=upload_op)

print("File uploaded.\n")

# 6. Query with Updated Content
# You'll often need to manage individual documents within your store.
# You can find a specific document by its display name.
print("Step 6: Querying with updated content...")
response_updated = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What details can you give me about Gemini now?",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[file_store.name]
                )
            )
        ]
    )
)

print(f"Updated model response: {response_updated.text}\n")


# Cleanup
# You are currently limited to 10 File Search Stores per project, so it's important to clean up resources when you are finished with development.
print("Cleaning up...")
client.file_search_stores.delete(name=file_store.name, config={'force': True})
print("Store deleted.")