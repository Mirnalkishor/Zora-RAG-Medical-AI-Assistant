import requests
from config import API_URL

def ask_question(question: str):
    return requests.post(f"{API_URL}/ask/", data={"question": question})

def upload_pdfs_api(files):
    file_data = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_pdfs/", files=file_data)