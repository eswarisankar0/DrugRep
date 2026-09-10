import sys
import os
from dotenv import load_dotenv
print("STEP 1: python is running", flush=True)

try:
    from Bio import Entrez
    print("STEP 2: biopython imported ok", flush=True)
except Exception as e:
    print(f"STEP 2 FAILED: {e}", flush=True)
    sys.exit(1)

load_dotenv()

ENTREZ_EMAIL = os.getenv("NCBI_EMAIL")
ENTREZ_API_KEY = os.getenv("NCBI_API_KEY") 

print("STEP 3: about to call NCBI...", flush=True)

try:
    handle = Entrez.esearch(db="pubmed", term="Amantadine AND Parkinson disease", retmax=5)
    print("STEP 4: got response from NCBI", flush=True)
    record = Entrez.read(handle)
    handle.close()
    print("STEP 5: parsed response:", record.get("IdList", []), flush=True)
except Exception as e:
    print(f"STEP 3/4/5 FAILED with error: {type(e).__name__}: {e}", flush=True)

print("DONE", flush=True)