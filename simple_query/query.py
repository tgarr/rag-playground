
import time
import sys
import os
from config import *

module_name = sys.argv[1]

setup_hf_inference()
index = get_index(module_name)

query_engine = index.as_query_engine()

while True:
    question = input("Question: ")
    if question == "quit" or question == "Quit":
        break

    start = time.time()
    response = query_engine.query(question)
    end = time.time()

    response_str = str(response).strip()
    print("Answer:",response_str)
    print(f"\n---------- {(end-start)*1000:.2f} ms ----------\n")

