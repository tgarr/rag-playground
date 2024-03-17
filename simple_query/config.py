
import os

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    set_global_service_context,
)
from llama_index.llms import HuggingFaceInferenceAPI

DATA_PATH = "./data"
STORAGE_PATH = "./storage"

def setup_hf_inference():
    llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta")

    service_context = ServiceContext.from_defaults(embed_model="local",llm=llm)
    set_global_service_context(service_context)

def get_storage_path(name):
    return f"{STORAGE_PATH}/{name}"

def get_data_path(name):
    return f"{DATA_PATH}/{name}"

def get_index(name):
    storage_path = get_storage_path(name)

    if not os.path.exists(storage_path):
        # load the documents and create the index
        documents = SimpleDirectoryReader(get_data_path(name)).load_data()
        index = VectorStoreIndex.from_documents(documents)

        # store it for later
        index.storage_context.persist(persist_dir=storage_path)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)

    return index

