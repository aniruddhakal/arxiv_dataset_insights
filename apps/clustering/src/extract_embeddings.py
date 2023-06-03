from pathlib import Path

import pandas as pd
import pymongo
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from arxiv_dataset import ArxivDataset

db_name = "arxiv-db"
collection_name = "arxiv-dataset-collection"

db = pymongo.MongoClient(host="localhost", port=21000).get_database(db_name)
collection = db[collection_name]

save_path = Path('/mnt/NVMe2/arxiv_embeddings/')

import asyncio
loop = asyncio.new_event_loop()


async def extract_embeddings(model, data_loader: DataLoader, device: torch.device, embedding_model: str):
    model = model.to(device)

    save_dir = save_path / embedding_model

    pooler_dir = save_dir / 'pooler'
    hidden_state_dir = save_dir / 'last_hidden_state'

    pooler_dir.mkdir(exist_ok=True, parents=True)
    hidden_state_dir.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            target_name = f"part-{i}.pt"

            task1 = asyncio.create_task(save_tensor(target_name=pooler_dir / target_name, tensor=outputs.pooler_output.detach().cpu()))
            # task2 = asyncio.create_task(save_tensor(target_name=hidden_state_dir / target_name, tensor=outputs.last_hidden_state.detach().cpu()))

            await asyncio.gather(task1)


async def save_tensor(target_name, tensor: torch.Tensor):
    with open(target_name, 'wb') as f:
        torch.save(tensor.detach().cpu(), f)

def load_data():
    data_df = pd.DataFrame(collection.find({}, {"title", "abstract", "categories"}))
    return data_df


def get_data_loader(data_df: pd.DataFrame, tokenizer, batch_size=128):
    dataset = ArxivDataset(data=data_df, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def init_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return model, tokenizer


if __name__ == '__main__':
    all_data_df = load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Set up the BERT model and tokenizer
    model_name = "bert-base-uncased"  # You can replace this with any other BERT model
    model, tokenizer = init_model_and_tokenizer(model_name=model_name)

    dataloader = get_data_loader(data_df=all_data_df[:], tokenizer=tokenizer, batch_size=384)

    task = asyncio.run(extract_embeddings(model=model, data_loader=dataloader, device=device, embedding_model=model_name))
