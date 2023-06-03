from torch.utils.data import Dataset
import pandas as pd


class ArxivDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        abstract = self.data.loc[index, 'abstract']
        encoded_input = self.tokenizer(abstract, padding=True, truncation=True, return_tensors='pt',
                                       pad_to_multiple_of=512)

        # Return a dictionary with the desired fields
        return {
            'index': index,
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze()
        }
