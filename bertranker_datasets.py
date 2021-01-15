from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class PairPrefDataset(Dataset):
    def __init__(self, poem_pairs: list, targets: list):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.poem_pairs = poem_pairs
        self.targets = targets
        self.max_len = 80

    def __len__(self):
        return len(self.poem_pairs)

    def __getitem__(self, i):
        # first item in the pair
        encoding1 = self.tokenizer.encode_plus(
            self.poem_pairs[i][0],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # second item in the pair
        encoding2 = self.tokenizer.encode_plus(
            self.poem_pairs[i][1],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text1': self.poem_pairs[i][0],
            'text2': self.poem_pairs[i][1],
            'input_ids1': encoding1['input_ids'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'targets': torch.tensor(self.targets[i], dtype=torch.float)
        }

class SingleDataset(Dataset):
    def __init__(self, poems: list, max_len=80):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # BertTokenizer.from_pretrained('bert-base-cased')
        self.poems = poems
        self.max_len = max_len

    def get_index_of_poem(self, poem):
        return self.poems.index(poem)

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, i):
        # first item in the pair
        encoding = self.tokenizer.encode_plus(
            self.poems[i],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': self.poems[i],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
