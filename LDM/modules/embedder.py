#%%
import warnings
warnings.filterwarnings("ignore", message=".*beta.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*weight.*")

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

text_model = BertModel.from_pretrained("bert-base-uncased").to('cuda')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BertWithCustomOutput(nn.Module):
    def __init__(self, bert_model, output_dim=512, max_len=77):
        super(BertWithCustomOutput, self).__init__()
        assert isinstance(output_dim, int), "output_dim must be an integer"
        self.bert_model = bert_model
        self.linear = nn.Linear(bert_model.config.hidden_size, output_dim)
        self.positional_embedding = nn.Embedding(max_len, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        custom_output = self.linear(last_hidden_state)

        return custom_output

class TextEmbedder:
    def __init__(self, model_name="bert-base-uncased", output_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = BertModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.text_model = BertWithCustomOutput(self.bert_model, output_dim).to(self.device)
    
    def generate_text_embeddings(self, texts):
        self.text_model.eval()

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, tuple):
            texts = list(texts)
        if not texts:
            raise ValueError("The `texts` input is empty or None.")

        tokenized_texts = self.tokenizer(texts, 
                                         padding="max_length",
                                         max_length=77,
                                         return_tensors="pt",
                                         truncation=True,
                                         return_overflowing_tokens=False
                                ).to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_model(input_ids=tokenized_texts['input_ids'], attention_mask=tokenized_texts['attention_mask'])
        return text_embeddings
            