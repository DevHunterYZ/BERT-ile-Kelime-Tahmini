# PyTorch ve Transformers kütüphanelerini yükle
!pip install torch torchvision
!pip install transformers

import torch
from transformers import BertTokenizer, BertForMaskedLM

# BERT tokenizer'ını yükle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Örnek bir cümle oluştur
sentence = "I enjoy [MASK] in the park."

# Cümleyi tokenlara ayır
tokens = tokenizer.tokenize(sentence)

# [MASK] tokeninin indeksini bul
mask_index = tokens.index('[MASK]')

# Tokenları tensöre dönüştür
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])

# BERT modelini yükle
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# Modeli kullanarak [MASK] için en olası kelimeleri tahmin et
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0][0, mask_index].topk(5) # En olası 5 tahmin
    predicted_ids = predictions.indices.tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)

# Tahmin edilen kelimeleri ekrana yazdır
print("Tahmin edilen kelimeler:")
for token in predicted_tokens:
    print(token)
