import torch
import torchtext
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


train_iter = torchtext.datasets.AG_NEWS(root='data', split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

print(vocab(['here', 'is', 'an', 'example']))


def text_pipeline(x):
    return vocab(tokenizer(x))


def label_pipeline(x):
    return int(x) - 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device)


dataloader = DataLoader(train_iter, batch_size=1,
                        shuffle=False, collate_fn=collate_batch)

for e in dataloader:
    print(e)
    break
