# %%
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from einops import rearrange
from torch.nn import functional as F
from tqdm import tqdm
import random

# %%
p = print

# %%
def generate_bracket_string(maxsize):
    BDICT = {
        '(': ')',
        '[': ']',
        '{': '}',
    }
    bracket_string = ''
    stack = []
    for _ in range(int(maxsize * 0.9)):
        if len(stack) == 0 or t.rand(1) < 0.5: # put new bracket
            bracket = random.choice(list(BDICT.keys()))
            stack.append(BDICT[bracket])
        else:
            bracket = stack.pop()
        bracket_string += bracket
    while len(stack) > 0:
        bracket_string += stack.pop()
    if len(bracket_string) > maxsize:
        return generate_bracket_string(maxsize)
    else:
        return bracket_string


# %%
# mean with these params is 517.3
BLEN = 500
ITERS = 1000
p(f'Average len: {sum([len(generate_bracket_string(BLEN)) for _ in range(ITERS)])/ITERS}')


# %%
generate_bracket_string(10)

# %%
def isValid(s: str) -> bool:
        stack = []
        for char in s:
            try:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    if stack.pop() != '(':
                        return False
                elif char == '[':
                    stack.append(char)
                elif char == ']':
                    if stack.pop() != '[':
                        return False
                elif char == '{':
                    stack.append(char)
                elif char == '}':
                    if stack.pop() != '{':
                        return False
                else:
                    raise Exception('Invalid character')
            except:
                return False
        return len(stack) == 0

# %%
def make_invalid_bracket_string(size):
    bracket_string = generate_bracket_string(size)
    # change random brackets to invalid
    corrupt_size = random.randint(1, int(size * 0.1))
    for _ in range(corrupt_size):
        bracket_string = bracket_string.replace(random.choice(list('()[]{}')), random.choice(list('()[]{}')), 1)

    if isValid(bracket_string):
        return make_invalid_bracket_string(size)
    else:
        return bracket_string

assert [isValid(make_invalid_bracket_string(10)) for i in range(1000)].count(False) == 1000

# %%
# create dataset with bracket strings
# and labels for each bracket

# tokenizer for brackets
class BracketTokenizer:
    def __init__(self, vocab, maxlen):
        self.maxlen = maxlen
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.pad_token_id = self.vocab['[PAD]']
        self.cls_token_id = self.vocab['[CLS]']
        self.sep_token_id = self.vocab['[SEP]']
        self.mask_token_id = self.vocab['[MASK]']
        self.unk_token_id = self.vocab['[UNK]']
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.vocab_inv[self.pad_token_id] = '[PAD]'
        self.vocab_inv[self.cls_token_id] = '[CLS]'
        self.vocab_inv[self.sep_token_id] = '[SEP]'
        self.vocab_inv[self.mask_token_id] = '[MASK]'

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.vocab_inv[id] for id in ids]

    def __call__(self, text):
        return self.encode(text)

    def encode(self, text):
        assert len(text) + 2 <= self.maxlen
        ids = self.convert_tokens_to_ids(text)
        ids = [self.cls_token_id] + ids + [self.sep_token_id] # add cls and sep tokens
        ids = ids + [self.pad_token_id] * (self.maxlen - len(ids)) # pad to max len
        return ids

    def decode(self, ids):
        if type(ids) == t.Tensor:
            ids = ids.tolist()
        ids = [id for id in ids if id != self.pad_token_id]
        # remove cls and sep tokens if present
        if ids[0] == self.cls_token_id:
            ids = ids[1:]
        if ids[-1] == self.sep_token_id:
            ids = ids[:-1]
        tokens = ''.join(self.convert_ids_to_tokens(ids))
        return tokens

vocab = {'(': 0, ')': 1, '[': 2, ']': 3, '{': 4, '}': 5, '[PAD]': 6, '[CLS]': 7, '[SEP]': 8, '[MASK]': 9, '[UNK]': 10}
tokenizer = BracketTokenizer(vocab, maxlen=512)


# %%
print(tokenizer.maxlen)

# %%
bracketss = [generate_bracket_string(10) for _ in range(10)]
for brackets in bracketss:
    print(brackets)
    assert len(tokenizer(brackets)) == 512
    assert tokenizer.decode(tokenizer(brackets)) == brackets

# %%
class BracketDataset(TensorDataset):
    def __init__(self, size, tokenizer: BracketTokenizer, validfrac=0.7):
        self.tokenizer = tokenizer
        self.size = size
        self.validfrac = validfrac
        self.rng = random.Random(42)
        self.train = self._make_dataset()
        super().__init__(*self.train)
    
    def _make_dataset(self):
        validsize = int(self.size * self.validfrac)
        invalidsize = self.size - validsize
        valid_bracket_strings = [self.tokenizer(generate_bracket_string(BLEN)) for _ in range(validsize)]
        valid_bracket_labels = [1] * validsize
        invalid_bracket_strings = [self.tokenizer(make_invalid_bracket_string(BLEN)) for _ in range(invalidsize)]
        invalid_bracket_labels = [0] * invalidsize
        
        bracket_strings = valid_bracket_strings + invalid_bracket_strings
        bracket_labels = valid_bracket_labels + invalid_bracket_labels
        # shuffle
        zipped = list(zip(bracket_strings, bracket_labels))
        self.rng.shuffle(zipped)
        bracket_strings, bracket_labels = zip(*zipped)
        # to tensor
        bracket_strings = t.tensor(bracket_strings, dtype=t.long)
        bracket_labels = t.tensor(bracket_labels, dtype=t.long)
        return bracket_strings, bracket_labels

# %%
trainset = BracketDataset(size=4096, tokenizer=tokenizer)
testset = BracketDataset(size=512, tokenizer=tokenizer)

# %%
for x,y in trainset:
    assert isValid(tokenizer.decode(x)) == y.item()

# %%
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# %%
for (x, y) in trainloader:
    print(x.shape)
    break


