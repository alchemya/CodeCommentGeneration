import torch
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from CodeNL import CodeNL
from config import sys_cfg

SRC_LANGUAGE = 'code'
TGT_LANGUAGE = 'nl'
token_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
print(token_transform[SRC_LANGUAGE])


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([sys_cfg.bos_index]),
                      torch.tensor(token_ids),
                      torch.tensor([sys_cfg.eos_index])))


def yield_tokens(data_iter, language):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


def collate_fn(batch, text_transform):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=sys_cfg.pad_index)
    tgt_batch = pad_sequence(tgt_batch, padding_value=sys_cfg.pad_index)
    return src_batch, tgt_batch


def get_vocab_transform():
    vocab_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        train_iter = CodeNL(split='train', src_path='./data/train.token.code', trg_path='./data/train.token.nl')
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=sys_cfg.special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(sys_cfg.unk_index)
    return vocab_transform


def get_text_transform(vocab_transform):
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],
                                                   vocab_transform[ln],
                                                   tensor_transform)
    return text_transform


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=sys_cfg.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=sys_cfg.device).type(torch.bool)

    src_padding_mask = (src == sys_cfg.pad_index).transpose(0, 1)
    tgt_padding_mask = (tgt == sys_cfg.pad_index).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(sys_cfg.device)
    src_mask = src_mask.to(sys_cfg.device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(sys_cfg.device)
    for i in range(max_len - 1):
        memory = memory.to(sys_cfg.device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(sys_cfg.device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == sys_cfg.eos_index:
            break
    return ys


def bleu(src, tgt):
    smooth = SmoothingFunction()
    score = sentence_bleu([src], tgt,weights=(1, 0, 0, 0))
    return score
