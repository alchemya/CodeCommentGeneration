import torch

from config import sys_cfg
from data_utils import greedy_decode, get_vocab_transform, get_text_transform
from model import Seq2SeqTransformer


def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform['code'](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=sys_cfg.bos_index).flatten()
    return " ".join(vocab_transform['nl'].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>",
                                                                                                 "").replace(
        "<eos>", "")


vocab_transform = get_vocab_transform()
text_transform = get_text_transform(vocab_transform)
SRC_VOCAB_SIZE = len(vocab_transform['code'])
TGT_VOCAB_SIZE = len(vocab_transform['nl'])
transformer = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
model = torch.load('model.pth', map_location='cpu')
res=translate(transformer, 'public BigFractionFormat ( ) { }')
print(res)
