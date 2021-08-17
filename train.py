from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from CodeNL import CodeNL
from config import sys_cfg
from data_utils import create_mask, bleu
from data_utils import get_vocab_transform, get_text_transform
from model import Seq2SeqTransformer
from show_table import plot_table

torch.manual_seed(2021)


def evaluate(model):
    model.eval()
    val_iter = CodeNL(split='valid', src_path='./data/valid.token.code', trg_path='./data/valid.token.nl')
    val_dataloader = DataLoader(val_iter, batch_size=sys_cfg.batch_size, collate_fn=collate_fn)
    sum_bleu = 0.
    num_sentences = 0

    for src, tgt in val_dataloader:
        src = src.to(sys_cfg.device)
        tgt = tgt.to(sys_cfg.device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        predict = torch.max(logits, dim=-1, keepdim=True).indices.squeeze(-1)
        tgt_out = tgt[1:, :]
        for p, t in zip(predict, tgt_out):
            num_sentences += 1
            pred = vocab_transform['nl'].lookup_tokens(list(p.cpu().detach().numpy()))
            gold = vocab_transform['nl'].lookup_tokens(list(t.cpu().detach().numpy()))
            bleu_score = bleu(pred, gold)
            sum_bleu += bleu_score
    return sum_bleu / num_sentences


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform['code'](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform['nl'](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=sys_cfg.pad_index)
    tgt_batch = pad_sequence(tgt_batch, padding_value=sys_cfg.pad_index)
    return src_batch, tgt_batch


def train_epoch(model, optimizer, criterion):
    model.train()
    losses = 0
    train_iter = CodeNL(split='valid', src_path='./data/train.token.code', trg_path='./data/train.token.nl')
    train_dataloader = DataLoader(train_iter, batch_size=sys_cfg.batch_size, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(sys_cfg.device)
        tgt = tgt.to(sys_cfg.device)

        tgt_input = tgt[:-1, :] # LOST THE LAST

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def train(model, optimizer, criterion):
    max_bleu = 0
    fw = open('train.log', 'a', encoding='utf-8')
    epoch_step = 0
    bleu_scores = []
    for epoch in range(sys_cfg.epochs):
        epoch_step += 1
        start_time = timer()
        train_loss = train_epoch(model, optimizer, criterion)
        end_time = timer()
        avg_bleu = evaluate(transformer)
        if avg_bleu > max_bleu:
            max_bleu = avg_bleu
            torch.save(model, 'model.pth')
        x = list(range(1, epoch_step + 1))
        bleu_scores.append(avg_bleu)
        plot_table(x, bleu_scores)
        print(
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Avg BLEU: {avg_bleu:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
        fw.write(
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Avg BLEU: {avg_bleu:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s" + '\n')


if __name__ == '__main__':
    vocab_transform = get_vocab_transform()
    text_transform = get_text_transform(vocab_transform)
    SRC_VOCAB_SIZE = len(vocab_transform['code'])
    TGT_VOCAB_SIZE = len(vocab_transform['nl'])
    transformer = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)
    transformer = transformer.to(sys_cfg.device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=sys_cfg.pad_index)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    train(transformer, optimizer, criterion)
