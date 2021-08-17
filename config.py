import torch


class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unk_index = 0  # 未知字符 unknown character
        self.pad_index = 1  # 填充字符 padding character
        self.bos_index = 2  # 开始字符 begin character
        self.eos_index = 3  # 结束字符 end character
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']  # special symbols in the word vocabulary
        self.batch_size = 128  # 批量大小 batch size
        self.embedding_size = 512  # 词嵌入维度 embedding size
        self.dim_feedforward = 512  # 全连接维度 Fully connected dimension
        self.num_heads = 8  # transformer heads num
        self.num_encoder_layers = 3  # 编码器层数 encoder layer
        self.num_decoder_layers = 3  # 解码器层数 decoder layer
        self.epochs = 20  # 迭代次数 epoch number


sys_cfg = Config()
