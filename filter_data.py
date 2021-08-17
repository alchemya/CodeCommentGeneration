def static_length():
    fr = open('./`data/train.token.nl', 'r', encoding='utf-8')
    len_dict = {}
    for content in fr:
        length = len(content.split(' '))
        if length not in len_dict:
            len_dict[length] = 1
        else:
            len_dict[length] += 1
    print(len_dict)
    print(sorted(len_dict.items(), key=lambda x: (x[0], x[1])))


def filter_data():
    fr1 = open('./data/valid.token.code', 'r', encoding='utf-8')
    fr2 = open('./data/valid.token.nl', 'r', encoding='utf-8')
    fw1 = open('valid.token.code', 'w', encoding='utf-8')
    fw2 = open('valid.token.nl', 'w', encoding='utf-8')
    len_dict = {}
    for content1,content2 in zip(fr1,fr2):
        length = len(content1.split(' '))
        if length not in len_dict:
            if 5 < length < 200:
                fw1.write(content1)
                fw2.write(content2)
filter_data()