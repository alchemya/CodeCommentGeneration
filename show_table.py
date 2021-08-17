import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_table(x, y):
    l1 = plt.plot(x, y, 'r--', label='BLEU')
    plt.title('average of BLEU')
    plt.xlabel('epoch')
    plt.ylabel('bleu_score')
    plt.legend()
    plt.savefig('bleu.jpg')


# x = [1, 2, 3]
# y = [2, 3, 6]
# plot_table(x, y)
