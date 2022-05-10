import jieba
import numpy as np
from wordcloud import WordCloud
from zhon.hanzi import punctuation
from snownlp import SnowNLP
import matplotlib.pyplot as plt


def getText(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        score = []
        comment = []
        for line in lines:
            line = line.strip().split('\t')
            comment.append(line[1])
            score.append(line[0])
        return score, comment


def getDescirbe(a: np.ndarray):
    """

    :param a:
    :return: 数组a的平均数、中位数、25%和75%分位数
    """
    return np.mean(a), np.median(a), np.percentile(a, 25), np.percentile(a, 75)


def draw_histogram(a: np.ndarray, lb: float, ub: float, step: float,
                   xlabel='', ylabel='', title='', save=False, savePath='./985.txt'):
    """

    :param savePath: 保存路径
    :param save: 是否保存
    :param title: 图标题
    :param ylabel: y轴注
    :param xlabel: x轴注
    :param a: 待描述的数组
    :param lb: 坐标轴下界
    :param ub: 坐标轴上界
    :param step: 坐标轴精度
    :return:
    """
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    bins = np.arange(lb, ub + step, step)
    plt.hist(a, bins, color='r', alpha=0.9)
    plt.xlim(lb, ub)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save:
        plt.savefig(savePath)
    plt.show()


def get_stop_words(path='./stop_words.txt') -> set:
    """
    获取停用词表
    :param path:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        wl = f.readlines()
    wl = [i.strip() for i in wl]
    return set(wl)


def get_wordcloud(word_list: list, savePath='./wordcloud.png'):
    """
    生成词云。
    :param word_list: 词语列表
    :param savePath: 词云保存路径
    :return:
    """
    sb = ' '.join(word_list)
    wc = WordCloud(font_path='C:\\Windows\\Font\\simkai.ttf', width=800, height=600, mode='RGBA',
                   background_color=None).generate(sb)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    wc.to_file(savePath)


def word_freq_analysis(word_list: list, top_x, pt=True):
    """
    词频统计。
    :param word_list: 词语列表
    :param top_x: 输出排名前几的词
    :param pt: 是否打印结果
    :return:
    """
    sett = set(word_list)
    top_x = min(top_x, len(sett))
    d = {}
    for i in sett:
        d[i] = word_list.count(i)
    res = sorted(d.items(), key=lambda x: x[1], reverse=True)  # 按value值进行排序
    res = res[:top_x]  # res是一个由(key, value)元组组成的列表
    if pt:
        print(f'top{top_x}词语统计结果:')

        for i in res:
            print(i[0], ' ' * (4 - len(i[0])), '次数:', i[1])
        print()
    return res


def pie_chart(stactics: list, labels: tuple, savePath='./pie_chart', show=True, title=''):
    """
    作出饼状图。
    :param stactics: 统计数字列表（包括每一类的个数）
    :param labels: 每一类对应的标签
    :param title: 饼状图标题
    :param savePath: 保存路径
    :param show: 是否展示图片
    :return:
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.pie(stactics, labels=labels, autopct='%1.2f%%', shadow=False)
    plt.title(title)
    plt.savefig(savePath)
    if show:
        plt.show()


if __name__ == '__main__':
    score, comment = getText('./pinglun.txt')
    cmt_list = []  # 评论分词结果列表
    emo_list = []  # 情感打分列表

    stop_ws = get_stop_words()  # 停用词集合
    neg_wordcut_list = []       # 消极评论分词列表

    pos = 0  # 积极评论数量
    neg = 0  # 消极评论数量
    for i in comment:
        # s = SnowNLP(i)
        for p in punctuation:
            i.replace(p, '')
        s = SnowNLP(i)
        emo_score = s.sentiments
        emo_list.append(emo_score)
        x = jieba.cut(i)
        x2 = []
        for i in x:
            if not (i in stop_ws):
                x2.append(i)  # 停用词处理
        cmt_list.extend(x2)
        if emo_score < 0.5:
            neg_wordcut_list.extend(x2)
            neg += 1
        else:
            pos += 1

    cmt_list = [i for i in cmt_list if i.strip() != '']
    neg_wordcut_list = [i for i in neg_wordcut_list if i.strip() != '']
    # 情感数据处理
    pie_chart([pos, neg], ('积极评论', '消极评论'), savePath='./pie.png', show=True)
    el = np.array(emo_list)
    print('情感数据描述:\nmean:{}\nmedian:{}\n25%:{}\n75:%{}\n'.format(*getDescirbe(el)))

    # 消极评论分析
    get_wordcloud(neg_wordcut_list, '消极评论词云.png')
    word_freq_analysis(neg_wordcut_list, 10, pt=True)

    sa = np.array([eval(i) for i in score])
    print('用户评分描述:\nmean:{}\nmedian:{}\n25%:{}\n75:%{}\n'.format(*getDescirbe(sa)))

    draw_histogram(el, 0, 1, 0.1, '评分', '数量', '情感分析直方图', True, './情感分析直方图.png')
    draw_histogram(sa, 0, 5, 0.5, '评分', '数量', '用户评分直方图', True, './用户评分直方图.png')
    get_wordcloud(cmt_list, '评论词云.png')
    word_freq_analysis(cmt_list, 10, pt=True)
