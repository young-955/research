import jieba
import jieba.posseg as posseg

jieba.load_userdict('dict.txt')

input = '我是一个人'

res = jieba.cut(input)
for r in res:
    print(r)

res = jieba.cut(input, cut_all=True)
for r in res:
    print(r)

res = jieba.cut_for_search(input)
for r in res:
    print(r)

res = jieba.cut_for_search(input, HMM=True)
for r in res:
    print(r)

# 词性标注
res = posseg.cut(input)
for r in res:
    print(r.word, r.flag)