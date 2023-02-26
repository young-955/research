import pkuseg

'''
    model_name:代表不同领域的模型
        default
        news
        web
        medcine
        tourism
    user_dict: default or path
    postag: bool
'''
seg = pkuseg.pkuseg(model_name='default', user_dict='default')

input = '我是一个人'

res = seg.cut(input)

print(res)

# seg = pkuseg.pkuseg(model_name='default', user_dict='default', postag=True)

# res = seg.cut(input)

# print(res)

if __name__ == '__main__':
    pkuseg.test(input_file='input.txt', output_file='output.txt')