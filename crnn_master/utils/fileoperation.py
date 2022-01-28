import os
def get_chinese(path):
    with open(path, 'r', encoding='utf-8') as f:
        cur_path = os.path.abspath(os.path.dirname(__name__))
        print(cur_path)
        chinese = f.read()
        f.close()
        return chinese

if __name__ == '__main__':
    path = '../crnn_master/data/formula.txt'
    with open(path, 'r', encoding='utf-8') as f:
        print('--')

