

path = './model_data/my_classes.txt'
with open(path, encoding='utf-8') as f:
    class_names = f.readlines()
print(class_names)