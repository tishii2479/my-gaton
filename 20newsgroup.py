from pprint import pprint
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(
    data_home='data', subset='train', remove=('headers', 'footers', 'quotes'))
target_names = list(newsgroups_train.target_names)

for i in range(10):
    print(newsgroups_train.data[i])
    print(target_names[newsgroups_train.target[i]])
