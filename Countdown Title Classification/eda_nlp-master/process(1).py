import pandas as pd


def main1():
    data = pd.read_csv('./data/markets.csv', encoding='utf_8_sig')
    name = data['name']
    cat = data['cat']
    data = data.dropna()
    f = open('./data/olddata.txt', 'w', encoding='utf_8_sig')
    for i in range(len(cat)):
        newline = str(cat[i]).strip()
        newline = newline + '\t' + str(name[i]).strip() + '\n'
        f.write(newline)


def main2():
    f = open('./data/data_augmented.txt', 'r', encoding='utf_8_sig')
    catlist = []
    namelist = []
    for line in f:
        line = line.strip()
        cat = line.split('\t')[0]
        name = line.split('\t')[1]
        namelist.append(name)
        catlist.append(cat)

    newdata = pd.DataFrame({
        'name': namelist,
        'cat': catlist
    })

    newdata.to_csv('./data/data_augmmented.csv', encoding='utf_8_sig', index=False)

if __name__ == '__main__':
    main2()