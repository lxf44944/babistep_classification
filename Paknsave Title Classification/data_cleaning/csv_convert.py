import csv
import json
import os


def find_all_file(data):
    file_list = []
    for root, ds, fs in os.walk(data):
        for f in fs:
            f = os.path.join(root, f)
            file_list.append(f)
    return file_list


def trans_0(json_path, csv_path):
    json_file = open(json_path, 'r', encoding='utf8')
    csv_file = open(csv_path, 'w', newline='')
    keys = []
    writer = csv.writer(csv_file)

    json_data = json_file.read()
    dic_data = json.loads(json_data, encoding='utf8')

    for dic in dic_data:
        keys = dic.keys()
        # 写入列名
        writer.writerow(keys)
        break

    for dic in dic_data:
        for key in keys:
            if key not in dic:
                dic[key] = ''
        writer.writerow(dic.values())
    json_file.close()
    csv_file.close()


def trans(json_path, csv_path):
    json_file = open(json_path, 'r', encoding='utf8')
    csv_file = open(csv_path, 'a', newline='')
    keys = []
    writer = csv.writer(csv_file)

    json_data = json_file.read()
    dic_data = json.loads(json_data, encoding='utf8')

    for dic in dic_data:
        for key in keys:
            if key not in dic:
                dic[key] = ''
        writer.writerow(dic.values())
    json_file.close()
    csv_file.close()


def main():
    data = '../raw_data/'
    fs = find_all_file(data)
    csv_dir = "./raw_data.csv"
    fs = [f for f in fs if "cats.json" not in f]
    for i in range(len(fs)):
        print(str(i) + "\t" + fs[i])
    trans_0(fs[0], csv_dir)
    for i in range(1, len(fs)):
        trans(fs[i], csv_dir)


if __name__ == '__main__':
    main()
