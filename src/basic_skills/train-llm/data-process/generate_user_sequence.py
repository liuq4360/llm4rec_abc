import csv

user_sequence_list = []
with open('../data/mind/behaviors.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    for row in reader:
        uid = row[1]
        history = row[3]
        if len(history.split(" ")) >= 5:  # 用户至少要有5个点击历史
            r = uid + " " + history
            user_sequence_list.append(r)

user_sequence_path = "../data/mind/user_sequence.txt"
with open(user_sequence_path, 'a') as file:
    for r in user_sequence_list:
        file.write(r + "\n")
