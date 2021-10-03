src = open("../labels/test_label.csv", 'r').readlines()
tar = open("../labels/test_result_label.csv", 'r').readlines()
limit = 301
var = 0

for i in range(limit):
    if src[i].split(',')[1].replace('\n', '') == tar[i].replace('\n', ''):
        var += 1

percentage = var / limit * 100

print("percentage: {}%".format(round(percentage)))