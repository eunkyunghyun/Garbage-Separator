import os

value = 0
limit = len(os.listdir("./labels/foodwaste/egg"))
path = "./labels/"

f = open("./labels/train_label.csv", 'w')
f.write('file,label\n')
f.close()

f = open("./labels/test_label.csv", 'w')
f.write('file,label\n')
f.close()

for d1 in os.listdir(path):
    r1 = path + d1
    try:
        if d1 == "foodwaste":
            value = 0
        elif d1 == "generalwaste":
            value = 1
        for d2 in os.listdir(r1):
            r2 = r1 + '/' + d2
            for d3 in os.listdir(r2):
                r3 = r2 + '/' + d3
                number = ""
                for i in range(len(d3)):
                    if d3[i].isdigit():
                        number += d3[i]
                if int(number) < limit / 2 + 1:
                    f = open("./labels/train_label.csv", 'a')
                    f.write("{},{}\n".format(r3, value))
                    f.close()
                elif int(number) > limit / 2:
                    f = open("./labels/test_label.csv", 'a')
                    f.write("{},{}\n".format(r3, value))
                    f.close()
    except NotADirectoryError:
        pass