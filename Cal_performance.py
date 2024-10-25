import numpy as np

test_acc, test_auc, test_sen, test_spe, test_f1 = [], [], [], [], []
for tr in range(1, 6):
    path = '/RESULT save path/' + str(tr)
    test_max_acc = open(path + '/test_max_acc1.txt', "r")
    string = test_max_acc.read()
    acc = string.split('best_acc=')[1]
    acc = float(acc)
    test_acc.append(acc)

    test_max_auc = open(path + '/test_max_auc.txt', "r")
    string = test_max_auc.read()
    auc = string.split('best_acc=')[1]
    auc = float(auc)
    test_auc.append(auc)

    test_max_sen = open(path + '/test_max_sen.txt', "r")
    string = test_max_sen.read()
    sen = string.split('best_acc=')[1]
    sen = float(sen)
    test_sen.append(sen)

    test_max_spe = open(path + '/test_max_spe.txt', "r")
    string = test_max_spe.read()
    spe = string.split('best_acc=')[1]
    spe = float(spe)
    test_spe.append(spe)

    test_max_f1 = open(path + '/test_max_f1.txt', "r")
    string = test_max_f1.read()
    f1 = string.split('best_acc=')[1]
    f1 = float(f1)
    test_f1.append(f1)

print('-----------------------Result ---------------------------')
test_acc = np.array(test_acc)
print('test_acc')
print('mean = %.4f' % np.mean(test_acc))
print('std = %.4f' % np.std(test_acc))
print(test_acc)

test_auc = np.array(test_auc)
print('test_auc')
print('mean = %.4f' % np.mean(test_auc))
print('std = %.4f' % np.std(test_auc))
print(test_auc)

test_sen = np.array(test_sen)
print('test_sen')
print('mean = %.4f' % np.mean(test_sen))
print('std = %.4f' % np.std(test_sen))
print(test_sen)

test_spe = np.array(test_spe)
print('test_spe')
print('mean = %.4f' % np.mean(test_spe))
print('std = %.4f' % np.std(test_spe))
print(test_spe)

test_f1 = np.array(test_f1)
print('test_f1')
print('mean = %.4f' % np.mean(test_f1))
print('std = %.4f' % np.std(test_f1))
print(test_f1)
