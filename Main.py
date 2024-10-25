import torch
from Model import fNIRS_T
from Readdata import Load_Dataset_f, Dataset
from pylab import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, confusion_matrix
import sklearn.metrics as metrics


def toTensor(arr, dtype=torch.float32, add_channel=0):
    for i in range(add_channel):
        arr = arr[np.newaxis, ...]
    arr = torch.as_tensor(arr.copy(), dtype=dtype)
    return arr


def toNumpy(tensor, dtype=np.float32, is_squeeze=True):
    if is_squeeze:
        tensor = torch.squeeze(tensor)
    tensor = tensor.detach().cpu().numpy().astype(dtype)
    return tensor


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_metrics(pred_scores, pred_labels, gt_labels, is_print=False):
    if is_print:
        pred_print = pred_labels.reshape(-1, pred_labels.shape[0])
        print(pred_print)

    acc = accuracy_score(gt_labels, pred_labels)

    p, r, f1, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, pos_label=1, average="binary", zero_division=0
    )

    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    cm = confusion_matrix(gt_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_true=gt_labels, y_pred=pred_labels).ravel()
    specificity = tn / (tn + fp + 1e-6)
    sensitivity = tp / (tp + fn + 1e-6)

    return (float(auc), float(acc), float(p), float(r), float(f1), cm, float(specificity), float(sensitivity))


if __name__ == "__main__":
    # Training epochs
    EPOCH = 100

    # Select dataset
    dataset = ['f']
    dataset_id = 0
    print(dataset[dataset_id])

    # Select model
    models = ['fNIRS-T']
    models_id = 0
    print(models[models_id])

    # Select the specified path
    data_path = '/public/path/five_fold/'
    # Save file

    save_path = '/public/Time_fold/save/' + dataset[dataset_id] + \
                '/KFold1/' + models[models_id]
    assert os.path.exists(save_path) is False, 'path is exist'
    os.makedirs(save_path)

    # Load dataset and set flooding levels. Different models may have different flooding levels.

    if dataset[dataset_id] == 'f':
        flooding_level = [0.35, 0.4, 0.2]

    n_runs = 0
    for n_tiems in range(1, 6):
        n_runs += 1
        print('======================================\n', n_runs)
        path = save_path + '/' + str(n_runs)
        if not os.path.exists(path):
            os.makedirs(path)
        trainpath = data_path + '/fold' + str(n_runs) + '/train/'
        valpath = data_path + '/fold' + str(n_runs) + '/val/'
        testpath = data_path + '/fold' + str(n_runs) + '/test/'
        X_train, y_train, s_train = Load_Dataset_f(trainpath)
        X_val, y_val, s_val = Load_Dataset_f(valpath)
        X_test, y_test, s_test = Load_Dataset_f(testpath)

        _, _, channels, sampling_points = X_train.shape
        X_train = X_train.reshape((y_train.shape[0], -1))

        _, _, channels, sampling_points = X_val.shape
        X_val = X_val.reshape((y_val.shape[0], -1))

        _, _, channels, sampling_points = X_test.shape
        X_test = X_test.reshape((y_test.shape[0], -1))

        X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
        X_val = X_val.reshape((X_val.shape[0], 2, channels, -1))
        X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

        train_set = Dataset(X_train, y_train, s_train, transform=True)
        val_set = Dataset(X_val, y_val, s_val, transform=True)
        test_set = Dataset(X_test, y_test, s_test, transform=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

        # -------------------------------------------------------------------------------------------------------------------- #
        device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        if dataset[dataset_id] == 'f':

            net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64,
                          pool='mean').to(device)

        criterion = LabelSmoothing(0.1)
        optimizer = torch.optim.AdamW(net.parameters())
        lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        # -------------------------------------------------------------------------------------------------------------------- #
        val_max_acc = 0
        test_max_acc = 0

        train_loss_app, train_acc_app, val_loss_app, val_acc_app, test_loss_app, test_acc_app = [], [], [], [], [], []
        for epoch in range(EPOCH):

            list_score_softmax = []
            list_label = []
            list_pred = []
            list_score_softmax_posneg = []

            net.train()
            train_running_loss = 0
            train_running_acc = 0
            total = 0
            loss_steps = []
            for i, data in enumerate(train_loader):
                inputs, labels, sheet1 = data
                inputs = inputs.to(device)
                sheet1 = sheet1.to(device)
                labels = labels.to(device)
                labels = torch.squeeze(labels)
                sheet1 = torch.squeeze(sheet1)

                outputs = net(inputs, sheet1)
                loss = criterion(outputs, labels.long())

                # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
                if epoch < 20:
                    b = flooding_level[0]
                elif epoch < 50:
                    b = flooding_level[1]
                else:
                    b = flooding_level[2]

                # flooding
                loss = (loss - b).abs() + b

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_steps.append(loss.item())
                total += labels.shape[0]
                pred = outputs.argmax(dim=1, keepdim=True)
                train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

            train_running_loss = float(np.mean(loss_steps))
            train_running_acc = 100 * train_running_acc / total
            print('[%d, %d] Train loss: %0.4f' % (n_runs, epoch, train_running_loss))
            print('[%d, %d] Train acc: %0.3f%%' % (n_runs, epoch, train_running_acc))
            train_loss_app.append(train_running_loss)
            train_acc_app.append(train_running_acc)
            # -------------------------------------------------------------------------------------------------------------------- #
            net.eval()
            val_running_loss = 0
            val_running_acc = 0
            total = 0
            loss_steps = []
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels, sheet1 = data
                    inputs = inputs.to(device)
                    sheet1 = sheet1.to(device)
                    sheet1 = torch.squeeze(sheet1)
                    labels = labels.to(device)
                    labels = torch.squeeze(labels)
                    outputs = net(inputs, sheet1)
                    loss = criterion(outputs, labels.long())

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    val_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                val_running_acc = 100 * val_running_acc / total
                val_running_loss = float(np.mean(loss_steps))
                print('     [%d, %d] Val loss: %0.4f' % (n_runs, epoch, val_running_loss))
                print('     [%d, %d] Val acc: %0.3f%%' % (n_runs, epoch, val_running_acc))
                val_loss_app.append(val_running_loss)
                val_acc_app.append(val_running_acc)

                if val_running_acc > val_max_acc:
                    val_max_acc = val_running_acc
                    torch.save(net.state_dict(), path + '/val_max_acc.pt')
                    val_save = open(path + '/val_max_acc.txt', "w")
                    val_save.write("best_acc= %.3f" % (val_running_acc))
                    val_save.close()

            Softmax = torch.nn.Softmax(dim=1)
            net.eval()
            test_running_acc = 0
            total = 0
            pred_scores, preds, llabels = [], [], []
            m_state_dict = torch.load(path + '/val_max_acc.pt')
            net.load_state_dict(m_state_dict)
            for data in test_loader:
                inputs, labels, sheet1 = data
                inputs = inputs.to(device)
                sheet1 = sheet1.to(device)
                sheet1 = torch.squeeze(sheet1)
                labels = labels.to(device)
                labels = torch.squeeze(labels)
                outputs = net(inputs, sheet1)

                total += labels.shape[0]
                pred = outputs.argmax(dim=1, keepdim=True)
                pred_score = Softmax(outputs)

                scores_softmax = toNumpy(pred_score, is_squeeze=False)
                labelss = toNumpy(labels, np.long, is_squeeze=False)
                list_score_softmax.append(scores_softmax[:, 1])
                list_label.append(labelss)
                list_score_softmax_posneg.append(scores_softmax)

                test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

            list_score_softmax = np.concatenate(list_score_softmax, axis=0)
            list_label = np.concatenate(list_label, axis=0)
            list_score_softmax_posneg = np.concatenate(list_score_softmax_posneg, axis=0)
            list_score_softmax_posneg = np.argmax(list_score_softmax_posneg, axis=1)

            auc, acc, p, r, f1, cm, specificity, sensitivity = get_metrics(pred_scores=list_score_softmax,
                                                                           pred_labels=list_score_softmax_posneg,
                                                                           gt_labels=list_label,
                                                                           is_print=False)

            test_running_acc = 100 * test_running_acc / total
            print('     [%d, %d] Test acc: %0.3f%%' % (n_runs, epoch, test_running_acc))
            test_acc_app.append(test_running_acc)

            test_max_acc = test_running_acc
            torch.save(net.state_dict(), path + '/test_max_acc.pt')
            test_save = open(path + '/test_max_acc.txt', "w")
            test_save.write("best_acc= %.3f" % (test_running_acc))
            test_save.close()
            test_save = open(path + '/test_max_auc.txt', "w")
            test_save.write("best_acc= %.4f" % (auc))
            test_save.close()
            test_save = open(path + '/test_max_acc1.txt', "w")
            test_save.write("best_acc= %.4f" % (acc))
            test_save.close()
            test_save = open(path + '/test_max_spe.txt', "w")

            test_save.write("best_acc= %.4f" % (specificity))
            test_save.close()
            test_save = open(path + '/test_max_sen.txt', "w")

            test_save.write("best_acc= %.4f" % (sensitivity))
            test_save.close()
            test_save = open(path + '/test_max_f1.txt', "w")
            test_save.write("best_acc= %.4f" % (f1))
            test_save.close()
        lrStep.step()
