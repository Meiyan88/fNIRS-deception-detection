import torch
from Model import fNIRS_T
from Readdata import Load_Dataset_f, Dataset
from pylab import *


##  apply in calculating the grad of the channel before intering the transformer

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

def toNumpy(tensor, dtype=np.float32, is_squeeze=True):
    if not isinstance(tensor, torch.Tensor):
        return tensor

    if is_squeeze:
        tensor = torch.squeeze(tensor)
    ndarray = tensor.detach().cpu().numpy().astype(dtype)
    return ndarray


def xx(v):
    def hook(module, grad_input, grad_output):
        v.append(grad_output[0])
    return hook

def yy(v):
    def hook(module, input, output):
        v.append(output)
    return hook


if __name__ == "__main__":


    # Select dataset
    dataset = ['ff10']
    ##0 rewrite 2
    dataset_id = 0
    print(dataset[dataset_id])

    # Select model
    models = ['fNIRS-T', 'fNIRS-PreT']
    models_id = 0
    print(models[models_id])

    # Select the specified path
    data_path = '/public/envs/Time_fold/five_fold_7/'
    # Save file
    save_path = '/public/envs/Time_fold/save/CAM/' + dataset[dataset_id] +'7nonoV13'
    assert os.path.exists(save_path) is False, 'path is exist'
    os.makedirs(save_path)

    n_runs = 0
    for n_tiems in range(1,6):
        'skf.split(feature,label):'
        n_runs=n_runs+1
        print('======================================\n', n_runs)
        path1 = '/public/envs/Time_fold/save/KFold1/fNIRS-T/' + str(n_runs)
        path = save_path + '/' + str(n_runs)
        if not os.path.exists(path):
            os.makedirs(path)
        trainpath = data_path + '/fold' + str(n_runs) + '/train/'

        X_train, y_train, s_train = Load_Dataset_f(trainpath)

        _, _, channels, sampling_points = X_train.shape
        X_train = X_train.reshape((y_train.shape[0], -1))

        X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))


        train_set = Dataset(X_train, y_train, s_train, transform=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)


        # -------------------------------------------------------------------------------------------------------------------- #
        device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

        net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64, pool='mean').to(
                    device)

        # -------------------------------------------------------------------------------------------------------------------- #

        m_state_dict = torch.load(path1 + '/test_max_acc.pt')
        net.load_state_dict(m_state_dict)
        # print(net)

        net.eval()
        test_running_acc = 0
        total = 0
        loss_steps = []

        for i, data in enumerate(train_loader):
            inputs, labels, sheet1 = data
            inputs = inputs.to(device)
            sheet1 = sheet1.to(device)
            sheet1 = torch.squeeze(sheet1)
            labels = labels.to(device)
            labels = torch.squeeze(labels)


            CAMo = zeros((41,1))
            for j in range(len(inputs)):

                v = []
                fmap_block = []
                grad_block = []
                net.transformer_channel.register_forward_hook(yy(v))
                net.transformer_channel.register_backward_hook(xx(v))

                outputs_c = net(inputs[j].unsqueeze(0),sheet1[j].unsqueeze(0)) #
                output_C = outputs_c.detach().cpu().numpy().astype(np.float32)
                if labels[j] == 1:

                    pred = outputs_c.argmax(dim=1, keepdim=True)
                    if pred == 1:
                        roi_area = outputs_c[0, 1]
                    else:
                        continue
                else:
                    continue


                roi_area.backward()
                feature = toNumpy(v[0])
                grad = toNumpy(v[1])

                cam = np.sum(grad * feature, axis=1)
                cam = (cam - cam.min()) / (cam.max() - cam.min())

                CAMo = CAMo + cam

            CAMo = np.mean(CAMo,0)
            f = open(path + '/cam' + str(n_runs) + '.txt', "w")  # .txt+
            f.write(str(CAMo))
            f.close()