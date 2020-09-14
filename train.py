
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from networks.ccnet import Res_Deeplab, RCCAModule
import timeit
from utils.criterion import CriterionCrossEntropy, CriterionOhemCrossEntropy, CriterionDSN, CriterionOhemDSN
import DataLoader
#from torchsummary import summary
from networks.unet import UNet
start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)




# def str2bool(v):
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')






def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


# def adjust_learning_rate(optimizer, i_iter):
#     """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
#     lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
#     optimizer.param_groups[0]['lr'] = lr
#     return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main():
    """Create the model and start the training."""

    h, w = (320, 320)
    input_size = (h, w)

    cudnn.enabled = False

    # Create network.
    #model = Res_Deeplab(num_classes=1)
    print("model_load")
    #rcca_module = RCCAModule(64,128,1)
    #model = deeplab
    model = UNet(n_channels=3, n_classes=1)
    #summary(rcca_module, (64, 320, 320))

    #model.train()
    #model.float()
    #print(model)
    ohem = True
    # if ohem:
    #     criterion = CriterionOhemDSN()
    # else:
    #     criterion = CriterionDSN()  # CriterionCrossEntropy()


    criterion = torch.nn.BCELoss()

    imgs, masks = DataLoader.Loader('dataset/imgs','dataset/masks',(320,320))
    print("load_data")
    masks =np.expand_dims(masks, -1)
    lr = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.95, weight_decay=0.9)
    optimizer.zero_grad()
    num_steps = 100
    for i_iter in range(100):
        images, labels = imgs[i_iter],masks[i_iter]
        images = torch.from_numpy(np.expand_dims(images,0)).permute(0,3,1,2)
        labels = torch.from_numpy(np.expand_dims(labels,0)).permute(0,3,1,2)
        optimizer.zero_grad()
        lr = lr
        preds= model(images)
        print(preds.shape)
        #print(labels.shape, preds.shape)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()


        print('iter = {} of {} completed, loss = {}'.format(i_iter, num_steps, loss.item()))

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
