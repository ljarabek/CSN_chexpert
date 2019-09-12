import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
device_ids = [0,1]

from argparse import ArgumentParser
import tensorboard
import torchvision.transforms as transforms
from data.data_chexpert import CXRDataset
from data.data_chexpert_BALANCE import balanced_dataloader
from torch.utils.data import DataLoader
from utils.logger import Logger
import torch
from collections import OrderedDict
import time
import torch.nn as nn
from utils.loss import WeightBCELoss
from models_multich import CSN_backbone, Classifier
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class BaseRun():
    def __init__(self, args):
        self.args = args

        if self.args.multi_channel is not None:
            from models_multich import CSN_backbone, Classifier
        else:
            from models import CSN_backbone, Classifier

        print("LETS DO THIS!!")
        print("Visualizations don't work on multiple GPUs!!")
        self.params = list()
        self.epoch = 0
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        if self.args.tag is not None:
            self.model_dir = self.args.save_dir + "TAG_{}_".format(self.args.tag) + str(
                time.strftime('%Y%m%d-%H:%M', time.gmtime())) + "/"
        else:
            self.model_dir = self.args.save_dir + str(time.strftime('%Y%m%d-%H:%M', time.gmtime())) + "/"
        print("model saved to : " + self.model_dir)
        self.im_dir = self.model_dir + "visualizations/"
        os.makedirs(self.im_dir, exist_ok=True)
        self.train_trainsform = self._init_train_transform()
        self.val_transform = self._init_train_transform()  ## TODO val_transform
        self.train_dataset = self._loader(self._init_dataset(self.args.train_csv, transform=self.train_trainsform),
                                          shuffle=False, validation=False)
        self._val_dataset = self._init_dataset(self.args.test_csv, transform=self.val_transform)
        self.class_names = self._val_dataset.classes

        self.logger = self._init_logger()

        self.auc = list()

        self.no_classes = len(self.class_names)
        if not self.args.train:  # shuffle only on train
            self.val_dataset = self._loader(self._val_dataset,
                                            shuffle=True, validation=True)
        else:
            self.val_dataset = self._loader(self._val_dataset,
                                            shuffle=False, validation=True)

        if self.args.tag != None:
            self.criterion = self._init_BCE()
        else:
            self.criterion = self._init_wBCE()
        self.model = self._init_model(Classifier(args))
        # self.params += list(self.model.parameters())
        print("no_params  =  " + str(len(self.params)))
        # INIT OPTIMIZER AT END!!
        self.optimizer = self._init_optimizer_sgd()

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _init_model(self, modellol):

        try:
            dct = torch.load(self.args.load_model_path)
            new_state_dict = OrderedDict()
            for k, v in dct.items():
                name = k[7:]  # remove `module.` (Dataparallel makes module.)
                new_state_dict[name] = v

            modellol.load_state_dict(new_state_dict)
            print("loaded model!!")
        except:
            print("loading model failed")
        modellol.to(device)
        if self.args.train:
            modellol = MyDataParallel(modellol, device_ids=device_ids)
        self.params += list(modellol.parameters())
        return modellol

    def _init_optimizer_adam(self):
        optimizer = torch.optim.Adam(params=self.params, lr=self.args.learning_rate)
        try:
            optimizer.load_state_dict(self.model_state_dict['optimizer'])
            return optimizer
        except:
            return optimizer

    def _init_optimizer_sgd(self):
        optimizer = torch.optim.SGD(self.params, lr=self.args.learning_rate, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        return optimizer

    def _init_wBCE(self):
        return WeightBCELoss()  # .cuda()

    def _init_BCE(self):
        if self.args.tag == None:
            return nn.BCELoss()
        else:
            weights = np.zeros(shape=self.no_classes)
            weights[self.args.tag] = 1.0
            return nn.BCELoss(weight=torch.tensor(weights).float().to(device))

    def _loader(self, dataset, shuffle=None, validation=False):
        if self.args.tag == None:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        if validation:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        else:
            return balanced_dataloader(dataset, self.args.tag, self.args)

    def _init_dataset(self, type, transform):
        return CXRDataset(self.args.data_root, dataset_source=self.args.dataset_source,
                          dataset_type=type, transform=transform, specific_image=self.args.specific_image)

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def train(self):
        tr_l_top = 1e9
        test_l_top = 1e9
        savedir = self.model_dir
        os.makedirs(savedir, exist_ok=True)
        for e in tqdm(range(self.epochs)):
            self.epoch = e
            tr_l = self.model_train()
            print("epoch train %s %s" % (e, tr_l))
            test_l = self.model_val()
            print("epoch validation %s %s" % (e, test_l))
            if test_l < test_l_top:
                test_l_top = test_l
                print("saved_model_val")
                torch.save(self.model.state_dict(), savedir + "best_val.pt")
            if tr_l < tr_l_top:
                tr_l_top = tr_l
                print("saved_model_tr")
                torch.save(self.model.state_dict(), savedir + "best_train.pt")

            self.logger.scalar_summary("train_loss", tr_l, e)
            self.logger.scalar_summary("val_loss", test_l, e)

    def model_train(self):
        raise NotImplementedError

    def model_val(self):
        raise NotImplementedError


class CSNrun(BaseRun):
    def __init__(self, args):
        super(CSNrun, self).__init__(args)
        # self.test_f()
        self.train()

    def model_train(self):
        if not self.args.train:
            return 0
        self.model = self.model.train()
        epoch_loss = 0
        for i, (images, labels, names) in enumerate(self.train_dataset):
            self.model.CSN.visualization_filename = self.im_dir + "batch{}_epoch_{}.png".format(i, self.epoch)
            self.model.CSN.visualize = False
            #print("batch %s"%time.time())
            if i < 10 or i % 1000 == 0:
                self.model.CSN.visualize = True

            self.optimizer.zero_grad()
            inp = torch.tensor(images).to(device)
            labels = torch.tensor(labels).float().to(device)
            otpt = self.model(inp)

            loss = self.criterion(otpt, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss
            if i%150==0: # todo: for batch size = 32 is 75!!
                print("saving batch_model")
                savedir = self.model_dir
                os.makedirs(savedir, exist_ok=True)
                self.model_val(dirname=savedir +"epoch_{}_batch_{}/".format(self.epoch,i))

        return epoch_loss

    def model_val(self, dirname=None):
        epoch_loss = 0
        self.model = self.model.eval()
        preds = list()
        truths = list()
        with torch.no_grad():
            for i, (images, labels, names) in enumerate(self.val_dataset):
                self.model.CSN.visualization_filename = self.im_dir + "batch{}_epoch_{}_val.png".format(i, self.epoch)

                self.model.CSN.visualize = False
                if i < 10 or i % 1000 == 0:
                    self.model.CSN.visualize = True
                inp = torch.tensor(images).to(device)
                labels = torch.tensor(labels).float().to(device)
                otpt = self.model(inp)
                tags_np = otpt.cpu().detach().numpy()
                loss = self.criterion(otpt, labels)
                labels_np = labels.cpu().detach().numpy()
                epoch_loss += loss

                for id_, (truth, pred) in enumerate(zip(labels_np, tags_np)):
                    truths.append(truth)
                    preds.append(pred)

        truths = np.array(truths)
        preds = np.array(preds)
        if dirname is None:
            print(truths.shape)
            print(preds.shape)
            for i in range(14):
                try:
                    print("ROCAUC for {} is {}".format(self.train_dataset.dataset.classes[i],
                                                       roc_auc_score(truths[:, i], preds[:, i], None)))
                except:
                    print("missing values")
        else:
            try:
                print(dirname)
                os.makedirs(dirname, exist_ok=True)
            except:
                print("direxists for validation")
            rows = list()
            for i in range(14):
                try:
                    rows.append(str([self.train_dataset.dataset.classes[i], roc_auc_score(truths[:, i], preds[:, i], None)]) + "\n")
                except:
                    rows.append("missing values\n")
            with open(dirname + "performance.txt", "w") as f:
                print("writing perf.txt")
                f.writelines(rows)
            torch.save(self.model.state_dict(), dirname + "model.pt")
            print("savedmodel")
        print("finished_val!")
        return epoch_loss


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train", type=bool, default=True,
                        help="if False, the model is ran on only the validation set")
    parser.add_argument("--specific_image", default = None)
    parser.add_argument("--CSN", type=bool, default=True,
                        help="whether to use CSN or no. If False --> default densenet is used. This was our baseline")
    parser.add_argument("--multi_channel", default=15) # set to int to number of wanted channels >> MUST BE DIVISIBLE BY 3 (RGB!)

    parser.add_argument("--tag", type=int, default=4,
                        help="makes training per this specified tag with 50:50 positive-negative balanced sampling")
    parser.add_argument("--visualize_", type=bool, default=False,
                        help="saves plots of a sample image, transformed image and histogram")  # visualize - make False for multiGPU!!

    parser.add_argument("--save_dir", default="./RUNS_ensamble/")
    parser.add_argument("--batch_size", type=int, default=16)  # ~140 is the max for 4x v100
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", default=3e-4)
    parser.add_argument("--data_root", default='/raid/Medical/DX')
    parser.add_argument("--dataset_source", default='./data/2Label/', help="dir of CSVs")
    parser.add_argument("--num_workers", default=15, help="number of workers for dataloader")
    parser.add_argument("--train_csv",
                        default="2Ltrain_Chex&MIMIC_Shuffle_Frontal",
                        help="CSV name")  # 2Ltrain_Chex&MIMIC_Shuffle_Frontal '2Ltrain_1000sample_Shuffle_Frontal'
    parser.add_argument("--test_csv", default='2Lvalid_Frontal')
    parser.add_argument('--resize', type=int, default=480)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--load_model_path',
                        default="./RUNS_ensamble/20190910-13:49/epoch_6_batch_2100/model.pt")#"./RUNS/EVEN_BESTER/best_val.pt")  # )"/home/leon/dev/CSN/RUNS/TAG_0/20190828-13:07/best_val.pt")  # "/home/leon/dev/CSN/RUNS/no_CSN/TRAINED/best_val.pt")#"/home/leon/dev/CSN/RUNS/no_CSN/20190827-13:07/best_val.pt")  # "./RUNS/BESTER_RUN_CSN/best_val.pt")  # "best_val.pt")
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=5e-3)

    args = parser.parse_args()
    if len(device_ids) > 1:
        args.visualize_ = False
    tag = args.tag
    if args.tag != None:
        args.save_dir = "./RUNS/TAG_{}/".format(str(tag))
    print(args)
    cr = CSNrun(args)
