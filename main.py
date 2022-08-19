__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import torch
import torch.nn as nn
from torch.optim import Adam
from trainer.trainer import MainTrainer
from Dataset.Datasets import TOPODataset
from model.vgg_unet import VGG_Unet
from losses.loss import BCELoss
from metrics.segmentation import Accuracy, Precision, Recall
from transforms.transforms import Transforms
from writer.writer import TensorboardWriter
import warnings

warnings.filterwarnings("ignore")


class Parameters:
    def __init__(self):
        self.experiment = 'TOPO/'
        self.file = 'TL_VGG16-Unet_init5_ep15_data1e3/'
        self.load_file = 'VGG16-Unet/'
        self.train_data_dir = 'E:\Gorkem Can Ates_old\TOPO DATA\data/'
        self.train_target_dir = 'E:\Gorkem Can Ates_old\TOPO DATA/target'
        self.LOGDIR = f'runs/' + self.experiment + self.file
        self.FIG_PATH = 'RESULTS/' + self.experiment + self.file + 'images/'
        self.result_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'metrics/'
        self.model_SAVEPATH = 'RESULTS/' + self.experiment + self.file + 'models/'
        self.model_LOADPATH = 'RESULTS/' + self.experiment + self.load_file + 'models/'
        self.METRIC_CONDITION = Accuracy.__name__.lower()
        self.TO_TENSORBOARD = True
        self.VALIDATION = True
        self.PRETRAINED = False
        self.VGG_PRETRAINED = True
        self.REG_GRAD = False
        self.DEBUG = False
        self.SHUFFLE = True
        self.TRANSFORM = True
        self.DEVICE = 'cuda'


class HyperParameters:
    def __init__(self):
        self.NUM_EPOCHS = 75
        self.LEARNING_RATE = 0.001
        self.FINETUNE_LEARNING_RATE = 0.00001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.FINETUNE_EPOCH = 15
        self.INIT_ITERATION = 5
        self.IN_CHANNELS = 3
        self.NUM_CLASSES = 1
        self.train_batch_size = 8
        self.test_batch_size = 8
        self.TOTAL_DATA = 1000
        self.NORM = 'gn'
        self.METRIC_CONDITION = 'max'



class MAIN:
    def __init__(self):
        self.params = Parameters()
        self.hyperparams = HyperParameters()

        self.model = VGG_Unet(out_features=self.hyperparams.NUM_CLASSES,
                              norm_type=self.hyperparams.NORM,
                              pretrained=self.params.VGG_PRETRAINED,
                              reg_grad=self.params.REG_GRAD,
                              device=self.params.DEVICE)


        self.metrics = [Accuracy(),
                        Precision(),
                        Recall()
                        ]


        self.criterion = BCELoss()


        self.optimizer = Adam(params=self.model.parameters(),
                              lr=self.hyperparams.LEARNING_RATE,
                              betas=(self.hyperparams.beta1,
                                     self.hyperparams.beta2))

        self.transforms = Transforms(transform=self.params.TRANSFORM)


        self.dataset = TOPODataset(data_path=self.params.train_data_dir,
                                   target_path=self.params.train_target_dir,
                                   train_transform=self.transforms.train_transform,
                                   val_transform=self.transforms.val_transform,
                                   total_data=self.hyperparams.TOTAL_DATA,
                                   init_iteration=self.hyperparams.INIT_ITERATION,
                                   debug=self.params.DEBUG
                                    )


        self.writer = TensorboardWriter(PATH=self.params.LOGDIR,
                                        fig_path=self.params.FIG_PATH,
                                        num_data=48)


        self.trainer = MainTrainer(model=self.model,
                                   params=self.params,
                                   hyperparams=self.hyperparams,
                                   metrics=self.metrics,
                                   dataset=self.dataset,
                                   optimizer=self.optimizer,
                                   criterion=self.criterion,
                                   writer=self.writer
                                   if self.params.TO_TENSORBOARD else None
                                   )
        self.experiment_summary()

    def run(self):
        self.trainer.fit()

    def validate(self):
        results = self.trainer.validate()
        return results

    def experiment_summary(self):
        print(self.model)
        print(f'Total model parameters : '
              f'{sum(p.numel() for p in self.model.parameters())}')
        print(f'MODEL : {self.model._get_name()} ')
        print(f'CRITERION : {self.criterion._get_name()} ')
        print(f'BATCH SIZE : {self.hyperparams.train_batch_size} ')
        print(f'DEVICE : {self.params.DEVICE.upper()} ')
        print(f'TOTAL EPOCHS : {self.hyperparams.NUM_EPOCHS} ')
        print(f'INITIAL TOPO ITERATION : {self.hyperparams.INIT_ITERATION} ')
        print(f'FINE-TUNE EPOCH : {self.hyperparams.FINETUNE_EPOCH} ')
        print(f'INITIAL LR : {self.hyperparams.LEARNING_RATE} ')
        print(f'FINE-TUNE LR : {self.hyperparams.FINETUNE_LEARNING_RATE} ')


if __name__ == '__main__':
    trainer = MAIN()
    trainer.run()





