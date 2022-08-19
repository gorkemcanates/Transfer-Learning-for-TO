import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class TensorboardWriter(SummaryWriter):
    def __init__(self,
                 PATH,
                 fig_path,
                 num_data=48,
                 clear=None):
        super(TensorboardWriter, self).__init__(PATH)
        self.fig_path = fig_path
        self.num_data = num_data

        if clear is not None:
            self.clear_Tensorboard(clear)

    def write_results(self,
                      keys: list,
                      results_train,
                      results_test,
                      epoch):
        for metric, index in zip(keys, range(len(results_test))):
            self.add_scalars(metric, {'Training': results_train[index],
                                      'Validation': results_test[index]},
                             epoch + 1)

    def write_images(self,
                    keys: list,
                    data: list,
                    step,
                    C=3,
                    best=True):

        rand_images = self.get_random_predictions(data=data,
                                                  num_data=self.num_data)


        image = rand_images[0]
        target = rand_images[1].unsqueeze(1)
        prediction = rand_images[2]

        if C == 1:

            target_arg = torch.eye(C + 1)[target.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
            pred_arg = torch.eye(C + 1)[prediction.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
            images = [image,
                      torch.argmax(target_arg, dim=1, keepdim=True),
                      torch.argmax(pred_arg, dim=1, keepdim=True)
                      ]
        else:
            target_arg = torch.eye(C)[target.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)
            pred_arg = torch.eye(C)[prediction.type(
                torch.LongTensor).squeeze(1)].permute(0, 3, 1, 2)

            images = [1 - image,
                      1 - target_arg,
                      1 - pred_arg
                      ]
        if best:
            self.visualize(data=images,
                           step=step)

        for key, im in zip(keys, images):
            self.add_images(f'' + key,
                            im,
                            global_step=step)



    def visualize(self, data, step):
        plt.ioff()
        if not os.path.exists(self.fig_path + 'data/'):
            os.mkdir(self.fig_path + 'data/')
        fig_data = plt.figure(figsize=(16, 12))
        for i in range(self.num_data):
            ax = fig_data.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            plt.imshow(1-data[0][i].permute(1, 2, 0), cmap='gray')

        fig_data.savefig(self.fig_path + 'data/' + str(step) + '.png')
        plt.close(fig_data)

        if not os.path.exists(self.fig_path + 'target/'):
            os.mkdir(self.fig_path + 'target/')
        fig_tar = plt.figure(figsize=(16, 12))
        for i in range(self.num_data):
            ax = fig_tar.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            plt.imshow(1-data[1][i].permute(1, 2, 0), cmap='gray')

        fig_tar.savefig(self.fig_path + 'target/' + str(step) + '.png')
        plt.close(fig_tar)

        if not os.path.exists(self.fig_path + 'prediction/'):
            os.mkdir(self.fig_path + 'prediction/')
        fig_pred = plt.figure(figsize=(16, 12))
        for i in range(self.num_data):
            ax = fig_pred.add_subplot(8, 6, i + 1, xticks=[], yticks=[])
            plt.imshow(1-data[2][i].permute(1, 2, 0), cmap='gray')

        fig_pred.savefig(self.fig_path + 'prediction/' + str(step) + '.png')
        plt.close(fig_pred)

    def write_hyperparams(self,
                          hparams_dict,
                          metric_dict):

        self.add_hparams(hparam_dict=hparams_dict,
                         metric_dict=metric_dict)

    def write_histogram(self):
        pass

    @staticmethod
    def clear_Tensorboard(file):
        dir = 'runs/' + file
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))

    @staticmethod
    def get_random_predictions(data: list,
                               num_data=36):
        seed = torch.randint(low=0,
                             high=len(data[0]),
                             size=(num_data,))
        random_data = [i[seed] for i in data]
        return random_data
