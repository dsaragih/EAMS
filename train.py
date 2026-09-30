import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
import os 
import numpy as np 
def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, 'r'), dtype='float64')
    return seg_labels
if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    ACC_lig, ACC_rid, ACC_avg, Epo_lig, Epo_rid, Epo_ave = 0, 0, 0, 0, 0, 0


    """
    这里要读取所有训练集的边的权重文件，然后传入model.optimize_parameters()函数中进行加权 
    """
    folder_path = os.path.join(opt.dataroot, "Train_lossweight")

    w_dict = {} 
    for weight_name in os.listdir(folder_path):
        if weight_name.endswith('.eseg'):
            w_name = weight_name.split("-save-weight.eseg")[0]
            w_path = os.path.join(folder_path, weight_name)
            weight = read_seg(w_path)+1     # 得到每一条边对应的权重，应该+1 
            # print("weight.shape:", w_name, weight.shape, weight)
            w_dict[w_name] = weight
            # print("w_dict:", w_dict)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(w_dict)

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)
        if epoch % opt.run_test_freq == 0:
            acc, acc_lig, acc_rid = run_test(epoch)

            writer.plot_acc(acc, epoch)
            if acc_lig > ACC_lig:
                ACC_lig = acc_lig
                Epo_lig = epoch
            if acc_rid > ACC_rid:
                ACC_rid = acc_rid
                Epo_rid = epoch
            if (acc_lig+acc_rid)/2 > ACC_avg:
                ACC_avg = (acc_lig+acc_rid)/2
                Epo_avg = epoch
            print(Epo_lig, ACC_lig, Epo_rid, ACC_rid, Epo_avg, ACC_avg)
    writer.close()
