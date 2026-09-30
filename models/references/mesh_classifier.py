import torch
from models.networks import networks
from os.path import join
from util.util import seg_accuracy, print_network
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
import torch


def dice_coefficient(predictions, targets, smooth=1e-6):
    """
    计算Dice系数。

    参数:
    - predictions: 模型预测结果，维度为[B, N]，B是批次大小，N是点数。
    - targets: 真实标签，维度与predictions相同。
    - smooth: 平滑项，避免除以0的情况，默认为1e-6。
    返回:
    - dice: 计算得到的Dice系数。
    """
    # 确保predictions和targets为二值化数据（0或1）
    predictions = predictions.float()
    targets = targets.float()

    # 计算交集
    intersection = (predictions * targets).sum()

    # 计算Dice系数
    dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
    # print(dice)
    # 返回批次内的平均Dice系数
    return dice.mean()


# # 示例使用
# # 假设有一个批次大小为2，每个样本有5个点的预测和标签
# predictions = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=torch.float32)
# targets = torch.tensor([[1, 0, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=torch.float32)
#
# dice_score = dice_coefficient(predictions, targets)
# print(f"Dice Score: {dice_score}")



def dice_loss(output_2, target):
    """
    计算Dice Loss，忽略标签为-1的点。
    参数:
    - input: 模型输出，大小为[B, 2, N]。
    - target: 真实标签，大小为[B, N]，包含0, 1和-1，-1需要被忽略。

    返回:
    - loss: Dice Loss的值。
    """

    # print("output_2.shape, target.shape:", output_2.shape, target.shape)
    # 确保输入被softmax处理，转换成概率分布
    input_soft = F.softmax(output_2, dim=1)
    # 选择前景类别的预测概率，即类别1
    input_foreground = input_soft[:, 1, :]  # 大小为[B, N]
    # 将标签为-1的点的mask计算出来
    valid_mask = target != -1  # 大小为[B, N]
    # 仅选择有效的前景标签点
    target_foreground = (target == 1) & valid_mask  # 大小为[B, N]
    # 将input和target都转换为float类型以计算Dice系数
    input_foreground = input_foreground.type(torch.float32)
    target_foreground = target_foreground.type(torch.float32)
    # 计算分子：预测和真实前景的交集，注意要乘2，因为Dice系数的定义
    intersection = 2 * (input_foreground * target_foreground).sum(-1)
    # 计算分母：预测和真实前景的和
    union = input_foreground.sum(-1) + target_foreground.sum(-1)
    # 计算Dice Loss，添加一个小数避免除以0
    dice_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    # 忽略无效点（-1标签的点）的影响，计算平均损失
    dice_loss = dice_loss[valid_mask.any(dim=1)].mean()
    return dice_loss




def dice_loss_new(output, target, w_loss):
    """
    计算Dice Loss，忽略标签为-1的点，并将两种前景类别视为同一类。
    参数:
    - output: 模型输出，大小为[B, 3, H, W]。
    - target: 真实标签，大小为[B, H, W]，包含0, 1, 2和-1，-1需要被忽略。
    
    返回:
    - loss: Dice Loss的值。
    """
    num_classes = output.shape[1]
    batch_size = output.shape[0]
    dice_losses = []

    # 将所有前景视为一类
    target_combined = target.clone()
    # target_combined[target_combined == 2] = 1  # 将类别2转换为类别1
    for i in range(batch_size):
        # 获取当前样本的输出和标签
        current_output = output[i]  # 大小为[3, H, W]
        current_target = target_combined[i]  # 大小为[H, W]
        # 选择有效的标签点
        valid_mask = current_target != -1  # 大小为[H, W]
        valid_output = current_output[:, valid_mask]  # 大小为[3, valid_HW]
        valid_target = current_target[valid_mask]  # 大小为[valid_HW]
        
        w_loss = torch.tensor(w_loss).to('cuda:0')
        weight = w_loss[i][:, valid_mask] 
        # 对标签进行one-hot编码，大小为[3, valid_HW]
        valid_target_one_hot = F.one_hot(valid_target, num_classes=num_classes).permute(1, 0).float()
        # 确保输入被softmax处理，转换成概率分布
        valid_output_soft = F.softmax(valid_output, dim=0)

        

        # 计算Dice损失
        # print("valid_output_soft.shape, valid_target_one_hot.shape:", valid_output_soft.shape, valid_target_one_hot.shape)
        # print("weight.shape:", weight.shape, weight)

        intersection = 2 * (weight * valid_output_soft * valid_target_one_hot).sum(-1)
        union = (weight * valid_output_soft).sum(-1) + (weight * valid_target_one_hot).sum(-1)
        dice_loss_sample = 1 - (intersection + 1e-6) / (union + 1e-6)

        # 计算样本的平均Dice损失
        dice_loss_sample = dice_loss_sample.mean()
        dice_losses.append(dice_loss_sample)

    # 计算批次的平均Dice损失
    dice_loss_batch = torch.stack(dice_losses).mean()
    return dice_loss_batch

def dice_loss_standard(output, target):
    """
    计算标准Dice Loss，将前景类别分开计算损失并平均。
    参数:
    - output: 模型输出，大小为[B, 3, H, W]。
    - target: 真实标签，大小为[B, H, W]，包含0, 1, 2和-1，-1需要被忽略。
    
    返回:
    - loss: 标准Dice Loss的值。
    """
    num_classes = output.shape[1]
    batch_size = output.shape[0]
    dice_losses = []

    for i in range(batch_size):
        # 获取当前样本的输出和标签
        current_output = output[i]  # 大小为[3, H, W]
        current_target = target[i]  # 大小为[H, W]
        # 选择有效的标签点
        valid_mask = current_target != -1  # 大小为[H, W]
        valid_output = current_output[:, valid_mask]  # 大小为[3, valid_HW]
        valid_target = current_target[valid_mask]  # 大小为[valid_HW]
        
        # 对标签进行one-hot编码，大小为[3, valid_HW]
        valid_target_one_hot = F.one_hot(valid_target, num_classes=num_classes).permute(1, 0).float()
        # 确保输入被softmax处理，转换成概率分布
        valid_output_soft = F.softmax(valid_output, dim=0)

        # 对每个类别分别计算Dice损失
        dice_loss_per_class = []
        for class_idx in range(num_classes):
            intersection = 2 * (valid_output_soft[class_idx] * valid_target_one_hot[class_idx]).sum()
            union = valid_output_soft[class_idx].sum() + valid_target_one_hot[class_idx].sum()
            dice_loss_class = 1 - (intersection + 1e-6) / (union + 1e-6)
            dice_loss_per_class.append(dice_loss_class)

        # 计算样本的平均Dice损失
        dice_loss_sample = torch.stack(dice_loss_per_class).mean()
        dice_losses.append(dice_loss_sample)
    # 计算批次的平均Dice损失
    dice_loss_batch = torch.stack(dice_losses).mean()
    return dice_loss_batch


# 假设你的模型和数据都在CUDA设备上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 创建权重张量，并确保它在正确的设备上
weight = torch.tensor([1.0, 10.0, 10.0], device=device)

ce_loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-1)

class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            print("------------------------self.is_train:", self.is_train)
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):
        # print(f"+++++++++++++Test mode: Training mode = {self.net.training}")  # 应该输出 False
        label = self.labels-4
        # print("label.shape, self.labels.shape:", label.shape, self.labels.shape, type(label), type(self.labels))
        # out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time = self.net(self.edge_features, self.labels, self.mesh)
        out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time, a_w, b_w = self.net(self.edge_features, self.labels, self.mesh)
        # 示例
        return out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time, a_w, b_w
        # return out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time

    def backward(self, out, low_out, low_lab, w_loss, d_out, e_out):
        # print("out.shape, self.labels.shape:", out.shape, out, self.labels.shape, self.labels)
        # self.loss = self.criterion(out, self.labels) + self.criterion(out_2, self.labels)
        # self.loss = ce_loss(out, self.labels) # + dice_loss(out, self.labels)
        # print("out.shape, self.labels.shape:", out.shape, self.labels.shape, out.max(), self.labels.max())
        # print("low_out.shape, low_lab.shape:", low_out.shape, low_lab.shape, low_out.max(), low_lab.max())

        
        # print("out.shape, self.labels.shape:", out.shape, self.labels.shape)dice_loss_standard
        loss_t = ce_loss(out, self.labels) + dice_loss_new(out, self.labels, w_loss) # + ce_loss(low_out, low_lab)
        loss_d = ce_loss(d_out, self.labels) + dice_loss_new(d_out, self.labels, w_loss) # + ce_loss(low_out, low_lab)
        loss_e = ce_loss(e_out, self.labels) + dice_loss_new(e_out, self.labels, w_loss) # + ce_loss(low_out, low_lab)
        # loss_t = ce_loss(out, self.labels) + dice_loss_standard(out, self.labels) # + ce_loss(low_out, low_lab)
        # loss_d = ce_loss(d_out, self.labels) + dice_loss_standard(d_out, self.labels) # + ce_loss(low_out, low_lab)
        # loss_e = ce_loss(e_out, self.labels) + dice_loss_standard(e_out, self.labels) # + ce_loss(low_out, low_lab)
        self.loss = (loss_t + loss_d + loss_e) / 3
        # self.loss = loss_t
        # print("self.loss:", self.loss)
        self.loss.backward()

    def optimize_parameters(self, w_dict):
        # print("self.mesh:", self.mesh, len(w_dict))
        
        self.optimizer.zero_grad()
        # out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time = self.forward()
        out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time, a_w, b_w = self.forward()
        # print("out.shape:", type(out), out.shape)
        w_array = np.ones(out.shape)

        for i in range(len(self.mesh)):
            name = self.mesh[i].filename.split(".obj")[0]
            # print("name:", name, w_dict[name].shape)
            extended_w = np.repeat(w_dict[name][np.newaxis, :], repeats=3, axis=0)
            # print("extended_w.shape:", extended_w.shape)
            
            w_array[i, :, :w_dict[name].shape[0]] = extended_w

        
        """
        这里将低分辨率的结果传给鉴别器，用鉴别器和CE损失进行优化
        """
        

        # 下面的可以不变，即上面的是优化鉴别器，下面的仍然是对高分辨率的结果进行优化，需要用到CE和加权的Dice损失
        self.backward(out, low_out, low_lab, w_array, d_out, e_out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)
        net.eval()
        

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        # 设置网络为评估模式
        with torch.no_grad():
        # if 2>1:
            # print(f"+++++++++++++Test mode: Training mode = {self.net.training}")  # 应该输出 False
            """
            测试时这里的有两个输出，一个高分辨率，一个低分辨率，然后就拿高分辨率的结果即可；
            """
            start_time = time.time()
            # out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time = self.forward()
            out, low_out, low_lab, fe_low, rounded_tensor, coarse_tensor, d_out, e_out, avg_inference_time, a_w, b_w = self.forward()
            Parameters = self.count_parameters()
            # print("Parameters:", Parameters)
            end_time = time.time()
            inference_time = end_time - start_time - avg_inference_time
            # inference_time_all = inference_time_all + inference_time
            # print(f"推理时间: {inference_time:.6f} 秒")
            # compute number of correct 
            print("out.shape, d_out.shape, e_out.shape:", out.shape, d_out.shape, e_out.shape)
            # 从模型输出的张量 out 中，沿着类别维度（C维度）寻找每个位置的最大值索引 
            pred_class = out.data.max(1)[1]     # max 函数返回两个值：第一个是最大值，第二个是最大值的索引 
            pred_d_out = d_out.data.max(1)[1]
            pred_e_out = e_out.data.max(1)[1]
            

            

            
            dilation = rounded_tensor.data.max(1)[1] 
            erosion = coarse_tensor.data.max(1)[1] 
            fe_low = fe_low.permute(0, 2, 1)
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            # print("pred_class.shape, label_class.shape:", pred_class.shape, label_class.shape)
            # correct = self.get_accuracy(pred_class, label_class)
            correct, correct_ridge = 0, 0
            # print("correct:", correct, len(label_class), len(label_class[0]))
            # pred.eq(labels).sum()

            # print("correct, len(label_class):", correct, len(label_class), len(label_class[0]))
            # print("pred_class.shape, fe_low.shape, rounded_tensor.shape, coarse_tensor.shape:", pred_class.shape, fe_low.shape, rounded_tensor.shape, coarse_tensor.shape)
            base_dir = join(self.opt.checkpoints_dir, self.opt.name, "output")
            for file_i in range(label_class.shape[0]):
                mask_re = label_class[file_i] != -1
                # print(pred_class[mask_re].shape, label_class[mask_re].shape)
                list_pred = list(pred_class[file_i][mask_re].cpu() )
                list_label = list(label_class[file_i][mask_re].cpu() )
                
                list_dilation = list(dilation[file_i][mask_re].cpu() )
                list_erosion = list(erosion[file_i][mask_re].cpu() )
                
                list_d_pred = list(pred_d_out[file_i][mask_re].cpu() )
                list_e_pred = list(pred_e_out[file_i][mask_re].cpu() )

                fe_rendai = fe_low[:, :, 1]
                fe_ridge = fe_low[:, :, 2]
                print("fe_rendai.shape, fe_ridge.shape:", fe_rendai.shape, fe_ridge.shape)
                list_colormap_rendai = list(fe_rendai[file_i][mask_re].cpu())
                list_colormap_ridge = list(fe_ridge[file_i][mask_re].cpu())
                
                list_a_w = list(a_w[file_i][mask_re].cpu())
                list_b_w = list(b_w[file_i][mask_re].cpu())
                
                # print("len(list_pred), len(list_label):", len(list_pred), len(list_label), self.mesh[0].filename)
                # print("m", self.mesh[file_i].filename)
                file_name_pred = self.mesh[file_i].filename.split(".obj")[0] + "-pred.eseg"
                file_name_label = self.mesh[file_i].filename.split(".obj")[0] + "-label.eseg"
                path_file_i_p = os.path.join(f"{base_dir}/results", file_name_pred)
                path_file_i_l = os.path.join(f"{base_dir}/results", file_name_label)
                np.savetxt(path_file_i_p, list_pred, fmt='%d')
                np.savetxt(path_file_i_l, list_label, fmt='%d')
                
                file_name_dilation = self.mesh[file_i].filename.split(".obj")[0] + "-dilation.eseg"
                file_name_erosion = self.mesh[file_i].filename.split(".obj")[0] + "-erosion.eseg"
                path_file_i_d = os.path.join(f"{base_dir}/results", file_name_dilation)
                path_file_i_e = os.path.join(f"{base_dir}/results", file_name_erosion)
                np.savetxt(path_file_i_d, list_dilation, fmt='%d')
                np.savetxt(path_file_i_e, list_erosion, fmt='%d')
                
                file_name_colormap_rendai = self.mesh[file_i].filename.split(".obj")[0] + "-colormap_rendai.eseg"
                file_name_colormap_ridge = self.mesh[file_i].filename.split(".obj")[0] + "-colormap_ridge.eseg"
                path_file_i_c_rendai = os.path.join(f"{base_dir}/results", file_name_colormap_rendai)
                path_file_i_c_ridge = os.path.join(f"{base_dir}/results", file_name_colormap_ridge)
                np.savetxt(path_file_i_c_rendai, list_colormap_rendai, fmt='%f')
                np.savetxt(path_file_i_c_ridge, list_colormap_ridge, fmt='%f')
                
                

                file_name_pred_d = self.mesh[file_i].filename.split(".obj")[0] + "-pred-d.eseg"
                file_name_pred_e = self.mesh[file_i].filename.split(".obj")[0] + "-pred-e.eseg"
                path_file_i_pd = os.path.join(f"{base_dir}/results-P-A", file_name_pred_d)
                path_file_i_pe = os.path.join(f"{base_dir}/results-P-B", file_name_pred_e)
                np.savetxt(path_file_i_pd, list_d_pred, fmt='%d')
                np.savetxt(path_file_i_pe, list_e_pred, fmt='%d')
                
                
                file_name_weight_a = self.mesh[file_i].filename.split(".obj")[0] + "-weight-a.eseg"
                file_name_weight_b = self.mesh[file_i].filename.split(".obj")[0] + "-weight-b.eseg"
                path_file_i_a_weight = os.path.join(f"{base_dir}/results-A", file_name_weight_a)
                path_file_i_b_weight = os.path.join(f"{base_dir}/results-B", file_name_weight_b)
                np.savetxt(path_file_i_a_weight, list_a_w, fmt='%f')
                np.savetxt(path_file_i_b_weight, list_b_w, fmt='%f')


                # mask_re = label_class[file_i] == 1
                # print(pred_class[mask_re].shape, label_class[mask_re].shape)
                # list_pred_1 = list(pred_class[file_i][mask_re].cpu())
                # list_label_1 = list(label_class[file_i][mask_re].cpu())
                # print(pred_class[file_i][mask_re].shape, label_class[file_i][mask_re].shape)
                # pred_class[file_i][mask_re][pred_class[file_i][mask_re]==2] = 0
                # pred_class[file_i][mask_re][label_class[file_i][mask_re]==2] = 0
                # label_class[file_i][mask_re][label_class[file_i][mask_re]==2] = 0
                
                # 假设 pred_class 和 label_class 已经根据 file_i 和 mask_re 索引
                # 使用 masked_fill_() 来原地修改满足条件的元素值
                pred_class_masked = pred_class[file_i][mask_re]
                label_class_masked = label_class[file_i][mask_re]

                # 将等于2的值设置为0
                pred_class_masked.masked_fill_(pred_class_masked == 2, 0)
                label_class_masked.masked_fill_(label_class_masked == 2, 0)

                # 重新检查最大值和最小值
                # print("After correction:")
                # print("pred_class max and min:", pred_class_masked.max(), pred_class_masked.min())
                # print("label_class max and min:", label_class_masked.max(), label_class_masked.min())



                # print("pred_class[file_i][mask_re].max(), pred_class[file_i][mask_re].min():", pred_class[file_i][mask_re].max(), pred_class[file_i][mask_re].min())
                # print("label_class[file_i][mask_re].max(), label_class[file_i][mask_re].min():", label_class[file_i][mask_re].max(), label_class[file_i][mask_re].min())
                correct_1 = dice_coefficient(pred_class_masked, label_class_masked)

                acc_1 = correct_1
                correct = correct + correct_1
                
                # 使用 masked_fill_() 来原地修改满足条件的元素值
                pred_class_masked_ridge = pred_class[file_i][mask_re]
                label_class_masked_ridge = label_class[file_i][mask_re]

                # 将等于2的值设置为0
                pred_class_masked_ridge.masked_fill_(pred_class_masked == 1, 0)
                label_class_masked_ridge.masked_fill_(label_class_masked == 1, 0)
                # print("pred_class_masked_ridge.max(), label_class_masked_ridge.max():", pred_class_masked_ridge.max(), label_class_masked_ridge.max())
                # 计算肝脊的预测准确率 
                pred_class_masked_ridge[pred_class_masked_ridge==2] = 1
                label_class_masked_ridge[label_class_masked_ridge==2] = 1
                correct_2 = dice_coefficient(pred_class_masked_ridge, label_class_masked_ridge)

                acc_2 = correct_2
                correct_ridge = correct_ridge + correct_2
        
        
                print("这个数据保存了两类，预测镰状韧带和脊的准确率分别为:", acc_1, acc_2, self.mesh[file_i].filename)
            # print("两个输出的平均指标为：", result_1_acc/label_class.shape[0], result_2_acc/label_class.shape[0])
        # self.net.train(self.is_train)
        # print("inference_time_all:", inference_time_all)
        return correct, len(label_class), acc_1, acc_2, correct_ridge, inference_time

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            # correct = seg_accuracy(pred, self.soft_label, self.mesh)
            correct = pred.eq(labels).sum()
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
