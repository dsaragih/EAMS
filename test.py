from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    result_1_acc, result_2_acc, data_number = 0, 0, 0
    for i, data in enumerate(dataset):

        model.set_input(data)
        ncorrect, nexamples, acc_1, acc_2, correct_ridge = model.test()
        result_1_acc = result_1_acc + ncorrect
        result_2_acc = result_2_acc + correct_ridge

        data_number = data_number + nexamples

        writer.update_counter(ncorrect, nexamples)

        print("i:", i, type(data))
    writer.print_acc(epoch, writer.acc)
    print("总输出结果的平均值：", result_1_acc/data_number, result_2_acc/data_number)
    """
    下面可以将两种类别前景的ACC都返回，主要就是在train.py代码中，打印平均指标最高的模型的epoch，然后后续用于测试！
    想象中，加权融合的测试的Dice应该是最高的！这个需要Dice损失作为约束，这样可以优先学习到两种前景在位置上的相关关系！从而指导权重的计算和生成！
    """
    return writer.acc, result_1_acc/data_number, result_2_acc/data_number


if __name__ == '__main__':
    run_test()
