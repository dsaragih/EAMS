from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='real_test', help='train, val, test, real_test, etc') #todo delete.
        self.parser.add_argument('--which_epoch', type=str, default='10', help='latest, 192, xxx')
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files')
        self.is_train = False