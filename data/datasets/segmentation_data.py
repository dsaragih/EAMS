import os
import numpy as np
import torch

from data.datasets.base_dataset import BaseDataset
from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData
from models.layers.mesh import Mesh
from models.layers.mesh_prepare import extract_features
from util.util import is_mesh_file, pad


class SegmentationData(BaseDataset):

    def __init__(
        self,
        opt,
        transforms=None,
        stage: str = "train",
        profile_transforms: bool = False,
    ):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.stage = stage
        self.profile_transforms = profile_transforms
        self.transform = None
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, "livermesh")
        self.paths = self.make_dataset(self.dir, self.opt.ids)
        self.seg_paths = self.get_seg_files(
            self.paths, os.path.join(self.root, "seg"), seg_ext=".eseg"
        )

        # self.sseg_paths = self.get_seg_files(self.paths, os.path.join(self.root, 'sseg'), seg_ext='.seseg')
        # Offset here just means the smallest label value
        self.classes, self.offset = self.get_n_segs(
            os.path.join(self.root, "classes.txt"), self.seg_paths
        )
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.nclasses = self.nclasses
        opt.input_nc = self.ninput_channels
        self.transform = compose_graph_transforms(
            transforms or [], profile=profile_transforms, stage=stage
        )

    def _mesh_to_runtime_data(self, mesh: Mesh) -> MeshData:
        edge_index = torch.from_numpy(mesh.edges.T).long()
        data = MeshData(
            pos=torch.from_numpy(mesh.vs).float(),
            edge_index=edge_index,
        )
        data.filename = mesh.filename
        return data

    def _apply_runtime_transforms(self, mesh: Mesh) -> None:
        if self.transform is None or not getattr(self.transform, "transforms", None):
            return

        transformed = self.transform(self._mesh_to_runtime_data(mesh))
        mesh.vs = (
            transformed.pos.detach().cpu().numpy().astype(mesh.vs.dtype, copy=False)
        )
        mesh.features = extract_features(mesh)

    def __getitem__(self, index):
        path = self.paths[index]
        mesh = Mesh(
            file=path,
            opt=self.opt,
            hold_history=True,
            export_folder=self.opt.export_folder,
        )
        meta = {}
        meta["mesh"] = mesh

        self._apply_runtime_transforms(mesh)

        # Segmentation label is just a long txt file of labels
        label = read_seg(self.seg_paths[index]) - self.offset
        label = pad(label, self.opt.ninput_edges, val=-1, dim=0)

        meta["label"] = torch.from_numpy(label).long()
        # soft_label = read_sseg(self.sseg_paths[index])
        # meta['soft_label'] = pad(soft_label, self.opt.ninput_edges, val=-1, dim=0)
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)

        # 提取edge_features的前num_channels_to_normalize个通道
        # print("--------edge_features.shape:", edge_features.shape)
        edge_features_first_part = edge_features[:5, :]
        # print("edge_features.shape, self.mean.shape, self.std.shape:", edge_features.shape, self.mean.shape, self.std.shape)

        # 由于self.mean和self.std的形状是(8, 1)，我们需要扩展它们的形状以匹配edge_features_first_part
        normalized_features = (
            edge_features_first_part - self.mean[:5, :].reshape(-1, 1)
        ) / (self.std[:5, :].reshape(-1, 1))
        # 将规范化后的特征放回edge_features的对应位置
        edge_features[:5, :] = normalized_features
        meta["edge_features"] = torch.from_numpy(edge_features).float()
        # meta['edge_features'] = (edge_features - self.mean) / self.std
        # print("self.mean, self.std:", self.mean, self.std)
        return meta

    def get_transform_timing_report(self) -> str:
        if self.profile_transforms and hasattr(self.transform, "get_timing_report"):
            return self.transform.get_timing_report()
        return "Transform profiling is not enabled. Set profile_transforms=True when creating the dataset."

    def reset_transform_timing(self) -> None:
        if self.profile_transforms and hasattr(self.transform, "reset_timing_stats"):
            self.transform.reset_timing_stats()

    def __len__(self):
        return self.size

    @staticmethod
    def get_seg_files(paths, seg_dir, seg_ext=".seg"):
        segs = []
        for path in paths:
            segfile = os.path.join(
                seg_dir, os.path.splitext(os.path.basename(path))[0] + seg_ext
            )
            # print("segfile:", segfile)
            assert os.path.isfile(segfile), "Segmentation file %s not found" % segfile
            segs.append(segfile)
        return segs

    @staticmethod
    def get_n_segs(classes_file, seg_files):
        if not os.path.isfile(classes_file):
            all_segs = np.array([], dtype="float64")
            for seg in seg_files:
                all_segs = np.concatenate((all_segs, read_seg(seg)))
            segnames = np.unique(all_segs)
            np.savetxt(classes_file, segnames, fmt="%d")
        classes = np.loadtxt(classes_file)
        offset = classes[0]
        classes = classes - offset
        return classes, offset

    @staticmethod
    def make_dataset(path, ids_file):
        meshes = []
        assert os.path.isdir(path), "%s is not a valid directory" % path

        # Read the list of file names from ids_file
        if ids_file and os.path.isfile(ids_file):
            with open(ids_file, "r") as f:
                allowed_names = set(line.strip() for line in f if line.strip())
        else:
            allowed_names = None

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if is_mesh_file(fname):
                    # If ids_file is provided, only include files in the allowed list
                    if (
                        allowed_names is None
                        or fname in allowed_names
                        or os.path.splitext(fname)[0] in allowed_names
                    ):
                        path = os.path.join(root, fname)
                        meshes.append(path)

        return meshes


def read_seg(seg):
    seg_labels = np.loadtxt(open(seg, "r"), dtype="float64")
    return seg_labels


def read_sseg(sseg_file):
    sseg_labels = read_seg(sseg_file)
    sseg_labels = np.array(sseg_labels > 0, dtype=np.int32)
    return sseg_labels
