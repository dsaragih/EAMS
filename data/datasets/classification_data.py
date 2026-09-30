import os

import torch

from data.datasets.base_dataset import BaseDataset
from data.transforms.transforms import compose_graph_transforms
from data.types import MeshData
from models.layers.mesh_prepare import extract_features
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh


class ClassificationData(BaseDataset):

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
        self.dir = os.path.join(opt.dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, opt.phase)
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
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(
            file=path,
            opt=self.opt,
            hold_history=False,
            export_folder=self.opt.export_folder,
        )
        self._apply_runtime_transforms(mesh)
        meta = {"mesh": mesh, "label": label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta["edge_features"] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    def get_transform_timing_report(self) -> str:
        if self.profile_transforms and hasattr(self.transform, "get_timing_report"):
            return self.transform.get_timing_report()
        return "Transform profiling is not enabled. Set profile_transforms=True when creating the dataset."

    def reset_transform_timing(self) -> None:
        if self.profile_transforms and hasattr(self.transform, "reset_timing_stats"):
            self.transform.reset_timing_stats()

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase) == 1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes
