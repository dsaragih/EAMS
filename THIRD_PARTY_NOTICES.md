# Third-party acknowledgments

Daniel Saragih's release contributions are MIT licensed. The existing root MIT notice (`Copyright (c) 2024 AAAAAA`) is retained verbatim and supplemented rather than replaced. Upstream author/contact comments in source are attribution, not release credentials.

| Component | Upstream | Included code / notice |
| --- | --- | --- |
| PointTransformerV3 | [Pointcept/PTv3](https://github.com/Pointcept/PointTransformerV3) | `PointTransformerV3/model.py`, `serialization/`, initializer; [MIT](PointTransformerV3/LICENSE) |
| MeshCNN | [MeshCNN](https://github.com/ranahanocka/MeshCNN) | Legacy mesh convolution/pooling/data utilities; [MIT](licenses/MeshCNN.txt) |
| PointNet++ utilities / PointConv | [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) | `models/point2_utils.py`, `models/layers/point_conv.py`; [MIT](licenses/PointNet_PointNet2.txt), original author comments retained |
| DGCNN | [dgcnn.pytorch](https://github.com/antao97/dgcnn.pytorch) | `models/networks/model_dgcnn.py`; [MIT](licenses/DGCNN.txt), Yue Wang/An Tao/Ziyi Wu attribution retained |

PTv3 vendoring is based on upstream revision `3229e9b7de1770c8ad17c316f8e349982de509f8`, with the reviewed local relative-serialization import and required local package initializer. Nested Pointcept, assets, Git metadata and caches are excluded. The runtime files are vendored as ordinary files, not a submodule.

EMNN/EGNN message passing, Fast-TGCN-compatible adapters, spectral mesh signatures and the dental/liver dataset protocols build on the works cited in the EAMS paper and on retained source attribution. Fast-TGCN's separate training project is not vendored. Dependencies (PyTorch, PyG, Lightning, SciPy, trimesh, PyMeshLab and optional PTv3 packages) retain their own licenses; install them separately.
