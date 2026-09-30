def create_model(opt):
    from .references.mesh_classifier import ClassifierModel # todo - get rid of this ?
    model = ClassifierModel(opt)
    return model
