def freeze_params(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
