from torch.nn import MSELoss, CrossEntropyLoss, BCELoss


LOSS_FN_DICT = {
    'mse': MSELoss,
    'bce': BCELoss,
    'crossentropy': CrossEntropyLoss
}