import torch.optim as optim


def get_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        return optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
        )
    raise NotImplementedError(f'Optimizer {cfg.TRAIN.OPTIMIZER} not supported')


def get_scheduler(config, optimizer):
    if config.TRAIN.LR_TYPE == 'linear':
        epoch_count = 0
        n_epochs = config.TRAIN.LR_STEP[0]
        n_epochs_decay = config.TRAIN.LR_STEP[1]

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif config.TRAIN.LR_TYPE == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=config.TRAIN.LR_STEP,
                                         gamma=config.TRAIN.LR_FACTOR)
    elif config.TRAIN.LR_TYPE == 'multistep':
        return optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP,
                                               config.TRAIN.LR_FACTOR)
    raise NotImplementedError(f'LR type {config.TRAIN.LR_TYPE} not supported')
