import torch
from .lars import LARS
from .cos_anneal import cosine_annealing

def load_optimizer_scheduler(model, args, train_loader):
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                        exclude_from_weight_decay=["batch_normalization", "bias"])
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                    weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=[args.epochs * len(train_loader) * 10, ], 
                                                         gamma=1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps= args.warmup_epochs * len(train_loader))
        )
    elif args.scheduler == None:
        scheduler = None
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False
    return optimizer, scheduler

def load_fc_optimizer_scheduler(model, args, train_loader):
    if args.fc_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.fc_lr, weight_decay=args.fc_weight_decay)
    elif args.fc_optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.fc_lr, weight_decay=args.fc_weight_decay,
                        exclude_from_weight_decay=["batch_normalization", "bias"])
    elif args.fc_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.fc_lr, 
                                    weight_decay=args.fc_weight_decay, momentum=args.fc_momentum)
    else:
        print("no defined optimizer")
        assert False

    if args.fc_scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=[args.fc_epochs * len(train_loader) * 10, ], 
                                                         gamma=1)
    elif args.fc_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.fc_epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.fc_lr,
                                                    warmup_steps=0))
    elif args.fc_scheduler == None:
        scheduler = None
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False
    return optimizer, scheduler