from .resnet import resnet18, resnet10, resnet50, resnet101, resnet152

def load_model(args):
    # define model
    if args.train_mode == 'transfer' or args.train_mode =='supervised' or args.train_mode=='semi':
        dataset_name = 'imagenet'
        imagenet=True

    elif args.train_mode =='eval' or args.train_mode =='ssl':
        dataset_name = args.dataset_name
        if args.dataset_name == 'imagenet':
            imagenet=True
        else:
            imagenet=False


    if args.model == 'res18':
        model = resnet18(pretrained=False, dataset = dataset_name, imagenet=imagenet)
    elif args.model == 'res10':
        model = resnet10(pretrained=False, dataset = dataset_name, imagenet=imagenet)
    elif args.model == 'res50':
        model = resnet50(pretrained=False, dataset = dataset_name, imagenet=imagenet)
    elif args.model == 'res101':
        model = resnet101(pretrained=False, dataset = dataset_name, imagenet=imagenet)
    elif args.model == 'res152':
        model = resnet152(pretrained=False, dataset = dataset_name, imagenet=imagenet)
    else:
        assert False
        
    return model
