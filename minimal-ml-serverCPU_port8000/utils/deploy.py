def load_model(opt, pth_name):
    def __patch_instance_norm_state_dict(state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            __patch_instance_norm_state_dict(state_dict, getattr(module, key),
                                             keys, i + 1)
    #load_path = path.join(opt.checkpoints_dir, opt.name, pth_name)
    # changed to pth_name ---> is senden as arg when called through Grasshopper
    load_path = str(pth_name)
    print(load_path)
    model = opt.netG
    print(model)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = load(load_path, map_location=str(opt.device))
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata
    for keys in list(state_dict.keys()):
        __patch_instance_norm_state_dict(state_dict, model, keys.split('.'))
    model.load_state_dict(state_dict)
    model.eval()
    return model