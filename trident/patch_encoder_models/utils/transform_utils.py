from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1, center_crop = False, **resize_kwargs):
    trsforms = []
    
    if target_img_size > 0:
        trsforms.append(transforms.Resize((target_img_size, target_img_size), **resize_kwargs))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))
        
    
    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms