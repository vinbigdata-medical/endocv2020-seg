import os


for bb, segm in zip(["b4", "b3", "r50"],
    ["unet", "unet", "fpn"]):
    for fold in range(5):
        cmd = f'python main.py --config cvcore/configs/{bb}_f{fold}.yaml'
        if bb == 'r50':
            cmd += f' --load weights/best_resnet50_{segm}_fold{fold}.pth'
        else:
            cmd += f' --load weights/best_{bb}_{segm}_fold{fold}.pth'
        cmd += f' --valid'
        # print(f'Run command {cmd}')
        os.system(cmd)
        # print(f'Run command {cmd}')
        # cmd = cmd.replace('valid', 'test')
        # os.system(cmd)