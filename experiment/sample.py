import torch
from torch.utils.data import random_split
from model.baseline import predict_starting_position
from data.data_loader import ProgramDataset
from model.predictor import StatePredictor

from PIL import Image, ImageDraw, ImageFont

import sys

sys.path.append('.')

from karel.world import World
from dsl.parser import Parser

INDEX = 7

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
RANDOM_SEED = 42

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = ProgramDataset('data/experiment.npz')

    # Train / Val / Test split
    train_len = int(TRAIN_SPLIT * len(dataset))
    val_len = int(VAL_SPLIT * len(dataset))
    test_len = len(dataset) - train_len - val_len
    _, _, test_ds = random_split(dataset, 
        [train_len, val_len, test_len],
        torch.Generator().manual_seed(RANDOM_SEED)
    )

    model = StatePredictor().to(device)

    model.load_state_dict(torch.load('output/experiment_model.pth', map_location=device))

    for index in range(len(test_ds)):

        sample = test_ds.__getitem__(index)

        z, s_s, s_f, prog = (torch.unsqueeze(d.to(device), 0) for d in sample)

        model.eval()
        output = model(s_s, z)

        predicton_output = torch.argmax(output, dim=1)

        prediction_baseline = predict_starting_position(s_s)

        # print(s_f[0].cpu().numpy())
        # print(predicton_output[0].cpu().numpy())
        # print(prediction_baseline[0].cpu().numpy())

        world_s = torch.moveaxis(s_s[0], [-1,-2,-3], [-2, -3, -1])

        world = World(world_s.cpu().numpy())

        im = Image.fromarray(world.to_image()).convert('RGB')
        draw = ImageDraw.Draw(im)

        (s_f_x, s_f_y) = (s_f[0].cpu().numpy() % 8, s_f[0].cpu().numpy() // 8)
        (pred_x, pred_y) = (
            predicton_output[0].cpu().numpy() % 8,
            predicton_output[0].cpu().numpy() // 8
        )
        (base_x, base_y) = (
            prediction_baseline[0].cpu().numpy() // 8,
            prediction_baseline[0].cpu().numpy() % 8
        )

        draw.ellipse(
            (s_f_x * 100, s_f_y * 100, (s_f_x+1) * 100, (s_f_y+1) * 100),
            outline=(255,0,0),
            width=3
        )
        draw.ellipse(
            (pred_x * 100 + 3, pred_y * 100 + 3, (pred_x+1) * 100 - 3, (pred_y+1) * 100 - 3),
            outline=(0,255,0),
            width=3
        )
        draw.ellipse(
            (base_x * 100 + 6, base_y * 100 + 6, (base_x+1) * 100 - 6, (base_y+1) * 100 - 6),
            outline=(0,0,255),
            width=3
        )

        prog_list = prog[0].cpu().numpy().tolist()
        prog_str = Parser.list_to_tokens(prog_list).replace(' <pad>', '')
        
        if len(prog_str) > 80:
            prog_str = prog_str[:80] + '\n' + prog_str[80:]
        if len(prog_str) > 160:
            prog_str = prog_str[:160] + '\n' + prog_str[160:]

        font = ImageFont.truetype('NotoSans-VF.ttf', 18)
        
        draw.text((10,10), prog_str, (0,0,0), font=font)

        im.save(f'output/samples/{index}.png')
        print('a')

        # TODO: create World from s_s, print image, print predictions on top

        # TODO: parse prog
