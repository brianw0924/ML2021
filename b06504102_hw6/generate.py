import argparse
import math
import os
import torch
from torchvision import utils
import random
from model import StyledGenerator

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2021)

@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None
    x = 10
    for i in range(x):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= x
    return mean_style

@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.25,
    )
    
    return image

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 3, shape, shape).to(device) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64, help='size of the image')
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('--path', type=str,default="./checkpoint", help='path to checkpoint file')
    
    args = parser.parse_args()
    
    device = 'cuda'

    s = 21000

    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(f'{args.path}/0{str(s)}.model'))
    generator.eval()

    mean_style = get_mean_style(generator, device)

    step = int(math.log(args.size, 2)) - 2
    

    # img = style_mixing(generator, step, mean_style, 4, 10, device) # n 張臉, m 組 的 n+1 風格??
    # print(img.size())
    # utils.save_image(
    #     img, f'mix.jpg', normalize=True, range=(-1, 1)
    # )
    # for j in range(img.size(0)-1):
    #     utils.save_image(
    #         img[j+1], f'./output/{j+54*19+1}.jpg', normalize=True, range=(-1, 1)
    #     )

    # # Mixing
    # # n + m*(n+1)
    # img = style_mixing(generator, step, mean_style, 1, 25, device) # n 張臉, m 組 的 n+1 風格??
    # print(img.size())
    # utils.save_image(
    #     img, f'mix.jpg', normalize=True, range=(-1, 1)
    # )

    # # Sample
    # img = sample(generator, step, mean_style, 50, device)
    # utils.save_image(img, f'sample_{str(s)}.jpg', nrow=args.n_col, normalize=True, range=(-1, 1))
    
    # Sample output
    output_file = "output"
    os.makedirs("./output", exist_ok=True)
    for i in range(1000):
        img = sample(generator, step, mean_style, 1, device)
        # print(img.size())
        utils.save_image(img[0], f'./output/{i+1}.jpg', normalize=True, range=(-1, 1))
    os.system(f'cd {output_file} && tar -zcf ../images.tgz *.jpg')


    # # Mixing output
    # for i in range(40):
    #     img = style_mixing(generator, step, mean_style, 1, 12, device) # n 張臉, m 組 的 n+1 風格??
    #     print(img.size())
    #     for j in range(img.size(0)-1):
    #         utils.save_image(
    #             img[j+1], f'./output/{i*25+j+1}.jpg', normalize=True, range=(-1, 1)
    #         )