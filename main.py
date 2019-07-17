# Goal - for a given file selection, output labels and probabilities in a .h5 with the same name in file directory

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from torch.autograd import Variable
import h5py 
from tqdm import tqdm
import os

from libs.models import PSPNet
from libs.utils import dense_crf

def process_image(model, config, cuda, image_size, image_path): 
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
    image = cv2.resize(image, image_size)
    image_original = image.astype(np.uint8)
    image = image[..., ::-1] - np.array(
        [config.IMAGE.MEAN.R, config.IMAGE.MEAN.G, config.IMAGE.MEAN.B]
    )
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.cuda() if cuda else image

    # Inference
    output = model(Variable(image, volatile=True))

    output = F.upsample(output, size=image_size, mode="bilinear")
    output = F.softmax(output, dim=1)
    output = output[0].cpu().data.numpy()

    output = dense_crf(image_original, output)
    
    # At this point output is label - x - y
    prob_map = output.transpose(1, 2, 0)
    label_map = np.argmax(prob_map, axis=2)
    return prob_map, label_map

@click.command()
@click.option("--image-file", "-i", default="images.txt")
@click.option("--config", "-c", default="config/ade20k.yaml")
@click.option("--output-dir", "-o", default="output")
@click.option("--cuda/--no-cuda", default=True)
def main(config, image_file, output_dir, cuda):
    CONFIG = Dict(yaml.load(open(config)))

    cuda = cuda and torch.cuda.is_available()

    # Label list
    with open(CONFIG.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]

    # Load a model
    state_dict = torch.load(CONFIG.PYTORCH_MODEL)

    # Model
    model = PSPNet(
        n_classes=CONFIG.N_CLASSES, n_blocks=CONFIG.N_BLOCKS, pyramids=CONFIG.PYRAMIDS
    )
    model.load_state_dict(state_dict)
    model.eval()
    if cuda:
        model.cuda()

    image_size = (CONFIG.IMAGE.SIZE.TEST,) * 2

    os.makedirs(output_dir, exist_ok=True)

    # Image preprocessing
    with open(image_file) as f:
        for image_path in f:
            path = image_path.strip()
            image_name = (path.split('/')[-1]).split('.')[0]
            segment_file = h5py.File("{}/{}.h5".format(output_dir, image_name), "w")
            prob_map, label_map = process_image(model, CONFIG, cuda, image_size, path)
            segment_file.create_dataset('prob_map', data=prob_map, dtype='f') # We do not need all this precision but
            segment_file.create_dataset('label_map', data=label_map, dtype='i')
            segment_file.close()

if __name__ == "__main__":
    main()
