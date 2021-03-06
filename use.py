import imageio
import numpy as np
import argparse

from skimage.transform import resize
from skimage import img_as_ubyte

import warnings
warnings.filterwarnings("ignore")

from demo import load_checkpoints
from demo import make_animation

parser = argparse.ArgumentParser(description='Deep fake.')
parser.add_argument('--img', type=str, default='./cartoons-01.png')
parser.add_argument('--video', type=str, default='./00.mp4')
parser.add_argument('--config', type=str, default='config/vox-256.yaml')
parser.add_argument('--model', type=str, default='./vox-adv-cpk.pth.tar')
parser.add_argument('--output', type=str, default='../generated.mp4')

args = parser.parse_args()

source_image = imageio.imread(args.img)
reader = imageio.get_reader(args.video)


#Resize image and video to 256x256
source_image = resize(source_image, (256, 256))[..., :3]

fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()

driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

generator, kp_detector = load_checkpoints(config_path=args.config, 
                            checkpoint_path=args.model)

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave(args.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)
