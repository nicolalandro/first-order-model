import imageio
import numpy as np

from skimage.transform import resize
from skimage import img_as_ubyte

import warnings
warnings.filterwarnings("ignore")

from demo import load_checkpoints
from demo import make_animation


source_image = imageio.imread('./cartoons-01.png')
reader = imageio.get_reader('./00.mp4')


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

generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                            checkpoint_path='./vox-adv-cpk.pth.tar')

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave('../generated.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
