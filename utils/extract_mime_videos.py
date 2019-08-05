from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import numpy as np
from absl import flags, app
import pdb
import scipy.misc
import subprocess


flags.DEFINE_string('MIME_videos_dir', '/checkpoint/shubhtuls/data/MIME/', 'Data Directory')
flags.DEFINE_integer('task_id_min', 1, 'lowest task id to download')
flags.DEFINE_integer('task_id_max', 20, 'max task id to download')


extraction_cmd = 'rm -rf {0}/frames; mkdir {0}/frames; ffmpeg  -loglevel panic -i {0}/rd_kinect_rgb.mp4 -vf fps=1 {0}/frames/im_%d.png'


def main(_):
    opts = flags.FLAGS
    for task_id in range(opts.task_id_min, opts.task_id_max+1):
        task_name = str(task_id)
        task_dir = os.path.join(opts.MIME_videos_dir, str(task_name))
        demo_names = next(os.walk(task_dir))[1]

        for dx, demo in enumerate(demo_names):
            if dx % 10 == 0:
                print('task {}, demo: {}/{}'.format(task_id, dx, len(demo_names)))
            cmd = extraction_cmd.format(os.path.join(task_dir, demo))
            os.system(cmd)
            # pdb.set_trace()


if __name__ == '__main__':
    app.run(main)