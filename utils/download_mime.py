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


taskwise_links = {
    '1': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AAAwr0AtY9aWTq8VRQF_nsq9a/MIME/1?dl=1',
    '2': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AABbKKUuuv6zme7h5_Z3PqVoa/MIME/2?dl=1',
    '3': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AABqgnusXBrIw9sMCKXnJWv6a/MIME/3?dl=1',
    '4': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AADjIEV3mTZTMzSG4ELcVAt0a/MIME/4?dl=1',
    '5': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AAD5RTdCHitOjTz4Lqlxqwdla/MIME/5?dl=1',
    '6': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AABvRE9PxTUOqh8UY7gTeolKa/MIME/6?dl=1',
    '7': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AAB84v7qmquGPDA0bTXZJ3G3a/MIME/7?dl=1',
    '8': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AACDKPBIgcjv3OQT_QxwH0MKa/MIME/8?dl=1',
    '9': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AABR36f9fvsGMu-OJxogHqnZa/MIME/9?dl=1',
    '10': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AACv1a0AVxluL9N38WA6qnrqa/MIME/10?dl=1',
    '11': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AACKrGba6uflPrYaFPIwqEoQa/MIME/11?dl=1',
    '12': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AAB1Y1HOSwG86INylMQYT07Ca/MIME/12?dl=1',
    '13': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AABQnSsylQw4dqPDco0oH1bAa/MIME/13?dl=1',
    '14': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AADHN516N3mL9SL3ufYYy714a/MIME/14?dl=1',
    '15': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AADKm-S9jvusGzvcjyyWR08ka/MIME/15?dl=1',
    '16': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AAApBPL4YHjxj1QgkQowF5Bsa/MIME/16?dl=1',
    '17': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AAC-NzV-sWbF6YUNmPnJKYgDa/MIME/17?dl=1',
    '18': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AADIzBu_QJmQOEPG0lLt0jIla/MIME/18?dl=1',
    '19': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AABxdg1vRYfXqTIpz2_oD5_ka/MIME/19?dl=1',
    '20': 'https://www.dropbox.com/sh/hwgdkzbu488gwy4/AACpeMRd9CXkutvTsPSDOVK1a/MIME/20?dl=1',
}


download_cmd = 'curl --output {} -L {}'


def main(_):
    opts = flags.FLAGS
    for task_id in range(opts.task_id_min, opts.task_id_max+1):
        # task_dir = os.path.join(opts.MIME_dir, 'MIME_jointangles', str(task_id))

        task_save_dir = os.path.join(opts.MIME_videos_dir, str(task_id))
        task_file = os.path.join(opts.MIME_videos_dir, str(task_id) + '.zip')

        cmd_download = download_cmd.format(task_file, taskwise_links[str(task_id)])
        print(cmd_download)

        cmd_unzip = 'unzip {} -d {}'.format(task_file, task_save_dir)
        print(cmd_unzip)

        os.system(cmd_download)
        os.system(cmd_unzip)


if __name__ == '__main__':
    app.run(main)