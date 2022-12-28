import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, bboxes, ax):
    """Draw detected bounding boxes."""
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i,:]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    


def demo(sess, net, image_name, path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_txt = os.path.join(path,image_name,'diff.txt')
    f = open(im_txt)
    lines = f.readlines()
    ids = []
    for line in lines:
        a = line.strip().split()
        if not len(a) == 5:
            ids.append(a[0])

    im1 = cv2.imread(os.path.join(path,image_name,ids[0]))
    im2 = cv2.imread(os.path.join(path,image_name,ids[1]))

    fig = plt.figure(figsize=(12, 12))
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    boxes, score = im_detect(sess, net, im1, im2, image_name)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im2[:, :, (2, 1, 0)]
    ax = fig.add_subplot(1,1,1)
    ax.imshow(im, aspect='equal')

    #score = score.reshape(-1,1)  

    #sort_id = score.argsort()[::-1]
    #last = 20
    #sort_last = sort_id[:last]
    sort_last = np.where(score > 1.5)[0]
    box = boxes[sort_last,0:4]
    sco = score[sort_last]
    sco = sco.reshape(-1,1)

    #print box.shape, sco.shape

    box_sco = np.hstack([box,sco])
    keep = nms(box_sco,0.4)
    box_sco = box_sco[keep]

    box_ = box_sco[:,0:4]


    vis_detections(im, box_, ax)
    save_path = os.path.join('/home/a409/users/liqing/diff_v3/result/',image_name+'.png')
    fig.savefig(save_path)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='Resnet50_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default='/home/a409/users/liqing/diff_v3/output/faster_rcnn_voc_vgg/VEDAI_1024_for_HPC')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ' or not os.path.exists(args.model):
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    saver = tf.train.Saver()
    model_file = tf.train.latest_checkpoint(args.model)
    saver.restore(sess, model_file )
    print (' done.')

    # # Warmup on a dummy image
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    # for i in xrange(2):
    #     _, _ = im_detect(sess, net, im)
    #
    # im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #            glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))

    path = '/home/a409/users/liqing/diff_pair/test/'

    f = open(os.path.join(path,'test.txt'))
    image = f.readlines()
    im_names = [x.strip() for x in image]

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name, path)
