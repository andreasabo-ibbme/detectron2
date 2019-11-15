# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified Oct 22, 2019 - Andrea

import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import csv
import datetime
import scipy.misc
import matplotlib
import ffmpeg
import shlex, subprocess

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"
OUTPUT_VIDEO = True
OUTPUT_BASE = "output"
Belmont_vids = True # If this is true, use the input video as the output foldername


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg




def get_parser():
    parser = argparse.ArgumentParser(description="Ambient Data Processor")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input",
                         metavar="FILE", 
                         help="A CSV with a list of input videos and corresponding output filenames")
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    if args.input:
        input_files, output_files = [], []
        logfile = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        processed_walks = []

        # Load all of the files we need to process into memory to avoid keeping the file open
        with open(args.input) as csv_file:
            csv_data = csv.reader(csv_file, delimiter=',')
            
            for file in csv_data:
                # print(file)
                input_files.extend(file)
                # output_files.append(file[1])

        # Now process each of the files separately
        num_files = len(input_files)
        for i in range(num_files):
            input_file = input_files[i]
            if (len(input_file) == 0):
                print('empty file')
                continue
            try:

                # output_fname = str(output_files[i])

                video = cv2.VideoCapture(input_file)
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = video.get(cv2.CAP_PROP_FPS)
                num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                basename = os.path.basename(input_file)


                base, ext = os.path.splitext(input_file)
                head, tail1 = os.path.split(base) # filename
                head, tail2 = os.path.split(head) # walk folder
                head, tail3 = os.path.split(head) # participant folder
                outpath_base = os.path.join(OUTPUT_BASE, tail3, tail2)

                if Belmont_vids:
                    outpath_base = os.path.join(OUTPUT_BASE, tail2, tail1)
                    print('output base: ' + outpath_base)
                out_txt = os.path.join(outpath_base, 'output_detectron.txt')
                out_img = os.path.join(outpath_base, 'predimg50_detectron.jpg')
                out_img_raw = os.path.join(outpath_base, 'predimg50.jpg')
                log_file = os.path.join(OUTPUT_BASE, logfile + '.txt')
                out_vid = os.path.join(outpath_base, 'detectron_video.mp4')


                # print(output_fname)
                if OUTPUT_VIDEO:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    fourcc = cv2.VideoWriter_fourcc(*"MPEG")

                    
                    output_fname = out_vid
                    output_file = cv2.VideoWriter(
                        filename=output_fname,
                        # some installation of opencv may not support x264 (due to its license),
                        # you can try other format (e.g. MPEG)
                        # fourcc=cv2.VideoWriter_fourcc(*"x264"),
                        #fourcc=cv2.VideoWriter_fourcc('F','M','P','4'),
                        fourcc = fourcc,
                        fps=float(frames_per_second),
                        frameSize=(width, height),
                        isColor=True,
                    )

                    directory = os.path.dirname(output_fname)
                    print(directory)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                if OUTPUT_VIDEO:
                    # for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                    for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                        output_file.write(vis_frame)

                
                print("starting predictions on: %s\n" % input_file)
                #a = demo.run_on_video(video)
                predictions, predimg, predimg_raw = demo.predictions_from_video(video)
                
                # If we don't have any predicitons, check the file format
                if len(predictions) == 0:
                    reformatVideo(input_file)



                # Make output directory if it doesnt already exist
                directory = os.path.dirname(out_txt)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # Save predictions to file
                with open(out_txt, 'w') as f:
                    for item in predictions:
                        f.write("%s\n" % item)

                predimg.save(out_img)

                matplotlib.image.imsave(out_img_raw, predimg_raw)


                processed_walks.append(input_file)
                print(processed_walks)
                print(input_file)

                # Save into the log file
                with open(log_file, 'w') as f:
                    for item in processed_walks:
                        f.write("%s, 0, Success\n" % item)

            except Exception as e:

                # Save into the log file
                with open(log_file, 'a') as f:
                    f.write("%s, -1, %s\n" % (input_file, e))



            # video.release()
            # output_file.release()


            # print(width, height, num_frames, basename)


    print("DONE- andrea")

