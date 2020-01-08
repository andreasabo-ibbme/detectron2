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
import shutil

import configparser

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# These can be overridden by command line arguments
WINDOW_NAME = "COCO detections"
OUTPUT_VIDEO = False
OUTPUT_BASE = "output"
Belmont_vids = False # If this is true, use the input video as the output foldername
FORCE_VERTICAL = False 


DEFAULT_INPUT_FILE = "/home/researchuser/Desktop/data_to_process/process_list_avi.csv"


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
                         default = DEFAULT_INPUT_FILE,
                         help="A CSV with a list of input videos and corresponding output filenames")

    parser.add_argument("--config-file-custom",
                        dest='custom_configs',
                         metavar="FILE")

    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=['MODEL.WEIGHTS', 'models/model_final_997cc7.pkl'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument('--output_video', dest='output_video', default=OUTPUT_VIDEO, action='store_true')
    parser.add_argument('--belmont', dest='is_belmont', default=Belmont_vids, action='store_true')
    parser.add_argument('--force_vert', dest='force_vertical', default=FORCE_VERTICAL, action='store_true')

    return parser

# https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
def copytree(src, dst, symlinks=False, ignore=None):

    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def saveUpdatedListOfVideosToProcess(list_of_vids_to_process, save_file_name):
    if len(list_of_vids_to_process) is 0 and os.path.exists(list_of_vids_to_process):
        os.remove(list_of_vids_to_process)
    else:
        with open(save_file_name, 'w') as f:
             for fn in list_of_vids_to_process:
                f.write("%s," % fn)
    


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()


    if args.custom_configs:
        config = configparser.ConfigParser()
        config.read(args.custom_configs)
        args.belmont = config.get('general', 'belmont_vids')
        args.force_vert = config.get('general', 'force_vertical')
        args.output_video = config.get('detectron', 'output_video')

    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    print(args)

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, force_vert=FORCE_VERTICAL)

    # Parse custom arguments
    OUTPUT_VIDEO = args.output_video
    Belmont_vids = args.belmont
    FORCE_VERTICAL = args.force_vertical

    print(f'OUTPUT_VIDEO: %s' % OUTPUT_VIDEO)



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
                head1, tail1 = os.path.split(base) # filename
                head2, tail2 = os.path.split(head1) # walk folder
                head3, tail3 = os.path.split(head2) # participant folder

                vid_folder = os.path.join(tail3, tail2)

                DEFAULT_OUTPUT_DIRECTORY = head3
                # print ("non-belmont save location: " + DEFAULT_OUTPUT_DIRECTORY)
                # print("non-belmont vid name: " + vid_folder)



                if Belmont_vids:
                    vid_folder = os.path.join(tail2, tail1)
                    # print("in belmont vids")
                    DEFAULT_OUTPUT_DIRECTORY = head2

                # print("output save location++++++++++++++++++++++++++++++++")
                # print(DEFAULT_OUTPUT_DIRECTORY)
                # print(vid_folder)

                outpath_base = os.path.join(OUTPUT_BASE, vid_folder)
                print('output base: ' + outpath_base)
                out_txt = os.path.join(outpath_base, 'output_detectron.txt')
                out_img = os.path.join(outpath_base, 'predimg50_detectron.jpg')
                out_img_raw = os.path.join(outpath_base, 'predimg50.jpg')
                log_file = os.path.join(OUTPUT_BASE, logfile + '.txt')
                out_vid = os.path.join(outpath_base, 'detectron_video.mp4')



                # print(output_fname)
                if OUTPUT_VIDEO:
                    video_for_vid_process = cv2.VideoCapture(input_file)

                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    fourcc = cv2.VideoWriter_fourcc(*"MPEG")

                    if FORCE_VERTICAL: # this is important so we can save it correctly when making the output video
                        if width > height:
                            temp = width
                            width = height
                            height = temp

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

                    print("saving video to: " + output_fname + '\n')
                    # for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
                    for vis_frame in tqdm.tqdm(demo.run_on_video(video_for_vid_process), total=num_frames):
                        output_file.write(vis_frame)

                
                # end if OUTPUT_VIDEO

                print("starting predictions on: %s\n" % input_file)
                #a = demo.run_on_video(video)
                predictions, predimg, predimg_raw = demo.predictions_from_video(video)
                print(len(predictions))
                # print(predictions)
                # If we don't have any predicitons, check the file format
                # if len(predictions) == 0:
                #     continue;
                    # reformatVideo(input_file)



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
                print("processed_walks: ") 
                print(processed_walks)
                print("input_file:" + input_file)
                print("out_txt file:" + out_txt)


                # Copy over the contents from the temp file to the final destination folder


                destination_folder = os.path.join(DEFAULT_OUTPUT_DIRECTORY, vid_folder)
                print("copying from {0} to {1}" , (outpath_base, destination_folder))
                copytree(outpath_base, destination_folder, symlinks=False, ignore=None)




                # Save into the log file
                with open(log_file, 'w') as f:
                    for item in processed_walks:
                        f.write("%s, 0, Success\n" % item)



            except Exception as e:
                print("CAUGHT EXCEPTION: " + str(e))
                # Save into the log file
                with open(log_file, 'a') as f:
                    f.write("%s, -1, %s\n" % (input_file, e))

            saveUpdatedListOfVideosToProcess(input_files[i+1:], args.input)


            # video.release()
            # output_file.release()


            # print(width, height, num_frames, basename)


    print("DONE- andrea")

