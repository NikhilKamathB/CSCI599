##########################################################################################################
# Assignment 2: Structure From Motion
# How to run this code:
#     - Open a terminal and navigate to the directory containing this file.
#     - Run the following command: `python assignment2.py`
##########################################################################################################

# Set system path to include the parent directory
import os
import cv2
import sys
import argparse
sys.path.append('..')

# Import the required modules
import src
from datetime import datetime


def main(args: argparse.Namespace) -> None:
    """
    Main function for Assignment 2.
    :param args: argparse.Namespace, the command line arguments
    :return: None
    """
    # Feature matching algorithm
    datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    image_matches, feat_dir, matches_dir, verbose_image_dir = src.FeatureExtractor(
        extractor=args.extractor,
        matcher=args.matcher,
        image_dir=args.image_dir,
        image_ext=args.ext.split(','),
        out_dir=args.out_dir,
        norm_type=args.norm_type,
        cross_check=args.cross_check,
        folder_suffix=datetime_str,
        verbose=args.verbose,
        verbose_match_percentage=args.verbose_match_percentage
    ).run()
    # Structure from Motion
    _ = src.SFM(
        image_dir=args.image_dir,
        features_dir=feat_dir,
        matches_dir=matches_dir,
        out_dir=args.out_dir,
        folder_suffix=datetime_str,
        image_matches=image_matches,
        image_ext=args.ext.split(','),
        matcher=args.matcher,
        norm_type=args.norm_type,
        cross_check=args.cross_check,
        fundamental_matrix_estimation_method=args.fundamental_matrix_estimation_method,
        calibration_matrix_style=args.calibration_matrix_style,
        pnp_method=args.pnp_method,
        pnp_estimation_confidence=args.pnp_estimation_confidence,
        reprojection_threshold=args.reprojection_threshold,
        verbose=args.verbose
    ).run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Implementation of SFM.")
    # Data
    parser.add_argument("-i", "--image_dir", type=str, default="../assets/Benchmarking_Camera_Calibration_2008/fountain-P11/images", help="Path to the directory containing images")
    parser.add_argument("-ext", "--ext", type=str, default="jpg,png,jpeg", help="Comma separated string of allowed image extensions")
    parser.add_argument("-o", "--out_dir", type=str, default="../assets/results/assignment2", help="Directory to store results in")
    # Feature extraction
    parser.add_argument("-e", "--extractor", type=str, default="SIFT", help="Feature extractor to use")
    parser.add_argument("-m", "--matcher", type=str, default="BFMatcher", help="Feature matcher to use")
    parser.add_argument("-n", "--norm_type", type=int, default=cv2.NORM_L2, help="Norm type to use") # cv2.NORM_L2
    parser.add_argument("-c", "--cross_check", type=src.str2bool, default='y', help="Whether to cross check feature matching or not")
    # SFM
    parser.add_argument("-fm", "--fundamental_matrix_estimation_method", type=int, default=cv2.FM_RANSAC, help="Method to estimate the fundamental matrix")
    parser.add_argument("-pm", "--pnp_method", type=int, default=cv2.SOLVEPNP_DLS, help="Method to estimate the pose of the camera using PnP algorithm")
    parser.add_argument("-cm", "--calibration_matrix_style", type=str, default="benchmark", help="Style of the calibration matrix")
    parser.add_argument("-ot", "--outlier_threshold", type=float, default=0.9, help="Threshold for outliers in the fundamental matrix estimation")
    parser.add_argument("-fc", "--fundamental_matrix_estimation_confidence", type=float, default=0.9, help="Confidence for fundamental matrix estimation")
    parser.add_argument("-pc", "--pnp_estimation_confidence", type=float, default=0.99, help="Confidence for pose estimation using PnP algorithm")
    parser.add_argument("-rt", "--reprojection_threshold", type=float, default=8.0, help="Reprojection threshold for pose estimation")
    # Misc
    parser.add_argument("-v", "--verbose", type=src.str2bool, default='y', help="Verbosity")
    parser.add_argument("-vm", "--verbose_match_percentage", type=float, default=0.1, help="Percentage of matches to display")
    args = parser.parse_args()

    # Make the output directory if it does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    ############################################################
    ############ Assignment 2. SFM implementation ##############
    ############################################################

    try:
        main(args)
    except Exception as e:
        raise e
    

    ############################################################
