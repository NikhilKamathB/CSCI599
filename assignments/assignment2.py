##########################################################################################################
# Assignment 2: Structure From Motion
# How to run this code:
#     - Open a terminal and navigate to the directory containing this file.
#     - Run the following command: `python assignment2.py`
##########################################################################################################

# Set system path to include the parent directory
import os
import argparse
import sys
sys.path.append('..')

# Import the required modules
import src


def main(args: argparse.Namespace) -> None:
    """
    Main function for Assignment 2.
    :param args: argparse.Namespace, the command line arguments
    :return: None
    """
    # Feature matching algorithm
    feat_dir, matches_dir, verbose_image_dir = src.FeatureExtractor(
        extractor=args.extractor,
        matcher=args.matcher,
        image_dir=args.image_dir,
        image_ext=args.ext.split(','),
        out_dir=args.out_dir,
        norm_type=args.norm_type,
        cross_check=args.cross_check,
        verbose=args.verbose,
        verbose_match_percentage=args.verbose_match_percentage
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
    parser.add_argument("-n", "--norm_type", type=int, default=4, help="Norm type to use") # cv2.NORM_L2
    parser.add_argument("-c", "--cross_check", type=src.str2bool, default='y', help="Whether to cross check feature matching or not")
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
