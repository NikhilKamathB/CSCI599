##########################################################################################################
# Any feature extraction angorithms should be implemented here.
# Feature extraction algorithms include:
# Feature extractor
#  - SIFT
# Feature matcher
#  - BFMatcher
##########################################################################################################

import os
import cv2
import time
import pickle
import logging
from datetime import datetime
from typing import List, Tuple
from src.utils.utils import serialize_keypoints, serialize_matches

logger = logging.getLogger(__name__)


class FeatureExtractor:

    """
        Class to extract features from images.
    """

    __LOG_PREFIX__ = "FeatureExtractor()"

    def __init__(self, 
                 extractor: str = "sift", 
                 matcher: str = "bfmatcher", 
                 image_dir: str = "../assets", 
                 image_ext: List[str] = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"],
                 out_dir: str = "../assets",
                 norm_type: int = cv2.NORM_L2,
                 cross_check: bool = True,
                 verbose: bool = True,
                 verbose_match_percentage: float = 0.25
                ) -> None:
        """
            Initialize the feature extractor object.
            Input parameters:
                - extractor: feature extractor to use
                - matcher: feature matcher to use
                - image_dir: directory containing images
                - image_ext: allowed image extensions
                - out_dir: directory to store results/data in
                - norm_type: norm type to use
                - cross_check: whether to cross check feature matching or not
                - verbose: verbosity
                - verbose_match_percentage: percentage of matches to display
        """
        self.images = [os.path.join(image_dir, x) for x in sorted(
            os.listdir(image_dir)) if x.split('.')[-1] in image_ext]
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.verbose = verbose
        self.verbose_match_percentage = verbose_match_percentage
        datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.out_feats_dir = os.path.join(out_dir, "feats", f"{datetime_str}")
        self.out_matches_dir = os.path.join(out_dir, "matches", f"{datetime_str}")
        self.out_verbose_dir = os.path.join(out_dir, "verbose", f"{datetime_str}")
        os.makedirs(self.out_verbose_dir, exist_ok=True)
        self._initialize_extractor(extractor); self._initialize_matcher(matcher)
    
    def _initialize_extractor(self, extractor: str = "sift") -> None:
        """
            Initialize the feature extractor.
            Input parameters:
                - extractor: feature extractor to use
        """
        if extractor == "sift" or extractor == "SIFT":
            self.extractor = cv2.SIFT_create()
            os.makedirs(self.out_feats_dir, exist_ok=True)
        else:
            logger.error(f"{self.__LOG_PREFIX__}: Invalid feature extractor")
            raise ValueError("Invalid feature extractor")
    
    def _initialize_matcher(self, matcher: str = "bfmatcher") -> None:
        """
            Initialize the feature matcher.
            Input parameters:
                - matcher: feature matcher to use
        """
        if matcher == "bfmatcher" or matcher == "BFMatcher" or matcher == "BFMATCHER":
            self.matcher = cv2.BFMatcher(normType=self.norm_type,
                crossCheck=self.cross_check)
            os.makedirs(self.out_matches_dir, exist_ok=True)
        else:
            logger.error(f"{self.__LOG_PREFIX__}: Invalid feature matcher")
            raise ValueError("Invalid feature matcher")
    
    def extract_features(self) -> List:
        """
            Extract features from the images.
            Returns: List of tuples containing image path, image name, keypoints, and descriptors
        """
        data = []
        for _, img in enumerate(self.images):
            start_time = time.time()
            image = cv2.imread(img)[:, :, ::-1] # BGR to RGB
            image_name = img.split('/')[-1].split('.')[0]
            keypoints, descriptors = self.extractor.detectAndCompute(image, None) # Second argument is mask - if you want to search only a part of image
            keypoints_serialized = serialize_keypoints(keypoints)
            with open(os.path.join(self.out_feats_dir, f"keypoints_{image_name}.pkl"), "wb") as f:
                pickle.dump(keypoints_serialized, f)
            with open(os.path.join(self.out_feats_dir, f"descriptors_{image_name}.pkl"), "wb") as f:
                pickle.dump(descriptors, f)
            if self.verbose:
                out_img = cv2.drawKeypoints(cv2.cvtColor(
                    image, cv2.COLOR_RGB2GRAY), keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imwrite(os.path.join(self.out_verbose_dir, f"verbose_{image_name}.jpg"), out_img)
            data.append(
                (img, image_name, keypoints, descriptors)
            )
            logger.info(f"{self.__LOG_PREFIX__}: Features extracted from image {img} - time taken: {time.time() - start_time}s")
        return data
    
    def match_features(self, data: List) -> None:
        """
            Match features from the data.
            Input parameters:
                - data: data list containing image path, image name, keypoints, and descriptors
        """
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                start_time = time.time()
                img1, image_name1, keypoints1, descriptors1 = data[i]
                img2, image_name2, keypoints2, descriptors2 = data[j]
                matches = self.matcher.match(descriptors1, descriptors2)
                matches = sorted(matches, key=lambda x: x.distance)
                matches_serialized = serialize_matches(matches)
                with open(os.path.join(self.out_matches_dir, f"matches_{image_name1}_{image_name2}.pkl"), "wb") as f:
                    pickle.dump(matches_serialized, f)
                if self.verbose:
                    out_img = cv2.drawMatches(
                        cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY),  
                        keypoints1, 
                        cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY),
                        keypoints2, 
                        matches[:int(len(matches) * self.verbose_match_percentage)],
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    cv2.imwrite(os.path.join(self.out_verbose_dir, f"verbose_{image_name1}_{image_name2}.jpg"), out_img)
                logger.info(f"{self.__LOG_PREFIX__}: Features matched between images {img1} and {img2} - time taken: {time.time() - start_time}s")

    def run(self) -> Tuple:
        """
            Run the feature extraction and matching.
            Returns: Tuple containing the paths to the extracted features and matches, also verbose images
        """
        try:
            data = self.extract_features()
            self.match_features(data)
            logger.info(f"{self.__LOG_PREFIX__}: Feature extraction and matching complete")
            return self.out_feats_dir, self.out_matches_dir, self.out_verbose_dir
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while extracting features")
            raise e
                