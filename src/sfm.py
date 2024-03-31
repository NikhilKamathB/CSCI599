##########################################################################################################
# Structure from Motion implementation.
##########################################################################################################

import os
import cv2
import time
import copy
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import List
from scipy import sparse
from datetime import datetime
from scipy.optimize import least_squares, OptimizeResult
from src.utils.utils import CV2Mixin, deserialize_keypoints, deserialize_matches, pts2ply

logger = logging.getLogger(__name__)


class SFM(CV2Mixin):

    """
        Class to implement Structure from Motion.
    """

    __LOG_PREFIX__ = "SFM()"

    def __init__(self,
                image_dir: str = "../assets",
                features_dir: str = "../assets/results/assignment2/feats",
                matches_dir: str = "../assets/results/assignment2/matches",
                out_dir: str = "../assets",
                folder_suffix: str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                image_matches: list = [],
                image_ext: List[str] = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"],
                matcher: str = "bfmatcher",
                norm_type: int = cv2.NORM_L2,
                cross_check: bool = True,
                fundamental_matrix_estimation_method: int = cv2.FM_RANSAC,
                outlier_threshold: float = 0.9,
                fundamental_matrix_estimation_confidence: float = 0.9,
                calibration_matrix_style: str = "benchmark",
                pnp_method: int = cv2.SOLVEPNP_DLS,
                pnp_estimation_confidence: float = 0.99,
                reprojection_threshold: float = 8.0,
                reprojection_remove_outliers: bool = True,
                reprojection_remove_outliers_threshold: float = 5.0,
                perform_bundle_adjustment: bool = False,
                least_squares_method: str = "trf",
                cloud_point_cleaning: bool = True,
                cloud_point_z_threshold: float = 3.0,
                true_circle_radius: int = 15,
                true_circle_color: tuple = (0, 255, 0),
                true_circle_thickness: int = -1,
                projection_circle_radius: int = 5,
                projection_circle_color: tuple = (0, 0, 255),
                projection_circle_thickness: int = -1,
                connecting_line_color: tuple = (0, 255, 255),
                connecting_line_thickness: int = 2,
                verbose: bool = True,) -> None:
        """
            Initialize the SFM object.
            Input parameters:
                - image_dir: directory containing images
                - features_dir: directory containing features
                - matches_dir: directory containing matches
                - out_dir: directory to store results/data in
                - folder_suffix: suffix for the output directory
                - image_matches: list of image matches
                - image_ext: allowed image extensions
                - matcher: feature matcher to use
                - norm_type: norm type to use
                - cross_check: whether to cross check feature matching or not
                - fundamental_matrix_estimation_method: method to estimate the fundamental matrix
                - outlier_threshold: threshold for outliers
                - fundamental_matrix_estimation_confidence: confidence for fundamental matrix estimation
                - calibration_matrix_style: style of the calibration matrix
                - pnp_method: method to estimate the pose of the camera using PnP algorithm
                - pnp_estimation_confidence: confidence for pose estimation using PnP algorithm
                - reprojection_threshold: reprojection threshold for pose estimation
                - reprojection_remove_outliers: whether to remove outliers post PnP or not
                - reprojection_remove_outliers_threshold: threshold for removing outliers post PnP
                - perform_bundle_adjustment: whether to perform bundle adjustment or not
                - least_squares_method: method to use for least squares optimization
                - cloud_point_cleaning: whether to clean the cloud points or not before saving
                - cloud_point_z_threshold: threshold for z-axis of the cloud points
                - true_circle_radius: radius of the true circle
                - true_circle_color: color of the true circle
                - true_circle_thickness: thickness of the true circle
                - projection_circle_radius: radius of the projection circle
                - projection_circle_color: color of the projection circle
                - projection_circle_thickness: thickness of the projection circle
                - connecting_line_color: color of the connecting line
                - connecting_line_thickness: thickness of the connecting line
                - verbose: verbosity
        """
        self.image_dir = image_dir
        self.features_dir = features_dir
        self.matches_dir = matches_dir
        self.out_dir = out_dir
        self.folder_suffix = folder_suffix
        self.image_matches = image_matches
        self.fundamental_matrix_estimation_method = fundamental_matrix_estimation_method
        self.outlier_threshold = outlier_threshold
        self.matcher_type = matcher
        self.norm_type = norm_type
        self.cross_check = cross_check
        self.fundamental_matrix_estimation_confidence = fundamental_matrix_estimation_confidence
        self.calibration_matrix_style = calibration_matrix_style
        self.pnp_method = pnp_method
        self.pnp_estimation_confidence = pnp_estimation_confidence
        self.reprojection_threshold = reprojection_threshold
        self.reprojection_remove_outliers = reprojection_remove_outliers
        self.reprojection_remove_outliers_threshold = reprojection_remove_outliers_threshold
        self.perform_bundle_adjustment = perform_bundle_adjustment
        self.least_squares_method = least_squares_method
        self.cloud_point_cleaning = cloud_point_cleaning
        self.cloud_point_z_threshold = cloud_point_z_threshold
        self.true_circle_radius = true_circle_radius
        self.true_circle_color = true_circle_color
        self.true_circle_thickness = true_circle_thickness
        self.projection_circle_radius = projection_circle_radius
        self.projection_circle_color = projection_circle_color
        self.projection_circle_thickness = projection_circle_thickness
        self.connecting_line_color = connecting_line_color
        self.connecting_line_thickness = connecting_line_thickness
        self.verbose = verbose
        self.out_verbose_dir = os.path.join(out_dir, "verbose", f"{folder_suffix}")
        self.output_dir = os.path.join(out_dir, "cloudpoints", f"{folder_suffix}")
        self.image_data, self.matches_data = {}, {} # image_data -> view: [rotation, translation, references] | matches_data -> (query view, train view): [matches, img_1_coords, img_2_coords, img_1_idx, img_2_idx]
        self.point_cloud = np.zeros((0, 3))
        self.images = [os.path.join(image_dir, x) for x in sorted(
            os.listdir(image_dir)) if x.split('.')[-1] in image_ext]
        self._initialize_calibration_matrix()
        os.makedirs(self.out_verbose_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _initialize_calibration_matrix(self) -> None:
        """
            Initialize the calibration matrix.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the calibration matrix")
        if self.calibration_matrix_style == "benchmark" or self.calibration_matrix_style == "Benchmark" or self.calibration_matrix_style == "BENCHMARK":
            self.K = np.array([[2759.48, 0, 1520.69],
                               [0, 2764.16, 1006.81],
                               [0, 0, 1]])
        elif self.calibration_matrix_style == "lg_g3" or self.calibration_matrix_style == "LG_G3" or self.calibration_matrix_style == "LG G3":
            self.K = np.array([[3.97*320, 0, 320],
                               [0, 3.97*320, 240],
                               [0, 0, 1]])
        else:
            logger.error(f"{self.__LOG_PREFIX__}: Calibration matrix style not implemented")
            raise NotImplementedError("Calibration matrix style not implemented")
    
    def _load_pickle(self, file_dir: str, file_name: str, mute: bool = False) -> dict:
        """
            Load the pickle file.
            Input parameters:
                - file_dir: directory of the file
                - file_name: name of the file
                - mute: boolean indicating whether to mute the logs
            Output:
                - dictionary
        """
        if not mute:
            logger.info(f"{self.__LOG_PREFIX__}: Loading pickle file {file_name}")
        with open(os.path.join(file_dir, f"{file_name}.pkl"), 'rb') as file:
            return pickle.load(file)
    
    def _load_features_from_file(self, file_name: str, mute: bool = False) -> tuple:
        """
            Load the features with keypoints and descriptors.
            Input parameters:
                - file_name: name of the file
                - mute: boolean indicating whether to mute the logs
            Output:
                - keypoints: keypoints
                - descriptors: descriptors
        """
        if not mute:
            logger.info(
                f"{self.__LOG_PREFIX__}: Loading features for {file_name}")
        keypoints = self._load_pickle(
            self.features_dir, f"keypoints_{file_name}", mute=True)
        descriptors = self._load_pickle(
            self.features_dir, f"descriptors_{file_name}", mute=True)
        keypoints = deserialize_keypoints(keypoints)
        return keypoints, descriptors
    
    def _load_features(self, file_name: str) -> tuple:
        """
            Load the features with keypoints and descriptors.
            Input parameters:
                - file_name: name of the file
                - mute: boolean indicating whether to mute the logs
            Output:
                - keypoints: keypoints
                - descriptors: descriptors
        """
        try:
            return self.keypoints[file_name], self.descriptors[file_name]
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error loading features for {file_name}")
            raise e
    
    def _load_matches_from_file(self, file_name_1: str, file_name_2: str) -> list:
        """
            Load the matches.
            Input:
                - file_name_1: name of the first image
                - file_name_2: name of the second image
            Output:
                - matches: matches
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Loading matches for {file_name_1} and {file_name_2}")
        matches = self._load_pickle(
            self.matches_dir, f"matches_{file_name_1}_{file_name_2}")
        matches = deserialize_matches(matches)
        return matches
    
    def _load_matches(self, file_name_1: str, file_name_2: str) -> list:
        """
            Load the matches.
            Input:
                - file_name_1: name of the first image
                - file_name_2: name of the second image
            Output:
                - matches: matches
        """
        try:
            return self.matches_data[(file_name_1, file_name_2)]
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error loading matches for {file_name_1} and {file_name_2}")
            raise e
    
    def _load_data(self) -> tuple:
        """
            Load the data.
            Output:
                - a tuple of keypoints, descriptors, and matches
        """
        try:
            keypoints, descriptors, matches = {}, {}, {}
            for i in range(len(self.image_list)):
                keypoints[self.image_list[i]], descriptors[self.image_list[i]] = self._load_features_from_file(self.image_list[i])
                for j in range(i+1, len(self.image_list)):
                    matches[(self.image_list[i], self.image_list[j])] = self._load_matches_from_file(self.image_list[i], self.image_list[j])
            return keypoints, descriptors, matches
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error loading data")
            raise e
    
    def _get_image_list(self) -> List[str]:
        """
            Get the list of images that will be used to perform SFM
            Output:
                - list of images
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the list of images")
        image_list = []
        for image_1, image_2 in self.image_matches:
            if image_1 not in image_list:
                image_list.append(image_1)
            if image_2 not in image_list:
                image_list.append(image_2)
        image_list = sorted(image_list)
        return image_list
    
    def _get_aligned_matches(self, keypoints_1: list, keypoints_2: list, matches: list) -> tuple:
        """
            Get the aligned matches.
            Input parameters:
                - keypoints_1: keypoints of the first image
                - keypoints_2: keypoints of the second image
                - matches: matches
            Output:
                - img_1_coords: image 1 coordinates
                - img_2_coords: image 2 coordinates
                - img_1_idx: image 1 indices; 
                - img_2_idx: image 2 indices; matched
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting aligned matches")
        # Get the indices of the matches
        img_1_idx = np.array([match.queryIdx for match in matches])
        img_2_idx = np.array([match.trainIdx for match in matches])
        # Get the keypoints by filtering the indices based on the matches
        keypoints_1_ = np.array(keypoints_1)[img_1_idx]
        keypoints_2_ = np.array(keypoints_2)[img_2_idx]
        # Get image coordinates
        img_1_coords = np.array([keypoint.pt for keypoint in keypoints_1_])
        img_2_coords = np.array([keypoint.pt for keypoint in keypoints_2_])
        return img_1_coords, img_2_coords, img_1_idx, img_2_idx
    
    def _estimate_F(self, keypoints_1: list, keypoints_2: list, matches: list) -> tuple:
        """
            Estimate the fundamental matrix.
            Input parameters:
                - keypoints_1: keypoints of the first/query image
                - keypoints_2: keypoints of the second/train image
                - matches: matches
            Output:
                - F: fundamental matrix
                - mask: mask representing the inliers
                - image 1 coordinates
                - image 2 coordinates
                - image 1 indices
                - image 2 indices
        """
        logger.info(f"{self.__LOG_PREFIX__}: Estimating the fundamental matrix")
        # Get the aligned matches
        img_1_coords, img_2_coords, img_1_idx, img_2_idx = self._get_aligned_matches(keypoints_1, keypoints_2, matches)
        # Estimate the fundamental matrix - mask represents the inliers
        F, mask = cv2.findFundamentalMat(img_1_coords, img_2_coords, 
                                         method=self.fundamental_matrix_estimation_method,
                                         ransacReprojThreshold=self.outlier_threshold, 
                                         confidence=self.fundamental_matrix_estimation_confidence)
        mask = mask.astype(bool).flatten()
        return F, mask, img_1_coords, img_2_coords, img_1_idx, img_2_idx

    def _baseline_pose_estimation(self, image_1: str, image_2: str) -> tuple:
        """
            Estimate the baseline pose.
            Input parameters:
                - image_1: first image name
                - image_2: second image name
            Returns: tuple containing the rotation and translation matrices
        """
        logger.info(f"{self.__LOG_PREFIX__}: Estimating initial baseline pose for {image_1} and {image_2}")
        # Load the features and matches
        keypoints_1, _ = self._load_features(image_1)
        keypoints_2, _ = self._load_features(image_2)
        matches = self._load_matches(image_1, image_2)
        # Estimate F
        F, mask, img_1_coords, img_2_coords, img_1_idx, img_2_idx = self._estimate_F(keypoints_1, keypoints_2, matches)
        # Estimate the essential matrix
        E = self.K.T @ F @ self.K
        # Get the camera poses
        _, R, t, _ = cv2.recoverPose(E, img_1_coords[mask], img_2_coords[mask], self.K)
        # Record the data
        self.image_data[image_1] = [np.eye(3), np.zeros((3, 1)), np.ones((len(keypoints_1),)) * -1]
        self.image_data[image_2] = [R, t, np.ones((len(keypoints_2),)) * -1]
        self.matches_data[(image_1, image_2)] = [
            matches,
            img_1_coords[mask],
            img_2_coords[mask],
            img_1_idx[mask],
            img_2_idx[mask]
        ]
        return R, t

    def _pose_estimation(self, image: str) -> None:
        """
            Estimate the pose by finding 2D-3D correspondences between the new image and existing views
            Input parameters:
                - image: image name
        """
        logger.info(f"{self.__LOG_PREFIX__}: Estimating pose for {image}")
        # Get cv2 matcher
        matcher = self._initialize_matcher(self.matcher_type, make_dir=False, default=True)
        # Get keypoints and descriptors of the processed images
        img_list = []
        keypoints, descriptors = [], []
        for img in self.image_data.keys():
            img_list.append(img)
            keypoint, descriptor = self._load_features(img)
            keypoints.append(keypoint)
            descriptors.append(descriptor)
        # Add the new image to the matcher and train it
        matcher.add(descriptors)
        matcher.train()
        # Load keypoints and descriptors of the new image
        new_keypoint, new_descriptor = self._load_features(image)
        # Match the new image with the trained matcher
        new_matches = matcher.match(queryDescriptors=new_descriptor)
        # Get 2D and 3D correspondences
        points_3d, points_2d = np.zeros((0, 3)), np.zeros((0, 2))
        for match in new_matches:
            train_image_idx, descriptor_idx, new_image_idx = match.imgIdx, match.trainIdx, match.queryIdx
            point_cloud_idx = self.image_data[img_list[train_image_idx]][-1][descriptor_idx]
            if point_cloud_idx >= 0:
                new_3d_point = self.point_cloud[int(point_cloud_idx)][None, ...]
                points_3d = np.concatenate((points_3d, new_3d_point), axis=0)
                new_2d_point = np.array(new_keypoint[int(new_image_idx)].pt)[None, ...]
                points_2d = np.concatenate((points_2d, new_2d_point), axis=0)
        # Solve the PnP problem using RANSAC - mask represents the inliers
        _, R_vec, t, mask = cv2.solvePnPRansac(points_3d[:, None, :], points_2d[:, None, :], self.K, None,
                                        confidence=self.pnp_estimation_confidence, flags=self.pnp_method, reprojectionError=self.reprojection_threshold)
        R, _ = cv2.Rodrigues(R_vec)
        self.image_data[image] = [R, t, np.ones((len(new_keypoint),)) * -1]

    def _get_image_path(self, image_name: str) -> str:
        """
            Get the image file path.
            Input parameters:
                - image_name: name of the image
            Output:
                - image file path
        """
        file_path = image_name + ".jpg"
        for f in self.images:
            if image_name in f:
                file_path = f
                break
        return file_path

    def _get_colors(self, point_cloud: np.ndarray) -> np.ndarray:
        """
            Get the colors.
            Input parameters:
                - point_cloud: point cloud
            Output:
                - numpy array of colors
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting colors")
        colors = np.zeros_like(point_cloud)
        for k in self.image_data.keys():
            _, _, ref = self.image_data[k]
            keypoint, _ = self._load_features(k)
            keypoint = np.array(keypoint)[ref >= 0]
            image_points = np.array([kp.pt for kp in keypoint])
            file_path = self._get_image_path(k)
            image = cv2.imread(file_path)[:, :, ::-1]
            colors[ref[ref >= 0].astype(int)] = image[image_points[:, 1].astype(int), image_points[:, 0].astype(int)]
        return colors
    
    def _triangulate_points(self, image_1_coords: np.ndarray, image_2_coords: np.ndarray, r1: np.ndarray, t1: np.ndarray, r2: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
            Triangulate the points based on the image coordinates.
            Input parameters:
                - image_1_coords: image 1 coordinates
                - image_2_coords: image 2 coordinates
                - r1: rotation matrix 1 - camera 1
                - t1: translation matrix 1 - camera 1
                - r2: rotation matrix 2 - camera 2
                - t2: translation matrix 2 - camera 2
            Returns: numpy array of 3D points
        """
        logger.info(f"{self.__LOG_PREFIX__}: Triangulating points")
        image_1_points_homogeneous = cv2.convertPointsToHomogeneous(image_1_coords).squeeze()
        image_2_points_homogeneous = cv2.convertPointsToHomogeneous(image_2_coords).squeeze()
        image_1_points_norm = (np.linalg.inv(self.K) @ image_1_points_homogeneous.T).T
        image_2_points_norm = (np.linalg.inv(self.K) @ image_2_points_homogeneous.T).T
        image_1_points_norm = cv2.convertPointsFromHomogeneous(image_1_points_norm).squeeze()
        image_2_points_norm = cv2.convertPointsFromHomogeneous(image_2_points_norm).squeeze()
        points_4d = cv2.triangulatePoints(np.hstack((r1, t1)), np.hstack((r2, t2)), image_1_points_norm.T, image_2_points_norm.T)
        points_3d = cv2.convertPointsFromHomogeneous(points_4d.T).squeeze()
        return points_3d
    
    def _update_references(self, upper_limit: int, lower_limit: int, ref_1: np.ndarray, ref_2: np.ndarray, ref_1_mask: np.array, ref_2_mask: np.array) -> tuple:
        """
            Update the references of the images.
            Input parameters:
                - upper_limit: upper limit
                - lower_limit: lower limit
                - ref_1: reference 1 - image 1 - old view
                - ref_2: reference 2 - image 2 - new view
                - ref_1_mask: reference 1 mask - old view mask
                - ref_2_mask: reference 2 mask - new view mask
            Output:
                - updated reference 1
                - updated reference 2
        """
        ref_1[ref_1_mask] = np.arange(upper_limit) + lower_limit
        ref_2[ref_2_mask] = np.arange(upper_limit) + lower_limit
        return ref_1, ref_2

    def _baseline_triangulation(self, image_1: str, image_2: str) -> None:
        """
            Triangulate the baseline.
            Input parameters:
                - image_1: first image name
                - image_2: second image name
        """
        logger.info(f"{self.__LOG_PREFIX__}: Triangulating the baseline for {image_1} and {image_2}")
        # Get the data
        r1, t1, ref1 = self.image_data[image_1]
        r2, t2, ref2 = self.image_data[image_2]
        _, img_1_coords, img_2_coords, img_1_idx, img_2_idx = self.matches_data[(image_1, image_2)]
        # Triangulate the points
        points_3d = self._triangulate_points(img_1_coords, img_2_coords, r1, t1, r2, t2)
        self.point_cloud = np.concatenate((self.point_cloud, points_3d), axis=0)
        # Update 3D reference points
        ref1, ref2 = self._update_references(points_3d.shape[0], self.point_cloud.shape[0]-points_3d.shape[0],
                                             ref1, ref2, img_1_idx, img_2_idx)
        self.image_data[image_1][-1] = ref1
        self.image_data[image_2][-1] = ref2

    def _triangulation(self, image: str) -> None:
        """
            Triangulate the points for the new view based on the existing views.
            Input parameters:
                - image: image name
        """
        logger.info(f"{self.__LOG_PREFIX__}: Triangulating points for {image}")
        for old_view in self.image_data.keys():
            if old_view != image:
                # Get view keypoints and descriptors along with the matches
                keypoints_1, _ = self._load_features(old_view)
                keypoints_2, _ = self._load_features(image)
                matches = self._load_matches(old_view, image)
                # Filter the matches
                matches = [match for match in matches if self.image_data[old_view][-1][match.queryIdx] < 0]
                # Estimate F
                F, mask, img_1_coords, img_2_coords, img_1_idx, img_2_idx = self._estimate_F(
                    keypoints_1, keypoints_2, matches)
                # Record data
                self.matches_data[(old_view, image)] = [
                    matches,
                    img_1_coords[mask],
                    img_2_coords[mask],
                    img_1_idx[mask],
                    img_2_idx[mask]
                ]
                # Triangulate the points
                self._baseline_triangulation(old_view, image)

    def clean_point_clouds(self) -> np.ndarray:
        """
            Clean the point clouds.
            Output:
                - z_mask: mask representing cleaned point clouds based on z values
        """
        logger.info(f"{self.__LOG_PREFIX__}: Cleaning the point clouds")
        z = np.abs(stats.zscore(self.point_cloud))
        z_mask = (z < self.cloud_point_z_threshold).all(axis=-1)
        logger.info(f"{self.__LOG_PREFIX__}: Found {np.sum(~z_mask)} outliers")
        return z_mask
    
    def _generate_point_cloud(self, file_name: str = "point_cloud.ply") -> None:
        """
            Generate the point cloud.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating the point cloud")
        point_cloud = copy.deepcopy(self.point_cloud)
        if self.cloud_point_cleaning:
            z_mask = self.clean_point_clouds()
            point_cloud[z_mask == False] = 0
        colors = self._get_colors(point_cloud=point_cloud)
        success = pts2ply(point_cloud, colors, os.path.join(self.output_dir, file_name))
        if success:
            logger.info(f"{self.__LOG_PREFIX__}: Point cloud generated successfully")
        else:
            logger.error(f"{self.__LOG_PREFIX__}: Point cloud generation failed")

    def compute_reprojection_error(self, image: str) -> float:
        """
            Compute the reprojection error.
            Input parameters:
                - image: image name
            Output:
                - reprojection error
        """
        logger.info(f"{self.__LOG_PREFIX__}: Computing reprojection error for {image}")
        # Get data
        r, t, ref = self.image_data[image]
        keypoints, _ = self._load_features(image)
        # Get the 3D points
        points_3d = self.point_cloud[ref[ref >= 0].astype(int)]
        # Project 3D points onto 2D image
        r_vec, _ = cv2.Rodrigues(r)
        projected_points = cv2.projectPoints(points_3d, r_vec, t, self.K, None)[0].squeeze()
        # Get the true 2D points
        keypoints = np.array(keypoints)[ref >= 0]
        keypoints = np.array([kp.pt for kp in keypoints])
        # Compute the reprojection error mask
        error = np.linalg.norm(keypoints - projected_points, axis=-1)
        # Remove outliers post PnP
        if self.reprojection_remove_outliers:
            points_3d, mask = self._remove_outliers(error, self.reprojection_remove_outliers_threshold, points_3d)
            error = error[mask]
            self.point_cloud[ref[ref >= 0].astype(int)] = points_3d
            keypoints = keypoints[mask]
            projected_points = projected_points[mask]
        mean_error = np.mean(error)
        logger.info(f"{self.__LOG_PREFIX__}: Reprojection error for {image}: {mean_error}")
        # Plot the reprojection error
        if self.verbose:
            fig, ax = plt.subplots()
            file_path = self._get_image_path(image)
            image_np = cv2.imread(file_path)
            for i in range(len(keypoints)):
                cv2.circle(image_np, (keypoints[i, 0].astype(int), keypoints[i, 1].astype(
                    int)), self.true_circle_radius, self.true_circle_color, self.true_circle_thickness)
                cv2.circle(image_np, (projected_points[i, 0].astype(int), projected_points[i, 1].astype(
                    int)), self.projection_circle_radius, self.projection_circle_color, self.projection_circle_thickness)
                cv2.line(image_np, (keypoints[i, 0].astype(int), keypoints[i, 1].astype(int)), (projected_points[i, 0].astype(
                    int), projected_points[i, 1].astype(int)), self.connecting_line_color, self.connecting_line_thickness)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            ax.imshow(image_np)
            ax.set_title(
                f"Reprojection error for {image} - error: {mean_error}")
            fig.savefig(os.path.join(self.out_verbose_dir, f"{image}_reprojection_error.png"))
            plt.close(fig)
        return mean_error

    def _remove_outliers(self, error: np.ndarray, threshold: float, points_3d: np.ndarray) -> tuple:
        """
            Remove the outliers.
            Input parameters:
                - error: error values
                - threshold: threshold value
            Output:
                - points_3d: 3D points
                - mask representing the inliers
        """
        logger.info(f"{self.__LOG_PREFIX__}: Removing outliers based on threshold: {threshold}")
        inliers = error < threshold
        points_3d[~inliers] = 0
        return points_3d, inliers
    
    def bundle_adjustment(self) -> None:
        """
            Perform bundle adjustment.
        """

        def initial_guess() -> tuple:
            """
                Compute the initial guess for the camera poses.
                Returns: tuple containing the rotation and translation matrices
            """
            camera_intrinsics = []
            for _, image in enumerate(self.image_data.keys()):
                r, t, _ = self.image_data[image]
                camera_intrinsics.extend(np.hstack((r.ravel(), t.ravel())))
            points_3d = self.point_cloud.ravel()
            return np.hstack((np.array(camera_intrinsics), points_3d))
        
        def update(result: OptimizeResult) -> None:
            """
                Update the camera poses and 3D points.
                Input parameters:
                    - result: optimization result
            """
            n_views = len(self.image_data)
            camera_params = result.x[:n_views*12].reshape(-1, 12) # 12 => 3x3 (rotation) + 3x1 (translation)
            points_3d = result.x[n_views*12:].reshape(-1, 3)
            for idx, image in enumerate(self.image_data.keys()):
                r = camera_params[idx, :9].reshape(3, 3)
                t = camera_params[idx, 9:].reshape(3, 1)
                self.image_data[image][0] = r
                self.image_data[image][1] = t
            if self.point_cloud.shape != points_3d.shape:
                logger.error(f"{self.__LOG_PREFIX__}: Point cloud shape mismatch")
                raise ValueError("Point cloud shape mismatch")
            self.point_cloud = points_3d
        
        def get_jac_sparsity() -> np.ndarray:
            """
                Get the Jacobian sparsity.
                Returns: numpy array representing the Jacobian sparsity
            """
            n_views = len(self.image_data)
            n_points_3d = self.point_cloud.shape[0]
            n_params = n_views*12 + n_points_3d*3 # 12 => 3x3 (rotation) + 3x1 (translation)
            sparsity = np.zeros((0, n_params))
            for idx, image in enumerate(self.image_data.keys()):
                _, _, ref = self.image_data[image]
                keypoints, _ = self._load_features(image)
                keypoints = np.array(keypoints)[ref >= 0]
                sparsity_ = np.zeros((2*len(keypoints), n_params))
                # Fill camera correspondings
                sparsity_[:, idx*12:idx*12+12] = 1
                # Fill 3D points correspondings using references
                i = np.arange(len(keypoints))
                sparsity_[2*i, n_views*12 + 3*ref[ref >= 0].astype(int) + 0] = 1
                sparsity_[2*i+1, n_views*12 + 3*ref[ref >= 0].astype(int) + 0] = 1
                sparsity_[2*i, n_views*12 + 3*ref[ref >= 0].astype(int) + 1] = 1
                sparsity_[2*i+1, n_views*12 + 3*ref[ref >= 0].astype(int) + 1] = 1
                sparsity_[2*i, n_views*12 + 3*ref[ref >= 0].astype(int) + 2] = 1
                sparsity_[2*i+1, n_views*12 + 3*ref[ref >= 0].astype(int) + 2] = 1
                # Stack the sparsity
                sparsity = np.vstack((sparsity, sparsity_))
            sparsity_lil = sparse.lil_matrix(sparsity, dtype=int)
            del sparsity
            return sparsity_lil
        
        def F(*args, **kwargs) -> np.ndarray:
            """
                Compute the residuals.
                Returns: numpy residuals
            """
            residuals = np.empty((0, 2)) # Pixel coordinates
            for image in self.image_data.keys():
                r, t, ref = self.image_data[image]
                r_vec, _ = cv2.Rodrigues(r)
                keypoints, _ = self._load_features(image)
                keypoints = np.array(keypoints)[ref >= 0]
                keypoints = np.array([kp.pt for kp in keypoints])
                points_3d = self.point_cloud[ref[ref >= 0].astype(int)]
                projected_points = cv2.projectPoints(points_3d, r_vec, t, self.K, None)[0].squeeze()
                # add noise to the projected points
                projected_points += np.random.normal(0, 100, projected_points.shape)
                residuals = np.vstack((residuals, keypoints - projected_points))
            return residuals.flatten()

        opt_start_time = time.time()
        logger.info(f"{self.__LOG_PREFIX__}: Performing bundle adjustment")
        logger.info(f"{self.__LOG_PREFIX__}: Getting initial guess and sparse Jacobian")
        x0 = initial_guess()
        jac_sparcity = None
        x_scale = 1.0
        logger.info(f"{self.__LOG_PREFIX__}: Performing least squares optimization for bundle adjustment with {self.image_data.keys()} views")
        result = least_squares(F, x0=x0, jac_sparsity=jac_sparcity, method=self.least_squares_method, x_scale=x_scale, verbose=0, args=())
        logger.info(f"{self.__LOG_PREFIX__}: Updating parameters")
        update(result)
        logger.info(f"{self.__LOG_PREFIX__}: Bundle adjustment completed - cost: {result.cost} | time taken: {(time.time() - opt_start_time)/60} minutes")
    
    def run(self) -> None:
        """
            Run the Structure from Motion algorithm.

            This method performs the following steps:
                1. Performs baseline pose estimation for the first two images.
                2. Performs baseline triangulation for the first two images.
                3. Generates a 3D point cloud and evaluates reprojection error for the first two images.
                4. Performs pose estimation, triangulation, and reprojection error evaluation for the remaining images.
                5. Calculates the mean reprojection error for all images.
        """
        start_time = time.time()
        errors = []
        logger.info(f"{self.__LOG_PREFIX__}: Running the Structure from Motion algorithm")
        # Get image/view list
        self.image_list = self._get_image_list()
        # Load data
        self.keypoints, self.descriptors, self.matches_data = self._load_data()
        # Perform `baseline pose estimation` for the first two images with the most overlapping features
        if len(self.image_list) < 2:
            logger.error(f"{self.__LOG_PREFIX__}: Not enough image matches to perform SFM")
            raise ValueError("Not enough image matches to perform SFM")
        image_1, image_2 = self.image_list[0], self.image_list[1]
        self._baseline_pose_estimation(image_1, image_2)
        # Perform `baseline triangulation` for the first two images
        self._baseline_triangulation(image_1, image_2)
        # Perform bundle adjustment if required
        if self.perform_bundle_adjustment:
            self.bundle_adjustment()
        # Generate the point cloud into a file
        self._generate_point_cloud(file_name="cloud_base_view.ply")
        # Compute the reprojection error for the first two images
        image_1_error = self.compute_reprojection_error(image_1)
        errors.append(image_1_error)
        image_2_error = self.compute_reprojection_error(image_2)
        errors.append(image_2_error)
        # Perform `pose estimation`, `triangulation`, and `reprojection error evaluation` for the remaining images
        views_ctr = 2
        for idx, img in enumerate(self.image_list[2:]):
            # Perform pose estimation for the new image
            self._pose_estimation(img)
            # Perform triangulation for the new image
            self._triangulation(img)
            # Perform bundle adjustment if required
            if self.perform_bundle_adjustment:
                self.bundle_adjustment()
            # Generate the point cloud into a file
            self._generate_point_cloud(file_name=f"cloud_view_{views_ctr+idx}.ply")
            # Compute the reprojection error for the new image
            img_error = self.compute_reprojection_error(img)
            errors.append(img_error)
        # Calculate the mean reprojection error
        mean_error = np.mean(errors)
        logger.info(f"{self.__LOG_PREFIX__}: Mean reprojection error for all views: {mean_error}")
        logger.info(f"{self.__LOG_PREFIX__}: Structure from Motion algorithm completed - time taken: {(time.time() - start_time)/60} minutes")