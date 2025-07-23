#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from hausdorff import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff
# In[2]:
import os
from matplotlib import pyplot as plt
from skimage import measure

# smoothing binary images

def smoothing_binary_image_array(image_array,kernal_size):
    size1, size2 = kernal_size
    kernel = np.ones((size1,size2), np.uint8)


    # Initialize an empty array to store the smoothed images
    smoothed_array = np.empty_like(image_array)

    # Loop through each image and apply the morphological closing operation
    for i in range(image_array.shape[0]):
        smoothed_array[i] = cv2.morphologyEx(image_array[i], cv2.MORPH_CLOSE, kernel)


    # # Display the first image and its corresponding smoothed image for comparison
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # axes[0].imshow(image_array[0], cmap='gray')
    # axes[0].set_title('Original Image')

    # axes[1].imshow(smoothed_array[0], cmap='gray')
    # axes[1].set_title('Smoothed Image')

    # plt.tight_layout()
    # plt.show()
    return smoothed_array


# In[ ]:


# def convert_images_bgr_to_rgb(images_np):
#     num_images = images_np.shape[0]
#     converted_images = np.empty_like(images_np)
    
#     for i in range(num_images):
#         image = images_np[i]
#         converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         converted_images[i] = converted_image
    
#     return converted_images


def convert_images_bgr_to_rgb_8bit(images_np):
    # Ensure the input images are of type uint8
    images_np = images_np.astype(np.uint8)
    
    # Pre-allocate the converted_images array
    converted_images = np.empty_like(images_np)
    
    # Convert each image from BGR to RGB
    for i in range(images_np.shape[0]):
        converted_images[i] = cv2.cvtColor(images_np[i], cv2.COLOR_BGR2RGB)
    
    return converted_images


# In[ ]:


def binarize_masks(masks, threshold=127):
    masks_bin = np.where(masks > threshold, 1, 0)
    return masks_bin


# In[ ]:


# def normalize_images(resized_images):
#     min_val = np.min(resized_images)
#     max_val = np.max(resized_images)
#     print("images Minimum value:", min_val)
#     print("images Maximum value:", max_val)

#     X_train_norm = np.vectorize(lambda x: x / max_val)(resized_images)

#     min_val_norm = np.min(X_train_norm)
#     max_val_norm = np.max(X_train_norm)
#     print("images_normalized Minimum value:", min_val_norm)
#     print("images_normalized Maximum value:", max_val_norm)

#     print("Normalization Done")
    
#     return X_train_norm


def normalize_images(resized_images):
    min_val = np.min(resized_images)
    max_val = np.max(resized_images)
    print("images Minimum value:", min_val)
    print("images Maximum value:", max_val)

    X_train_norm = np.empty_like(resized_images, dtype=np.float32)
    
    for i in tqdm(range(resized_images.shape[0]), desc="Normalizing images"):
        X_train_norm[i] = resized_images[i] / max_val
        # X_train_norm[i] = np.vectorize(lambda x: x / max_val)(resized_images[i])
        # X_train_norm[i]= X_train_norm[i].astype(np.float32)

    min_val_norm = np.min(X_train_norm[4])
    max_val_norm = np.max(X_train_norm[4])
    print("images_normalized Minimum value:", min_val_norm)
    print("images_normalizsed Maximum value:", max_val_norm)

    print("Normalization Done")
    
    return X_train_norm

# In[ ]:


def resize_images_and_masks(x_train, y_train, new_size):
    msize1, msize2 = new_size
    resized_images = []
    resized_masks = []
    
    for image in tqdm(x_train, desc="Resizing images"):
        resized_image = resize(image, (msize1, msize2), preserve_range=True)
        resized_images.append(resized_image)
    resized_images = np.array(resized_images)
    
    for mask in tqdm(y_train, desc="Resizing masks"):
        resized_mask = resize(mask, (msize1, msize2), preserve_range=True)
        resized_masks.append(resized_mask)
    resized_masks = np.array(resized_masks)
    
    print("x_train shape:", resized_images.shape, "x_train number of images:", len(resized_images))
    print("y_train shape:", resized_masks.shape, "y_train number of images:", len(resized_masks))
    
    return resized_images, resized_masks


def resize_masks(masks, new_size):
    """
    Resize a list of masks to a new size.

    Parameters:
    - masks: list or numpy array of masks to be resized
    - new_size: tuple of integers (new_height, new_width) for the new size

    Returns:
    - numpy array of resized masks
    """
    resized_masks = []
    for mask in tqdm(masks, desc="Resizing masks"):
        resized_mask = resize(mask, new_size, preserve_range=True)
        resized_masks.append(resized_mask)
    return np.array(resized_masks)

# def calculate_iou(image1, image2):

#     mask1 = image1
#     mask2 = image2
#     # Convert arrays to binary masks (apply threshold if necessary)
#     mask1 = (mask1 > 0.5).astype(np.uint8)  # Modify threshold as needed
#     mask2 = (mask2 > 0.5).astype(np.uint8)  # Modify threshold as needed

#     # Calculate intersection
#     intersection = np.logical_and(mask1, mask2).sum()

#     # Calculate union
#     union = np.logical_or(mask1, mask2).sum()

#     # Calculate IoU
#     iou = intersection / union

#     return iou

def calculate_iou(image1, image2):
    # Check for None
    if image1 is None or image2 is None:
        print("One or both input images are None. Returning IoU = 0.")
        return 0.0

    # Remove last channel if present
    if image1.ndim == 4 and image1.shape[-1] == 1:
        image1 = np.squeeze(image1, axis=-1)
    if image2.ndim == 4 and image2.shape[-1] == 1:
        image2 = np.squeeze(image2, axis=-1)

    # Convert to binary masks
    mask1 = (image1 > 0.5).astype(np.uint8)
    mask2 = (image2 > 0.5).astype(np.uint8)

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Handle zero union
    if union == 0:
        print("Union is zero, returning IoU = 0.")
        return 0.0

    return intersection / union



def calculate_dice_coefficient(mask1, mask2):
    # Remove singleton channel dimension if present
    if mask1.ndim == 4 and mask1.shape[-1] == 1:
        mask1 = np.squeeze(mask1, axis=-1)
    if mask2.ndim == 4 and mask2.shape[-1] == 1:
        mask2 = np.squeeze(mask2, axis=-1)
    intersection = np.logical_and(mask1, mask2).sum()
    total_pixels = mask1.sum() + mask2.sum()
    dice_coefficient = 2 * intersection / total_pixels if total_pixels != 0 else 0.0
    return dice_coefficient



def calculate_hausdorff_distance(mask1, mask2):
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    coords1 = np.column_stack(np.nonzero(mask1))
    coords2 = np.column_stack(np.nonzero(mask2))

    if coords1.size == 0 or coords2.size == 0:
        return 1e6  # fallback for empty masks

    hd1 = directed_hausdorff(coords1, coords2)[0]
    hd2 = directed_hausdorff(coords2, coords1)[0]
    return max(hd1, hd2)

def compute_all_hausdorff_distances(y_true, y_pred, max_valid=400):
    distances = []
    valid_distances = []

    for i in range(len(y_true)):
        hd = calculate_hausdorff_distance(y_true[i], y_pred[i])
        distances.append(hd)
        # print(f"Hausdorff Distance for image {i+1}: {hd:.2f} pixels")

        if hd <= max_valid:
            valid_distances.append(hd)
        else:
            print(f"⚠️ Skipped from mean: image {i+1} (HD={hd:.2f} > {max_valid})")

    if valid_distances:
        mean_hd = np.mean(valid_distances)
        # print(f"\n✅ Mean Hausdorff Distance (excluding > {max_valid} px): {mean_hd:.2f} pixels")
    else:
        mean_hd = None
        print("\n⚠️ No valid distances under threshold to compute mean.")

    return mean_hd



from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from scipy import stats

def calculate_precision_recall(image1, image2):
    """
    Calculate precision and recall for single-class segmentation.
    
    Args:
        image1: Ground truth binary mask (numpy array)
        image2: Predicted binary mask (numpy array)
        
    Returns:
        tuple: (precision, recall)
    """
    # Check if either input is None
    if image1 is None or image2 is None:
        print("One or both input images are None. Returning precision=recall=0.")
        return 0.0, 0.0
    
    # Convert arrays to binary masks (apply threshold if necessary)
    gt = (image1 > 0.5).astype(np.uint8)
    pred = (image2 > 0.5).astype(np.uint8)
    
    # Calculate true positives (TP), false positives (FP), false negatives (FN)
    TP = np.sum(np.logical_and(pred == 1, gt == 1))
    FP = np.sum(np.logical_and(pred == 1, gt == 0))
    FN = np.sum(np.logical_and(pred == 0, gt == 1))
    
    # Calculate precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Calculate recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return precision, recall


def calculate_volumetric_similarity(image1, image2):
    """
    Calculate volumetric similarity for single-class segmentation.

    Volumetric Similarity = 1 - abs(V1 - V2) / (V1 + V2)
    where V1 and V2 are the volumes (or areas in 2D) of the two masks.

    Args:
        image1: Ground truth binary mask (numpy array)
        image2: Predicted binary mask (numpy array)

    Returns:
        float: Volumetric similarity (range: 0–1, higher is better)
    """
    if image1 is None or image2 is None:
        print("One or both input images are None. Returning volumetric similarity = 0.")
        return 0.0

    # Convert to binary masks
    gt = (image1 > 0.5).astype(np.uint8)
    pred = (image2 > 0.5).astype(np.uint8)

    # Compute volumes as float64 to avoid overflow
    vol_gt = np.sum(gt, dtype=np.float64)
    vol_pred = np.sum(pred, dtype=np.float64)

    denominator = vol_gt + vol_pred
    if denominator == 0:
        return 1.0  # Both are empty, perfect match

    volumetric_similarity = 1.0 - abs(vol_gt - vol_pred) / denominator
    return volumetric_similarity

def calculate_boundary_f1_score(image1, image2, tolerance=2):
    """
    Calculate boundary F1-score for single-class segmentation.
    
    The boundary F1-score measures the accuracy of the segmentation boundaries.
    It identifies boundary pixels in both masks and calculates precision and recall
    of boundary pixels within a tolerance distance.
    
    Args:
        image1: Ground truth binary mask (numpy array)
        image2: Predicted binary mask (numpy array)
        tolerance: Maximum distance (in pixels) to consider a boundary match (default: 2)
        
    Returns:
        float: Boundary F1-score (0-1, higher is better)
    """
    # Check if either input is None
    if image1 is None or image2 is None:
        print("One or both input images are None. Returning boundary F1-score = 0.")
        return 0.0
    
    # Convert arrays to binary masks (apply threshold if necessary)
    gt = (image1 > 0.5).astype(np.uint8)
    pred = (image2 > 0.5).astype(np.uint8)
    
    # Extract boundaries of each mask
    # A boundary pixel is a pixel that belongs to the foreground but has
    # at least one background pixel in its neighborhood
    gt_eroded = binary_erosion(gt)
    pred_eroded = binary_erosion(pred)
    
    gt_boundary = gt - gt_eroded
    pred_boundary = pred - pred_eroded
    
    # Create distance transforms
    gt_boundary_dist = distance_transform_edt(1 - gt_boundary)
    pred_boundary_dist = distance_transform_edt(1 - pred_boundary)
    
    # Count boundary pixels
    gt_boundary_pixels = np.sum(gt_boundary)
    pred_boundary_pixels = np.sum(pred_boundary)
    
    # If either mask has no boundary pixels, handle special case
    if gt_boundary_pixels == 0 or pred_boundary_pixels == 0:
        if gt_boundary_pixels == 0 and pred_boundary_pixels == 0:
            return 1.0  # Both have no boundaries - perfect match
        else:
            return 0.0  # One has boundary, other doesn't - no match
    
    # Calculate precision: % of pred boundary pixels within tolerance of gt boundary
    pred_boundary_correct = np.sum(pred_boundary * (gt_boundary_dist <= tolerance))
    precision = pred_boundary_correct / pred_boundary_pixels
    
    # Calculate recall: % of gt boundary pixels within tolerance of pred boundary
    gt_boundary_correct = np.sum(gt_boundary * (pred_boundary_dist <= tolerance))
    recall = gt_boundary_correct / gt_boundary_pixels
    
    # Calculate F1-score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1
def calculate_p_value(ground_truth_set, prediction_set, metric_function):
    """
    Calculate p-value to determine statistical significance of results.
    
    This function compares your segmentation performance against a set of ground truth
    and prediction masks using a specified metric. It uses a paired t-test to
    determine if your results are statistically significant.
    
    Args:
        ground_truth_set: List/array of ground truth binary masks or a single ground truth mask
        prediction_set: List/array of predicted binary masks or a single prediction mask
        metric_function: Function that calculates the metric to compare
                        (should accept image1, image2 as parameters)
        
    Returns:
        tuple: (mean_value, std_dev, p_value)
    """
    import numpy as np
    from scipy import stats
    
    # Handle the case where single masks are passed instead of lists
    if not isinstance(ground_truth_set, list) and not isinstance(prediction_set, list):
        # If single masks are passed, we need to calculate just one metric value
        try:
            metric_value = metric_function(ground_truth_set, prediction_set)
            
            # If the metric function returns multiple values (like precision_recall),
            # we need to handle that case specifically
            if isinstance(metric_value, tuple):
                # For now, let's take the first value (e.g., precision)
                metric_value = metric_value[0]
                
            # With only one sample, we can't do statistical testing properly
            # But we can return the value, 0 std dev, and a default p-value
            return metric_value, 0.0, 1.0
        except Exception as e:
            print(f"Error calculating metric: {e}")
            return 0.0, 0.0, 1.0
    
    # Convert to lists if they are numpy arrays but not single masks
    if hasattr(ground_truth_set, 'shape') and len(ground_truth_set.shape) > 2:
        ground_truth_list = [ground_truth_set[i] for i in range(ground_truth_set.shape[0])]
    else:
        ground_truth_list = ground_truth_set
        
    if hasattr(prediction_set, 'shape') and len(prediction_set.shape) > 2:
        prediction_list = [prediction_set[i] for i in range(prediction_set.shape[0])]
    else:
        prediction_list = prediction_set
    
    # Validate inputs
    if len(ground_truth_list) == 0 or len(prediction_list) == 0:
        print("Empty input sets. Cannot calculate p-value.")
        return 0.0, 0.0, 1.0
    
    if len(ground_truth_list) != len(prediction_list):
        print("Input sets must have the same length. Cannot calculate p-value.")
        return 0.0, 0.0, 1.0
    
    # Calculate metric for each pair of images
    metric_values = []
    
    for gt, pred in zip(ground_truth_list, prediction_list):
        try:
            metric_value = metric_function(gt, pred)
            
            # If the metric function returns multiple values (like precision_recall),
            # we need to handle that case specifically
            if isinstance(metric_value, tuple):
                # For now, let's take the first value (e.g., precision)
                metric_value = metric_value[0]
                
            metric_values.append(metric_value)
        except Exception as e:
            print(f"Error calculating metric for a sample: {e}")
            # Skip this sample
    
    if not metric_values:
        print("No valid metric values calculated. Cannot perform statistical test.")
        return 0.0, 0.0, 1.0
    
    # Calculate mean and standard deviation
    mean_value = np.mean(metric_values)
    std_dev = np.std(metric_values)
    
    # Perform one-sample t-test against a neutral value (0.5)
    # This tests whether our results are significantly better than random chance
    t_stat, p_value = stats.ttest_1samp(metric_values, 0.5)
    
    return mean_value, std_dev, p_value
# def calculate_p_value(ground_truth_set, prediction_set, metric_function):
#     """
#     Calculate p-value to determine statistical significance of results.
    
#     This function compares your segmentation performance against a set of ground truth
#     and prediction masks using a specified metric. It uses a paired t-test to
#     determine if your results are statistically significant.
    
#     Args:
#         ground_truth_set: List of ground truth binary masks (numpy arrays)
#         prediction_set: List of predicted binary masks (numpy arrays)
#         metric_function: Function that calculates the metric to compare
#                         (should accept image1, image2 as parameters)
        
#     Returns:
#         tuple: (mean_value, std_dev, p_value)
#     """
#     # Validate inputs
#     if not ground_truth_set or not prediction_set:
#         print("Empty input sets. Cannot calculate p-value.")
#         return 0.0, 0.0, 1.0
    
#     if len(ground_truth_set) != len(prediction_set):
#         print("Input sets must have the same length. Cannot calculate p-value.")
#         return 0.0, 0.0, 1.0
    
#     # Calculate metric for each pair of images
#     metric_values = []
    
#     for gt, pred in zip(ground_truth_set, prediction_set):
#         metric_value = metric_function(gt, pred)
        
#         # If the metric function returns multiple values (like precision_recall),
#         # we need to handle that case specifically
#         if isinstance(metric_value, tuple):
#             # For now, let's take the first value (e.g., precision)
#             metric_value = metric_value[0]
            
#         metric_values.append(metric_value)
    
#     # Calculate mean and standard deviation
#     mean_value = np.mean(metric_values)
#     std_dev = np.std(metric_values)
    
#     # Perform one-sample t-test against a neutral value (0.5)
#     # This tests whether our results are significantly better than random chance
#     t_stat, p_value = stats.ttest_1samp(metric_values, 0.5)
    
#     return mean_value, std_dev, p_value

def calculate_average_surface_distance(image1, image2):
    """
    Calculate Average Surface Distance (ASD) between two binary masks.
    
    ASD is the average of all distances from points on boundary of one mask
    to the nearest point on boundary of the other mask, and vice versa.
    
    Args:
        image1: Ground truth binary mask (numpy array)
        image2: Predicted binary mask (numpy array)
        
    Returns:
        float: Average Surface Distance (lower is better)
    """
    # Check if either input is None
    if image1 is None or image2 is None:
        print("One or both input images are None. Returning ASD = inf.")
        return float('inf')
    
    # Convert arrays to binary masks (apply threshold if necessary)
    gt = (image1 > 0.5).astype(np.uint8)
    pred = (image2 > 0.5).astype(np.uint8)
    
    # Extract boundaries of each mask
    gt_eroded = binary_erosion(gt)
    pred_eroded = binary_erosion(pred)
    
    gt_boundary = gt - gt_eroded
    pred_boundary = pred - pred_eroded
    
    # Create distance transforms
    gt_boundary_dist = distance_transform_edt(1 - gt_boundary)
    pred_boundary_dist = distance_transform_edt(1 - pred_boundary)
    
    # Count boundary pixels
    gt_boundary_pixels = np.sum(gt_boundary)
    pred_boundary_pixels = np.sum(pred_boundary)
    
    # If either mask has no boundary pixels, return infinite distance
    if gt_boundary_pixels == 0 or pred_boundary_pixels == 0:
        return float('inf')
    
    # Calculate mean surface distance from pred to gt
    mean_dist_pred_to_gt = np.sum(gt_boundary_dist * pred_boundary) / pred_boundary_pixels
    
    # Calculate mean surface distance from gt to pred
    mean_dist_gt_to_pred = np.sum(pred_boundary_dist * gt_boundary) / gt_boundary_pixels
    
    # Average of both distances
    asd = (mean_dist_pred_to_gt + mean_dist_gt_to_pred) / 2.0
    
    return asd


# from scipy import stats
# from typing import Tuple

# # Your Dice function
# def calculate_dice_coefficient(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()
#     total_pixels = mask1.sum() + mask2.sum()
#     dice_coefficient = 2 * intersection / total_pixels if total_pixels != 0 else 0.0
#     return dice_coefficient

# # Compute Dice scores for all test samples
# def compute_dice_scores(y_true_all: np.ndarray, y_pred_all: np.ndarray) -> np.ndarray:
#     return np.array([
#         calculate_dice_coefficient(y_true_all[i], y_pred_all[i])
#         for i in range(len(y_true_all))
#     ])

# # Confidence interval function
# def confidence_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float]:
#     mean = np.mean(data)
#     se = stats.sem(data)
#     margin = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
#     return mean - margin, mean + margin

# # Main analysis function
# def analyze_dice_groups(y_true_all: np.ndarray, y_pred_all: np.ndarray):
#     dice_scores = compute_dice_scores(y_true_all, y_pred_all)

#     # Grouping by Dice threshold
#     bad = dice_scores[dice_scores < 0.5]
#     moderate = dice_scores[(dice_scores >= 0.5) & (dice_scores < 0.8)]
#     good = dice_scores[dice_scores >= 0.8]

#     groups = {'Bad': bad, 'Moderate': moderate, 'Good': good}

#     # Confidence intervals and stats
#     print("Group Statistics:")
#     for name, group in groups.items():
#         if len(group) == 0:
#             print(f"{name}: No samples.")
#             continue
#         mean = np.mean(group)
#         ci_low, ci_high = confidence_interval(group)
#         print(f"{name}: N={len(group)} | Mean Dice={mean:.3f} | 95% CI=({ci_low:.3f}, {ci_high:.3f})")

#     # Kruskal-Wallis Test (non-parametric)
#     if all(len(g) > 0 for g in groups.values()):
#         stat, p_value = stats.kruskal(bad, moderate, good)
#         print(f"\nKruskal-Wallis P-value: {p_value:.4f}")
#     else:
#         print("\nP-value cannot be computed: one or more groups are empty.")

# # Example usage
# # y_true_all = np.load("ground_truth.npy")     # Shape: (N, H, W)
# # y_pred_all = np.load("predicted_masks.npy")  # Shape: (N, H, W)
# # analyze_dice_groups(y_true_all, y_pred_all)
import numpy as np
import os
from scipy import stats
from typing import Tuple
import imageio.v2 as imageio  # for saving .png images

# # Your Dice function
# def calculate_dice_coefficient(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2).sum()
#     total_pixels = mask1.sum() + mask2.sum()
#     dice_coefficient = 2 * intersection / total_pixels if total_pixels != 0 else 0.0
#     return dice_coefficient

# Compute Dice scores for all test samples
def compute_dice_scores(y_true_all: np.ndarray, y_pred_all: np.ndarray) -> np.ndarray:
    return np.array([
        calculate_dice_coefficient(y_true_all[i], y_pred_all[i])
        for i in range(len(y_true_all))
    ])

# Confidence interval function
def confidence_interval(data: np.ndarray, confidence=0.95) -> Tuple[float, float]:
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
    return mean - margin, mean + margin

# Save image and mask to group folders
def save_sample(image, mask, index, group_name, base_path):
    group_folder = os.path.join(base_path, group_name)
    os.makedirs(group_folder, exist_ok=True)
    image_path = os.path.join(group_folder, f"image_{index}.png")
    mask_path = os.path.join(group_folder, f"mask_{index}.png")
    imageio.imwrite(image_path, (image * 255).astype(np.uint8))  # save binary as 0-255
    imageio.imwrite(mask_path, (mask * 255).astype(np.uint8))

# Main analysis and grouping function
def analyze_and_save_groups(x_test: np.ndarray, y_true_all: np.ndarray, y_pred_all: np.ndarray, base_output_path: str):
    dice_scores = compute_dice_scores(y_true_all, y_pred_all)

    bad, moderate, good = [], [], []
    group_indices = {'Bad': [], 'Moderate': [], 'Good': []}

    for i, score in enumerate(dice_scores):
        if score < 0.5:
            bad.append(score)
            group_name = 'Bad'
        elif score < 0.8:
            moderate.append(score)
            group_name = 'Moderate'
        else:
            good.append(score)
            group_name = 'Good'
        
        group_indices[group_name].append(i)
        save_sample(x_test[i], y_pred_all[i], i, group_name, base_output_path)

    # Statistics
    print("Group Statistics:")
    for group_name, scores in [('Bad', bad), ('Moderate', moderate), ('Good', good)]:
        scores = np.array(scores)
        if len(scores) == 0:
            print(f"{group_name}: No samples.")
            continue
        mean = np.mean(scores)
        ci_low, ci_high = confidence_interval(scores)
        print(f"{group_name}: N={len(scores)} | Mean Dice={mean:.3f} | 95% CI=({ci_low:.3f}, {ci_high:.3f})")

    # Kruskal-Wallis test
    if all(len(g) > 0 for g in [bad, moderate, good]):
        stat, p_value = stats.kruskal(bad, moderate, good)
        print(f"\nKruskal-Wallis P-value: {p_value:.4f}")
    else:
        print("\nP-value cannot be computed: one or more groups are empty.")

# Example usage
# Paths
# base_path = "C:/Users/mehdi.eshragh.SSILIFT/Downloads/My Hybrid Model Data_May 15"
# x_test = np.load(os.path.join(base_path, "x_test_1024_101112.npy"))
# y_true = np.load(os.path.join(base_path, "y_test_1024_101112.npy"))
# y_pred = np.load(os.path.join(base_path, "y_pred_1024_101112.npy"))
# analyze_and_save_groups(x_test, y_true, y_pred, base_path)


# def draw_contours_on_images(x_test, y_pred_final, save_path):
#     import os
#     import numpy as np
#     import cv2

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     for idx in range(len(x_test)):
#         img = x_test[idx].copy()
#         mask = y_pred_final[idx]

#         # Convert image to uint8 if necessary
#         if img.dtype != np.uint8:
#             img = np.clip(img * 255, 0, 255).astype(np.uint8)

#         # Process the mask
#         if mask.dtype != np.uint8:
#             mask = (mask > 0.5).astype(np.uint8)
#         elif np.max(mask) > 1:
#             mask = (mask > 127).astype(np.uint8)

#         mask_uint8 = mask * 255

#         # Find contours
#         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#         # Draw contours in red (OpenCV uses BGR)
#         img_with_contour = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)

#         # Convert RGB to BGR before saving
#         bgr_image = cv2.cvtColor(img_with_contour, cv2.COLOR_RGB2BGR)

#         # Save the image
#         out_filename = os.path.join(save_path, f"image_with_contour_{idx}.png")
#         cv2.imwrite(out_filename, bgr_image)

def draw_contours_on_images(x_test, y_pred_final, y_test, save_path):


    def calculate_dice_coefficient(y_true, y_pred):
        y_true = y_true.astype(np.bool_)
        y_pred = y_pred.astype(np.bool_)
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-7)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(len(x_test)):
        img = x_test[idx].copy()
        pred_mask = y_pred_final[idx]
        true_mask = y_test[idx]

        # Convert image to uint8
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

        # Binarize predicted mask
        if pred_mask.dtype != np.uint8:
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
        elif np.max(pred_mask) > 1:
            pred_mask = (pred_mask > 127).astype(np.uint8)
        pred_mask_uint8 = pred_mask * 255

        # Compute Dice accuracy
        dice = calculate_dice_coefficient(true_mask, pred_mask)
        dice_text = f"Dice: {dice:.4f}"

        # Find detailed contours
        contours, _ = cv2.findContours(pred_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw contours on the image
        img_with_contour = cv2.drawContours(img.copy(), contours, -1, (0, 255, 255), 2)

        # Convert to BGR for saving
        bgr_img = cv2.cvtColor(img_with_contour, cv2.COLOR_RGB2BGR)

        # Put Dice score in top-left
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_position = (10, 30)
        cv2.putText(bgr_img, dice_text, text_position, font, font_scale, (255, 255, 255), thickness)

        # Save the image
        out_filename = os.path.join(save_path, f"image_with_contour_{idx}.png")
        cv2.imwrite(out_filename, bgr_img)



###################### draw_three_contours #################################################################
def draw_three_contours(x_test, y_pred_1, y_pred_2, y_test, save_path):
    def calculate_dice_coefficient(y_true, y_pred):
        y_true = y_true.astype(np.bool_)
        y_pred = y_pred.astype(np.bool_)
        intersection = np.logical_and(y_true, y_pred).sum()
        return (2. * intersection) / (y_true.sum() + y_pred.sum() + 1e-7)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(len(x_test)):
        img = x_test[idx].copy()
        pred_mask_1 = y_pred_1[idx]
        pred_mask_2 = y_pred_2[idx]
        true_mask = y_test[idx]

        # Convert image to uint8 RGB
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)

        img_with_contours = img.copy()

        # Preprocessing masks
        def preprocess_mask(mask):
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
            elif np.max(mask) > 1:
                mask = (mask > 127).astype(np.uint8)
            return mask * 255

        true_mask_bin = preprocess_mask(true_mask)
        pred_mask_1_bin = preprocess_mask(pred_mask_1)
        pred_mask_2_bin = preprocess_mask(pred_mask_2)

        # Compute Dice scores
        dice_1 = calculate_dice_coefficient(true_mask, pred_mask_1)
        dice_2 = calculate_dice_coefficient(true_mask, pred_mask_2)

        # Find and draw contours
        contours_true, _ = cv2.findContours(true_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_pred1, _ = cv2.findContours(pred_mask_1_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_pred2, _ = cv2.findContours(pred_mask_2_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(img_with_contours, contours_true, -1, (0, 255, 0), 2)   # Green for GT
        cv2.drawContours(img_with_contours, contours_pred2, -1, (255, 0, 0), 2)  # Blue for pred 1
        cv2.drawContours(img_with_contours, contours_pred1, -1, (0, 0, 255), 2)  # Red for pred 2

        # Convert to BGR for saving
        bgr_img = cv2.cvtColor(img_with_contours, cv2.COLOR_RGB2BGR)

        # Write Dice scores with color info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y0 = 25
        dy = 25
        texts = [
            ("Ground Truth: Green", (0, 255, 0)),
            (f"U_net_Model Dice: {dice_2:.4f} (Red)", (0, 0, 255)),
            (f"Hybrid_Model Dice: {dice_1:.4f} (Blue)", (255, 0, 0)),
        ]
        for i, (text, color) in enumerate(texts):
            y = y0 + i * dy
            cv2.putText(bgr_img, text, (10, y), font, font_scale, color, thickness)

        # Save image
        out_filename = os.path.join(save_path, f"image_with_3_contours_{idx}.png")
        cv2.imwrite(out_filename, bgr_img)

############ Save Binary images from numpy array #####################

def save_binary_masks(binarry_image_array, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, mask in enumerate(binarry_image_array):
        # Ensure binary format (0 or 255)
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8) * 255
        elif np.max(mask) <= 1:
            mask = mask * 255

        # Convert to uint8
        mask = mask.astype(np.uint8)

        # Save mask
        filename = os.path.join(save_path, f"mask_{idx}.png")
        cv2.imwrite(filename, mask)

############ describe_numpy_array #####################

def describe_numpy_array(arr, name="Array"):
    print(f"--- {name} Info ---")
    print(f"Shape: {arr.shape}")
    print(f"Data type: {arr.dtype}")
    print(f"Min value: {arr.min()}")
    print(f"Max value: {arr.max()}")
    print(f"Mean: {arr.mean():.4f}")
    print(f"Std Dev: {arr.std():.4f}")
    print(f"Unique values: {np.unique(arr)[:10]}{'...' if len(np.unique(arr)) > 10 else ''}")
    print(f"Is binary: {np.array_equal(np.unique(arr), [0, 1])}")



############ descrCalculate precision, recall (sensitivity), specificity, and F1-scoreibe_numpy_array #####################

def calculate_all_metrics(image1, image2):
    """
    Calculate precision, recall (sensitivity), specificity, and F1-score
    for single-class segmentation.

    Args:
        image1: Ground truth binary mask (numpy array)
        image2: Predicted binary mask (numpy array)

    Returns:
        dict: A dictionary with precision, recall, specificity, and F1-score
    """
    if image1 is None or image2 is None:
        print("One or both input images are None. Returning all metrics as 0.")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1_score": 0.0
        }

    # Convert arrays to binary masks
    gt = (image1 > 0.5).astype(np.uint8)
    pred = (image2 > 0.5).astype(np.uint8)

    # Confusion matrix components
    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    # Metric calculations
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print ("precision:", precision,"recall:" ,recall,"specificity:", specificity,"f1_score:", f1_score)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score
    }

def draw_mask_contour_on_image(image_np, mask_np, contour_color=(0, 255, 255), contour_thickness=2):
    """
    Draws contour of binary mask onto RGB image and returns a PIL image.
    
    Args:
        image_np (np.ndarray): RGB image array (H, W, 3), values 0–255 or 0–1
        mask_np (np.ndarray): Binary mask array (H, W), values 0 or 1
        contour_color (tuple): BGR color tuple for contour (OpenCV format)
        contour_thickness (int): Contour line thickness

    Returns:
        PIL.Image: Image with contour overlay
    """
    import cv2
    from PIL import Image
    import numpy as np

    # Ensure image is uint8 RGB (0–255)
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # Ensure mask is binary uint8 (0 or 255)
    if mask_np.max() <= 1:
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255
    else:
        mask_np = (mask_np > 127).astype(np.uint8) * 255

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Find contours on binary mask
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw contours in place
    image_with_contours = cv2.drawContours(image_bgr.copy(), contours, -1, contour_color, contour_thickness)

    # Convert back to RGB for Pillow
    image_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)
