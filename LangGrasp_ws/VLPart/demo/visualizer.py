# Copyright (c) Facebook, Inc. and its affiliates.
import colorsys
import logging
import math
import numpy as np
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask

logger = logging.getLogger(__name__)


def _create_text_labels(classes, scores, class_names, is_crowd=None, args=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    replace_set = [['monitor_(computer_equipment) computer_monitor', 'computer_monitor'],
                   ['_(computer_equipment)', ''],
                   ['_(automobile)', ''],
                   ['_(drink container)', ''],
                ]
    labels = None
    if classes is not None:
        if args.MODEL.EVAL_PROPOSAL:
            labels = ['' for _ in classes]
        elif class_names is not None and len(class_names) > 0:
            labels = []
            for i in classes:
                class_name = class_names[i]
                for long_name, short_name in replace_set:
                    class_name = class_name.replace(long_name, short_name)
                labels.append(class_name)
            # labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels



class CustomVisualizer(Visualizer):
    """
    Support to visualize the output of different supervision types 
    """
    def draw_instance_predictions(self, predictions, args):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        scores = None if not args.VIS.SCORE else scores
        boxes = None if not args.VIS.BOX else boxes

        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None), args=args)
        labels = None if not args.VIS.LABELS else labels
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            # The predicted mask output for a given part, in the form of an array of the number of predicted objects
            masks = np.asarray(predictions.pred_masks)
            # +++++++++++++++++++Get a mask map of the target location+++++++++++++++++++++++++++++
            for m, l in zip(masks, labels):
                # 创建一张与 masks[0] 形状相同的全零图像
                mask_image = np.zeros_like(masks[0], dtype=np.uint8)
                # 创建 PIL 图像对象，并将 m 中值为 True 的位置设为 255，即白色（255 对应的 uint8 值）
                mask_image[m] = 255

                # Extended mask
                # expanded_mask = expand_mask_with_dilation(mask_image, scale_factor=3, kernel_size=7)
                # Or use contour dilation
                expanded_mask = expand_mask_with_contours(mask_image, scale_factor=3)
                masks[0] = expanded_mask
                expanded_image = Image.fromarray(expanded_mask)
                extracted_labels = l.rsplit(' ', 1)[0]
                # Save the image as a PNG file with the name label
                expanded_image.save(f"VLPart_ws/data/segImg/{extracted_labels}.png")
                expanded_image.close()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Get the box plot of the target location +++++++++++++++++++++++++++++++++++++++++
            for box, label in zip(boxes, labels):
                mask_image = np.zeros((self.output.height, self.output.width), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, box)
                mask_image[y1:y2, x1:x2] = 255
                kernel1 = np.ones((80, 80), np.uint8)
                mask_image = cv2.dilate(mask_image, kernel1, iterations=1)
                # Convert numpy array to PIL image
                pil_image = Image.fromarray(mask_image)
                # Extract labels (remove the part after the last space to prevent long category names)
                extracted_labels = label.rsplit(' ', 1)[0]
                # Saving the image
                pil_image.save(f"/data/boxImg/{extracted_labels}_box.png")
                pil_image.close()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
        masks = None if not args.VIS.MASK else masks

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            # boxes=boxes,
            boxes=None,
            labels=labels,
            # labels=None,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
    
    
    def draw_box(self, box_coord, alpha=0.9, edge_color="g", line_style="-"):
        """
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output


# Contour expansion function
def expand_mask_with_contours(mask, scale_factor=1.2, iterations=1):
    """
    Expand the mask by extracting contours and dilating them with high precision.

    Args:
        mask (numpy array): The binary mask to be expanded.
        scale_factor (float): The factor by which to scale the mask size.
        iterations (int): Number of times to apply dilation.

    Returns:
        expanded_mask (numpy array): The dilated (expanded) mask.
    """
    # Ensure the mask is binary (0 and 255)
    mask = np.uint8(mask)

    # Step 1: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for the expanded area
    expanded_mask = np.zeros_like(mask)

    # Step 2: Create a high precision structuring element
    # This can be a disk-shaped kernel for better shape preservation during dilation
    kernel_size = int(scale_factor * 5)  # Adjust kernel size based on scale_factor
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Step 3: Iterate over each contour and expand it
    for contour in contours:
        # Create a mask for each contour and fill it
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Apply dilation to the contour region with high precision
        dilated_mask = cv2.dilate(contour_mask, kernel, iterations=iterations)

        # Add the dilated contour mask back to the expanded_mask
        expanded_mask = cv2.bitwise_or(expanded_mask, dilated_mask)

    return expanded_mask
