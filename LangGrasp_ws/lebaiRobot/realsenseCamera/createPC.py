import open3d as o3d
import numpy as np

# readingImages
color_image = o3d.io.read_image('../../data/input/17.png')
depth_image = o3d.io.read_image('../../data/depthImg/depth17.png')
mask = o3d.io.read_image('../../data/segImg/mug handle.png')

# Create an RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)
intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
intrinsic.set_intrinsics(width=1280, height=720, fx=653.22937, fy=653.45892, cx=645.9348, cy=368.38336)
intrinsic_matrix = intrinsic.intrinsic_matrix
point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
depth_array = np.asarray(rgbd_image.depth)
mask_array = np.asarray(mask)
green_color = [0, 1, 0]
mask_points = np.argwhere(mask_array == 255)

points_list = []
colors_list = []

for y in range(depth_array.shape[0]):
    for x in range(depth_array.shape[1]):
        depth_value = depth_array[y, x]
        if depth_value == 0:
            continue
        world_point = np.array([
            (x - intrinsic_matrix[0, 2]) * depth_value / intrinsic_matrix[0, 0],
            (y - intrinsic_matrix[1, 2]) * depth_value / intrinsic_matrix[1, 1],
            depth_value
        ])
        color_value = np.asarray(color_image)[y, x] / 255.0
        if (y, x) in mask_points:
            color_value = green_color
        points_list.append(world_point)
        colors_list.append(color_value)

# Add all points and colors to the point cloud
point_cloud.points = o3d.utility.Vector3dVector(np.array(points_list))
point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors_list))

# Visualize the entire point cloud (points in the region of interest in green, other areas in the original color)
o3d.visualization.draw_geometries([point_cloud])