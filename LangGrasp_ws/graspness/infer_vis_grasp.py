import os
import numpy as np
import argparse
from PIL import Image
import time
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup
import sys
sys.path.append('.')


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from utils.collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='../data')
parser.add_argument('--checkpoint_path', default='data/checkpoint/minkuresunet_epoch10.tar')
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/home/wwq/Downloads/practice/VLPart_ws/graspness/data/logs')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_point', type=int, default=60000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float,
                    default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=True)
parser.add_argument('--vis', action='store_true', default=True)
cfgs = parser.parse_args()

if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# 作用：生成全景点云
def test():
    root = cfgs.dataset_root
    input_rgb = 'test_color.png'
    input_depth = 'test_depth.png'
    workspace = 'workspace_mask.png'

    color = np.array(Image.open(os.path.join(root, 'input', input_rgb)), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(root, 'depthImg', input_depth)))
    workspace_mask = np.array(Image.open(os.path.join(root, 'input', workspace)))

    camera = CameraInfo(1280.0, 720.0, 1.206*1080/2, 2.14450693*720/2, 640, 360, 1000.0)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'point_colors': color_sampled.astype(np.float32),
                }
    return ret_dict

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def data_process(seg_label):
    root = cfgs.dataset_root

    input_rgb = 'test_color1.png'
    input_depth = 'test_depth1.png'

    color = np.array(Image.open(os.path.join(root, 'input', input_rgb)), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(root, 'depthImg', input_depth)))
    seg = np.array(Image.open(os.path.join(root, 'segImg',  seg_label + '.png')))

    camera = CameraInfo(1280.0, 720.0, 1.206*1080/2, 2.14450693*720/2, 640, 360, 1000.0)
    # camera = CameraInfo(1280.0, 720.0, 645, 645, 640, 360, 1000)

    # generate cloud
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(root, 'camera/camera_poses.npy'))
    align_mat = np.load(os.path.join(root, 'camera/cam0_wrt_table.npy'))
    # trans = np.dot(align_mat, camera_poses[int(index)])
    trans = np.dot(align_mat, camera_poses[int(0000)])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points random
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                'point_colors': color_sampled.astype(np.float32),
                'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
                'feats': np.ones_like(cloud_sampled).astype(np.float32),
                }
    return ret_dict


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference(data_input, seg_label):
    batch_data = minkowski_collate_fn([data_input])
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    tic = time.time()

    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # Forward pass
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)
    # collision detection
    if cfgs.collision_thresh > 0:
        cloud = data_input['point_clouds']
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
        gg = gg[~collision_mask]

    # save grasps
    save_dir = os.path.join(cfgs.dump_dir)
    save_path = os.path.join(save_dir, seg_label + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)

    toc = time.time()
    print('inference time: %fs' % (toc - tic))


# the grasp pose with the highest score is output
def get_max_score_grasp(grasp):
    gg1 = grasp[0]
    rot = gg1.rotation_matrix
    trans = gg1.translation

    # rotation matrix format conversion
    data = np.array2string(rot)
    data1 = data.replace('\n', ' ').replace('  ', ' ').strip()
    elements = data1.replace('[', '').replace(']', '').split()
    # A data structure reorganized into three rows and three columns
    rotation = []
    row = []
    for i, elem in enumerate(elements):
        row.append(float(elem))
        if (i + 1) % 3 == 0:
            rotation.append(row)
            row = []
    trans_string = np.array2string(trans)
    # Convert the string to a numpy array
    array = np.fromstring(trans_string.strip('[]'), sep=' ')

    # Convert the numpy array to a list and format it to the desired format
    translation = array.tolist()
    return translation, rotation, gg1.score


def main(seg_label):
    # the text label of the segmented part
    data_dict = data_process(seg_label)
    if cfgs.infer:
        inference(data_dict, seg_label)
    if cfgs.vis:
        pc = data_dict['point_clouds']
        gg = np.load(os.path.join(cfgs.dump_dir, seg_label + '.npy'))
        gg = GraspGroup(gg)
        gg = gg.nms()
        gg = gg.sort_by_score()

        if gg.__len__() > 0:
            gg = gg[:1]
        else:
            print("there is no grasping posture")
        grippers = gg.to_open3d_geometry_list()

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(data_dict['point_colors'].astype(np.float32))   # 标记，，生成rgb点云

        o3d.visualization.draw_geometries([cloud, *grippers])
        return get_max_score_grasp(gg)

if __name__ == '__main__':
    seg_label = "knife handle"
    main(seg_label)

