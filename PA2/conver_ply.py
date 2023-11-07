import numpy as np
import open3d as o3d

threeD = np.load('two_view_recon_info/3D_points.npy')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(threeD)
o3d.io.write_point_cloud("result/test.ply", pcd)