#%%
import open3d as o3d
import time
import matplotlib.pyplot as plt
from pymeshfix import _meshfix
import mcubes

pcd = o3d.io.read_point_cloud('./test_data/apple.ply', format='ply')
pcd_orig = pcd

def show_pc(pc):
    if isinstance(pc, list):
        o3d.visualization.draw_geometries(pc)
    else:
        o3d.visualization.draw_geometries([pc])

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def show_point_cloud_clusters(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=3, min_points=100, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def show_largest_mesh_cluster(mesh):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print("Show largest cluster")
    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_1])

# %%

filter_aggressiveness = 0.2
g = filter_aggressiveness*10
if g < 1:
    g = 1

pcd = pcd_orig
pcd = pcd.voxel_down_sample(voxel_size=max(0.1, g*0.05))
pcd, ind = pcd.remove_radius_outlier(nb_points=int((1+0.5*g + 0.5*g**2)), radius=2)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=int((1+0.5*(g+2) + 0.5*(g+2)**2)), std_ratio=(2-0.1*g))

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=2, min_points=int((1+0.5*(g+2) + 0.5*(g+2)**1)), print_progress=True))
ind = []
for i in range(len(labels)):
    if labels[i] >= 0:
        ind.append(i)
pcd = pcd.select_by_index(ind)

o3d.io.write_point_cloud("./test_data/pcd.ply", pcd)
pcd_tmp = pcd

pcd = pcd.uniform_down_sample(32)
pcd.estimate_normals()

show_pc(pcd)

# %% Create Mesh
method = 'mcubes'

if method == 'poisson':
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12, linear_fit=True)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
elif method == 'ball_pivoting':
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    dist_90pct = np.percentile(distances, 90)
    radius = 1 * dist_90pct
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

mesh.remove_non_manifold_edges()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh = mesh.merge_close_vertices(eps=1.0)
mesh = mesh.subdivide_midpoint(number_of_iterations=2)
mesh = mesh.filter_smooth_taubin(number_of_iterations=50)
mesh = mesh.simplify_vertex_clustering(voxel_size=max(0.1, (1-g)*0.05))
mesh = mesh.filter_smooth_taubin(number_of_iterations=100)
o3d.io.write_triangle_mesh("./test_data/mesh.ply", mesh)

show_pc(mesh)

mesh_tmp = mesh
# %%
num_points = np.array(pcd.points).shape[0]
pcd_tmp = pcd
# %%
pcd = mesh.sample_points_poisson_disk(num_points)
show_pc(pcd)

