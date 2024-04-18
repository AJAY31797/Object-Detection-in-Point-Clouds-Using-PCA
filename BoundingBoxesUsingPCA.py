#Importing PCA from sklearn
from sklearn.decomposition import PCA
import math
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load Point Cloud Data Clusters

# Read data from the text file
file_path = "C:/Users/aagr657/pointcloud_clusters.txt"
data = np.loadtxt(file_path, delimiter=' ')

# Well, I am not sure If I should do the standardization or not, but for factor of safety, I am doing it. 
# Although, I don't think that is needed because both the axes are on the same scale. 

# Separate the coordinates and cluster numbers
coordinates = data[:, :3]  # Extract x, y, z coordinates
cluster_numbers = data[:, 3]  # Extract cluster numbers

# Group coordinates by cluster number
clusters = {}
for i, cluster_number in enumerate(cluster_numbers):
    if cluster_number not in clusters:
        clusters[cluster_number] = []
    clusters[cluster_number].append(coordinates[i])

def calculate_bounding_box(cluster):
    #Fitting the PCA on the data
    model = PCA(n_components=2)
    
    #scaler = StandardScaler()
    #standardized_data = scaler.fit_transform(cluster[:, :2])

    principalcomponentmodel = model.fit(cluster[:, :2])
    
    # Calculate centroid along X and Y axes
    centroid_xy = np.mean(cluster[:, :2], axis=0)
    # print(centroid_xy)

    # Get principal components (eigenvectors)
    principal_components = principalcomponentmodel.components_
    # Principal components store the unit vectors along the maximum variance axes.
    # The result here is a 2X2 matrix, where each row represents a vector. 
    
    # You need to get the bounding box along these principal components.
    # For that, we need to find the maximum distance of projected points along these axes, which will help us to get the
    # maximum coordinates. 
    # print(principal_components)
    # First finding the projection of each point and then finding their distance from origin
    projections = np.abs(np.dot(cluster[:,:2]-centroid_xy, principal_components.T))
    projections_PC1 = np.max(projections[:,0])
    projections_PC2 = np.max(projections[:,1])
    
    # First finding the angle between the original X axis and the Principal component 1. 
    PC1 = principal_components[0]
    X_axis = np.array([1,0])
    
    angle = np.arccos(np.dot(PC1, X_axis))
    
    # There is something called a rotation matrix to identify the coordinates of a point from X-Y to a rotated X-Y. 
    # To do the inverse of it, you can use the inverse rotation matrix.
    
    inverse_rotation_matrix = np.array([[math.cos(angle), -1*math.sin(angle)],
                                      [math.sin(angle), math.cos(angle)]])
    
    # Get the bounding box of these projections
    vertices = np.array([[+projections_PC1, +projections_PC2],
                         [-projections_PC1, +projections_PC2],
                         [-projections_PC1, -projections_PC2],
                         [+projections_PC1, -projections_PC2]])
    
    # These vertices are still in X-Y coordinates. We need to rotate it so that these values
    # transformed_vertices = vertices * np.std(cluster[:, :2], axis=0) + np.mean(cluster[:, :2], axis=0)
    
    # You need to find the dot product of this rotation matrix with the rotated coordinates to convert them in actual coordinates. 
    for i in range(len(vertices)):
        vertices[i] = np.dot(inverse_rotation_matrix, vertices[i].T)
        
    # Now you need to add these coordinates to the centeroid to shift the whole set of coordinates
    
    transformed_vertices = vertices + centroid_xy
    
    # Fix minimum and maximum values along Z axis
    min_z = np.min(cluster[:, 2])
    max_z = np.max(cluster[:, 2])
    
    # Define the Z coordinates for the bottom and top faces of the bounding box
    vertices_z = np.array([[min_z], [max_z]])
    # Form combinations of X, Y, and Z coordinates to create 8 vertices
    # Combine XY vertices with Zmin
    vertices_zmin = np.hstack((transformed_vertices, np.full((4, 1), min_z)))
    # Combine XY vertices with Zmax
    vertices_zmax = np.hstack((transformed_vertices, np.full((4, 1), max_z)))
    # Combine both sets of vertices
    vertices = np.vstack((vertices_zmin, vertices_zmax))
    
    return vertices, principal_components
 
def plot_bounding_box(bounding_box, color):
    # Define the vertices of the bounding box
    vertices = [
        bounding_box[0],
        bounding_box[1],
        bounding_box[2],
        bounding_box[3],
        bounding_box[4],
        bounding_box[5],
        bounding_box[6],
        bounding_box[7]
    ]
    # Define the indices to form the edges of the bounding box
    edges = [
        [0, 1], [1, 2], [2, 3],
        [0, 3], [4, 5], [5, 6],
        [6, 7], [4, 7], [0, 4],
        [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        ax.plot3D([vertices[edge[0]][0], vertices[edge[1]][0]],
                  [vertices[edge[0]][1], vertices[edge[1]][1]],
                  [vertices[edge[0]][2], vertices[edge[1]][2]],
                  color=color)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colormap = plt.cm.get_cmap('hsv')  # You can choose any colormap you like
num_clusters = len(clusters)
cluster_colors = [colormap(i / num_clusters) for i in range(num_clusters)]


# Plot bounding boxes for each cluster
for cluster_number, cluster in clusters.items():
    cluster = np.array(cluster)
    bounding_box, components = calculate_bounding_box(cluster)
    #print(bounding_box)
    plot_bounding_box(bounding_box, color=cluster_colors[int(cluster_number)]) 

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Clusters with Bounding Boxes')

# Set the angle of view (elevation, azimuth)
ax.view_init(elev=90, azim=90)

# Show plot
plt.show()