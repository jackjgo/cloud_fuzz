# -*- coding: utf-8 -*-
"""
Placeholder sctipt as I develop my methodology

My methodology does indeed work, so I'll be rewriting this into a useable 
format.
"""

import pdal
import numpy as np
import pandas as pd
import laspy

inputLas = "./data/test1.las"

cloud = laspy.read(inputLas)
Xs = cloud.x
Xs = ((Xs.array * Xs.scale) + Xs.offset)
# Xs = Xs[::10]
Ys = cloud.y
Ys = ((Ys.array * Ys.scale) + Ys.offset)
# Ys = Ys[::10]
Zs = cloud.z
Zs = ((Zs.array * Zs.scale) + Zs.offset)
# Zs = Zs[::10]
xyz = np.vstack([[Xs],[Ys],[Zs]])

#------------Downsample into thin point cloud------------
thin_json = """
[
     "./data/test1.las",
     {
          "type": "filters.voxelcenternearestneighbor",
          "cell": 0.1
     },
     {
          "type": "filters.mad",
          "dimension": "z",
          "k": 2.0
     },
     {
          "type": "filters.voxelcenternearestneighbor",
          "cell": 1
     },
     "./data/thinned.las"
]
"""
thin_pipeline = pdal.Pipeline(thin_json)
thin_count = thin_pipeline.execute()
thin_arrays = thin_pipeline.arrays
thin_metadata = thin_pipeline.metadata
thin_log = thin_pipeline.log

#--------------------Calculate Normals-------------------
normal_json = """
[
     "./data/thinned.las",
     {
          "type": "filters.normal",
          "knn": 8
     },
     {
          "type": "writers.bpf",
          "filename": "./data/normals.bpf",
          "output_dims": "X,Y,Z,NormalX,NormalY,NormalZ,Curvature"
     }
]
"""
normal_pipeline = pdal.Pipeline(normal_json)
normal_count = normal_pipeline.execute()
normal_arrays = normal_pipeline.arrays
normal_metadata = normal_pipeline.metadata
normal_log = normal_pipeline.log

# Write normals to text instead of bpf
txtNorm_json = """
[
     "./data/normals.bpf",
     {
          "type": "writers.text",
          "filename": "./data/normals.txt"
     }
]
"""
txtNorm_pipeline = pdal.Pipeline(txtNorm_json)
txtNorm_count = txtNorm_pipeline.execute()
txtNorm_arrays = txtNorm_pipeline.arrays
txtNorm_metadata = txtNorm_pipeline.metadata
txtNorm_log = txtNorm_pipeline.log

#----------------------Cylinder time---------------------
radius = 0.5 # cylinder radius in point cloud units
length = 2 # cylinder length in point cloud units
maxDist = np.sqrt((radius**2) + (length / 2)**2)

normalsDf = pd.read_csv('./data/normals.txt')
normalPoint = normalsDf.to_numpy()

x = normalPoint[500,0] # For no just doing it at a single point, but will have to iterate through all of them later
y = normalPoint[500,1]
z = normalPoint[500,2]
xNorm = normalPoint[500,3]
yNorm = normalPoint[500,4]
zNorm = normalPoint[500,5]

# Get all points that could possibly be within the cylinder
dists = np.zeros(np.shape(xyz[0,:]))
dists[:] = np.sqrt((x - xyz[0,:])**2 + (y - xyz[1,:])**2 + (z - xyz[2,:])**2)
inds = np.where(dists <= maxDist)
spherePoints = xyz[:,inds][:,0,:]

# Center sphere points on center point
spherePoints[0,:] += (x * (-1))
spherePoints[1,:] += (y * (-1))
spherePoints[2,:] += (z * (-1))

# Build rotation matrix
zTheta = np.arctan(yNorm / xNorm)
xTheta = np.arctan(zNorm / yNorm)
yTheta = np.arctan(zNorm / xNorm)

roll = -np.arcsin(xNorm)
pitch = np.arcsin(yNorm)


# rot = np.vstack([[np.cos(yTheta)*np.cos(zTheta),(np.cos(xTheta)*np.sin(zTheta))+(np.sin(xTheta)*np.sin(yTheta)*np.cos(zTheta)),(np.cos(xTheta)*np.sin(zTheta))-(np.cos(xTheta)*np.sin(yTheta)*np.cos(zTheta))],
#                  [-np.cos(yTheta)*np.cos(zTheta),(np.cos(xTheta)*np.cos(zTheta))-(np.sin(xTheta)*np.sin(yTheta)*np.sin(zTheta)),(np.sin(xTheta)*np.cos(zTheta))+(np.cos(xTheta)*np.sin(yTheta)*np.sin(zTheta))],
#                  [np.sin(yTheta),-np.sin(xTheta)*np.cos(yTheta),np.cos(xTheta)*np.cos(yTheta)]])

# # Apply rotation
# spherePointsList = list(spherePoints.T)
# rot_spherePoints = np.matmul(spherePointsList[:],rot)

# Rotate about x axis (pitch)
rot_spherePoints = spherePoints
rot_spherePoints[1,:] = ((spherePoints[1,:] * np.cos(pitch)) - 
                         (spherePoints[2,:] * np.sin(pitch)))
rot_spherePoints[2,:] = ((spherePoints[1,:] * np.sin(pitch)) + 
                         (spherePoints[2,:] * np.cos(pitch)))

rot_xNorm = xNorm
rot_yNorm = (yNorm * np.cos(pitch)) - (zNorm * np.sin(pitch))
rot_zNorm = (yNorm * np.sin(pitch)) + (zNorm * np.cos(pitch))

# Rotate about y axis (roll)
rot_spherePoints[0,:] = ((rot_spherePoints[0,:] * np.cos(roll)) + 
                         (rot_spherePoints[2,:] * np.sin(roll)))
rot_spherePoints[1,:] = rot_spherePoints[1,:]
rot_spherePoints[2,:] = ((rot_spherePoints[2,:] * np.cos(roll)) - 
                         (rot_spherePoints[0,:] * np.sin(roll)))

rot_xNorm = (rot_xNorm * np.cos(roll)) + (rot_zNorm * np.sin(roll))
rot_yNorm = rot_yNorm
rot_zNorm = (rot_zNorm * np.cos(roll)) - (rot_xNorm * np.sin(roll))

# Get points within cylinder by getting points within xy circle, then within z bounds
# Xy circle centered on (0,0): x^2 + y^2 = r^2

# circleInds = np.where(np.sqrt(radius**2 - (rot_spherePoints[0,:]**2) >=
#                       np.abs(rot_spherePoints[1,:])))
circleInds = np.where(np.sqrt(rot_spherePoints[0,:]**2 + 
                              rot_spherePoints[1,:]**2) <= radius)
xyz_inCircle = rot_spherePoints[:,circleInds][:,0,:]

cylInds = np.where(np.abs(xyz_inCircle[2,:]) <= length / 2)
xyz_inCyl = xyz_inCircle[:,cylInds]

z_stdev = np.std(xyz_inCyl[2,:]) # This will be our fuzziness








