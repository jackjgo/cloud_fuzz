# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Jack Gonzales.

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU Lesser General Public License as published by  
the Free Software Foundation, version 2.1.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License 
along with this program. If not, see 
http://http://www.gnu.org/licenses
"""

import pdal
import numpy as np
import pandas as pd
import laspy
import os
from tqdm import tqdm
from scipy.signal import find_peaks
from scipy.spatial import KDTree

def downSample(inFile, samplingDist):
    outFile = "./temp/normals.txt"
    json = """
    [
         "%s",
         {
             "type": "filters.smrf"
         },
         {
             "type":"filters.range",
             "limits":"Classification[2:2]"
         },
         {
             "type": "filters.voxelcenternearestneighbor",
             "cell": %s
         },
         {
             "type": "filters.normal",
             "knn": 8
         },
         {
             "type": "writers.text",
             "order": "X,Y,Z,NormalX,NormalY,NormalZ,Curvature",
             "keep_unspecified": "false",
             "filename": "%s"
         }
    ]
    """ % (inFile, samplingDist, outFile)  
    
    #-----------Alternative downsampling pipeline----------
    
    # inFile: input filename (.las or .laz)
    # outFile: file destination for downsampled point cloud with normals (.txt)
    
    # json = """
    # [
    #      "%s",
    #      {
    #          "type": "filters.voxelcenternearestneighbor",
    #          "cell": %s
    #      },
    #      {
    #          "type": "filters.mad",
    #          "dimension": "z",
    #          "k": 2.0
    #      },
    #      {
    #          "type": "filters.voxelcenternearestneighbor",
    #          "cell": %s
    #      },
    #      {
    #          "type": "filters.normal",
    #          "knn": 8
    #      },
    #      {
    #          "type": "writers.text",
    #          "order": "X,Y,Z,NormalX,NormalY,NormalZ,Curvature",
    #          "keep_unspecified": "false",
    #          "filename": "%s"
    #      }
    # ]
    # """ % (inFile, (samplingDist / 10), samplingDist, outFile)
    pipeline = pdal.Pipeline(json)
    count = pipeline.execute()
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log
    return

def sphereSearch(x,y,z,points,radius):
    # Takes a point's xyz coordinates, then calculates the distance to this 
    # point from every point in array points, and returns all points within
    # distance radius.
    # Nota Bene: this is now obselete as the KD Tree search is much faster
    # x: center point x coordinate
    # y: center point y coordinate
    # z: center point z coordinate
    # points: array of all xyz coordinates in a point cloud
    # radius: radius of spherical search
    
    dists = np.zeros(np.shape(points[0,:]))
    dists[:] = np.sqrt((x - points[0,:])**2 + 
                       (y - points[1,:])**2 + 
                       (z - points[2,:])**2)
    inds = np.where(dists <= radius)
    spherePoints = points[:,inds][:,0,:]
    
    return spherePoints

def cylinderSearch(x,y,z,xNorm,yNorm,zNorm,tree,points,radius,length):
    # Returns points within a cylinder of given dimensions centered on xyz and
    # aligned with the normal vector. Returned point coordinates are in a 
    # modified space, with the origin at xyz and the z axis aligned with the
    # normal vector.
    # x: center point x coordinate
    # y: center point y coordinate
    # z: center point z coordinate
    # xNorm: Normal vector x component
    # yNorm: Normal vector y component
    # zNorm: Normal vector z component
    # tree: KD tree containing all the points for accelerated cylinder search
    # points: array of all xyz coordinates in a point cloud
    # radius: radius of cylinder
    # length: length of cylinder
    
    #--------Search for all possible cylinder points--------
    maxDist = np.sqrt(radius**2 + (length / 2)**2) # The maximum distance a 
    # point can be from the center and still be within the cylinder
    spherePointInds = tree.query_ball_point([x,y,z], maxDist)
    spherePoints = points[:,spherePointInds]
    
    #-------------------Set origin to xyz-------------------
    spherePoints[0,:] += (x * (-1))
    spherePoints[1,:] += (y * (-1))
    spherePoints[2,:] += (z * (-1))

    #----Rotate points to align z axis with normal vector---
    roll = -np.arcsin(xNorm)
    pitch = np.arcsin(yNorm)
    
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
    
    #--------------Get points within cylinder---------------
    circleInds = np.where(np.sqrt(rot_spherePoints[0,:]**2 + 
                                  rot_spherePoints[1,:]**2) <= radius)
    xyz_inCircle = rot_spherePoints[:,circleInds][:,0,:]

    cylInds = np.where(np.abs(xyz_inCircle[2,:]) <= length / 2)
    xyz_inCyl = xyz_inCircle[:,cylInds]
    
    return xyz_inCyl

def fuzzStdev(x,
              y,
              z,
              xNorm,
              yNorm,
              zNorm,
              tree,
              points,
              radius,
              length,
              layer_thickness):
    # Returns the standard deviation of points' position along the cylinder 
    # axis and number of peaks
    # x: center point x coordinate
    # y: center point y coordinate
    # z: center point z coordinate
    # xNorm: Normal vector x component
    # yNorm: Normal vector y component
    # zNorm: Normal vector z component
    # tree: KD tree containing all the points for accelerated cylinder search
    # points: xyz coordinates of full point cloud
    # radius: radius of cylinder
    # length: length of cylinder
    
    cylXYZ = cylinderSearch(x,y,z,xNorm,yNorm,zNorm,tree,points,radius,length)
    zStd = np.std(cylXYZ[2,:])
    pointCount = np.shape(cylXYZ[1,:])[1]
    numBins = int(length / layer_thickness)
    histo = np.histogram(cylXYZ[2,:],bins=numBins) # Increase the resolution of
    # the layer search by increasing the number of bins
    return zStd, histo, pointCount

def cloud_fuzz(inFile,  
               outFile, 
               samplingDist, 
               radius, 
               length, 
               layer_thickness = 0.05,
               dist = 2,
               prom = 5):
    # inFile: input point cloud
    # normalsFile: destination for downsampled normals point cloud
    # samplingDist: downsampling distance
    # outFile: destination for downsampled point cloud with fuzz values
    # radius: radius of the cylinder in point cloud units. better results tend
    # to come from using larger radius, but the radius should be appropriate
    # for the terrain.
    # length: length of the cylinder in point cloud units
    # layer_thickness: thickness of each cylinder layer used to identify
    # ghosting layers
    # dist: Distance argument for find_peaks (see scipy docs)
    # prom: prominence aregument for find_peaks (seescipy docs)

    #--------------------Load point cloud-------------------
    fullCloud = laspy.read(inFile)
    Xs = fullCloud.x
    Xs = ((Xs.array * Xs.scale) + Xs.offset)
    Ys = fullCloud.y
    Ys = ((Ys.array * Ys.scale) + Ys.offset)
    # Ys = Ys[::10]
    Zs = fullCloud.z
    Zs = ((Zs.array * Zs.scale) + Zs.offset)
    points = np.vstack([[Xs],[Ys],[Zs]])  
    
    #----------------------Build KD Tree--------------------
    tree = KDTree(points.T)
    
    #-----------------------Downsample----------------------
    os.mkdir('./temp/')
    downSample(inFile, samplingDist)
    normalsFile = './temp/normals.txt'
    df = pd.read_csv(normalsFile)
    downCloud = df.to_numpy().T
    os.remove(normalsFile)
    os.rmdir('./temp/')
    
    #------------------Calculate Deviations-----------------
    zStds = np.zeros(np.shape(downCloud[0,:]))
    numPeaks = np.zeros(np.shape(downCloud[0,:]))
    maxDist = np.zeros(np.shape(downCloud[0,:]))
    numPoints = np.zeros(np.shape(downCloud[0,:]))
    def calcDev(i):
        zStds[i], histo, pointCount = fuzzStdev(downCloud[0,i], downCloud[1,i], 
                                                downCloud[2,i], downCloud[3,i], 
                                                downCloud[4,i], downCloud[5,i], 
                                                tree, points, radius, 
                                                length,layer_thickness)
        peaks = find_peaks(histo[0])[0] # "height" parameter can reduce noise
        numPeaks[i] = np.shape(peaks)[0]
        numPoints[i] = pointCount

        if numPeaks[i] > 1:
            maxDist[i] = (max(peaks) - min(peaks)) * layer_thickness
        return

    for i in tqdm(range(0,(np.shape(downCloud[1,:])[0])), 
                  position=0, 
                  leave=True):
        calcDev(i)
        
        
    #-------------------Write output file-------------------
    fuzzCloud = np.zeros([7,(np.shape(downCloud[1,:])[0])])
    fuzzCloud[0:3,:] = downCloud[0:3,:]
    fuzzCloud[3,:] = zStds
    fuzzCloud[4,:] = numPeaks
    fuzzCloud[5,:] = maxDist
    fuzzCloud[6,:] = numPoints
    fuzzCloud = fuzzCloud.T
    outDf = pd.DataFrame(fuzzCloud, columns=['X',
                                             'Y',
                                             'Z',
                                             'fuzz',
                                             'layers',
                                             'layer distance',
                                             'num points'])
    outDf.to_csv(outFile,index=False)
    
    return

#------------------------Example------------------------
# cloud_fuzz('./data/test_section.las',
#             './data/output.csv',
#             0.5,
#             2,
#             1,
#             layer_thickness=0.1)
