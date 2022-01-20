# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 09:47:46 2021

@author: jgonzales
"""

import pdal
import numpy as np
import pandas as pd
import laspy
from tqdm import tqdm
# from joblib import Parallel, delayed
from scipy.signal import find_peaks

def downSample(inFile, outFile, samplingDist):
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

def cylinderSearch(x,y,z,xNorm,yNorm,zNorm,points,radius,length):
    # Returns points within a cylinder of given dimensions centered on xyz and
    # aligned with the normal vector. Returned point coordinates are in a 
    # modified space, with the origin at xyz and the z axis aligned with the
    # normal vector
    # x: center point x coordinate
    # y: center point y coordinate
    # z: center point z coordinate
    # xNorm: Normal vector x component
    # yNorm: Normal vector y component
    # zNorm: Normal vector z component
    # points: array of all xyz coordinates in a point cloud
    # radius: radius of cylinder
    # length: length of cylinder
    
    #--------Search for all possible cylinder points--------
    maxDist = np.sqrt(radius**2 + (length / 2)**2) # The maximum distance a 
    # point can be from the center and still be within the cylinder
    spherePoints = sphereSearch(x,y,z,points,maxDist)
    
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

def fuzzStdev(x,y,z,xNorm,yNorm,zNorm,points,radius,length,layer_thickness):
    # Returns the standard deviation of points' position along the cylinder 
    # axis and number of peaks
    # x: center point x coordinate
    # y: center point y coordinate
    # z: center point z coordinate
    # xNorm: Normal vector x component
    # yNorm: Normal vector y component
    # zNorm: Normal vector z component
    # points: xyz coordinates of full point cloud
    # radius: radius of cylinder
    # length: length of cylinder
    
    cylXYZ = cylinderSearch(x,y,z,xNorm,yNorm,zNorm,points,radius,length)
    zStd = np.std(cylXYZ[2,:])
    numBins = int(length / layer_thickness)
    histo = np.histogram(cylXYZ[2,:],bins=numBins) # Increase the resolution of the 
    # layer search by increasing the number of bins
    return zStd, histo

def cloud_fuzz(inFile, 
               normalsFile, 
               outFile, 
               samplingDist, 
               radius, 
               length, 
               layer_thickness = 0.05,
               dist = 2,
               prom = 20):
    # inFile: input point cloud
    # normalsFile: destination for downsampled normals point cloud
    # samplingDist: downsampling distance
    # outFile: destination for downsampled point cloud with fuzz values
    # radius: radius of the cylinder in point cloud units
    # length: length of the cylinder in point cloud units
    # layer_thickness: thickness of each cylinder layer used to identify
    # ghosting layers
    # dist: Distance argument for find_peaks (see scipy docs)
    # prom: prominence aregument for find_peaks (seescipy docs)

    fullCloud = laspy.read(inFile)
    Xs = fullCloud.x
    Xs = ((Xs.array * Xs.scale) + Xs.offset)
    Ys = fullCloud.y
    Ys = ((Ys.array * Ys.scale) + Ys.offset)
    # Ys = Ys[::10]
    Zs = fullCloud.z
    Zs = ((Zs.array * Zs.scale) + Zs.offset)
    points = np.vstack([[Xs],[Ys],[Zs]])    
    
    #-----------------------Downsample----------------------
    downSample(inFile, normalsFile, samplingDist)
    df = pd.read_csv(normalsFile)
    downCloud = df.to_numpy().T
    
    #------------------Calculate Deviations-----------------
    zStds = np.zeros(np.shape(downCloud[0,:]))
    numPeaks = np.zeros(np.shape(downCloud[0,:]))
    maxDist = np.zeros(np.shape(downCloud[0,:]))
    def calcDev(i):
        global histo
        zStds[i], histo = fuzzStdev(downCloud[0,i], downCloud[1,i], 
                                    downCloud[2,i], downCloud[3,i], 
                                    downCloud[4,i], downCloud[5,i], 
                                    points, radius, length,layer_thickness)
        peaks = find_peaks(histo[0],distance=dist,prominence=prom)[0]
        print(peaks)
        numPeaks[i] = np.shape(peaks)[0]
        if numPeaks[i] > 1:
            maxDist[i] = (max(peaks) - min(peaks)) * layer_thickness
        return
    

    for i in tqdm(range(0,(np.shape(downCloud[1,:])[0])), 
                  position=0, 
                  leave=True):
        calcDev(i)
        
        
    #-------------------Write output file-------------------
    fuzzCloud = np.zeros([6,(np.shape(downCloud[1,:])[0])])
    fuzzCloud[0:3,:] = downCloud[0:3,:]
    fuzzCloud[3,:] = zStds
    fuzzCloud[4,:] = numPeaks
    fuzzCloud[5,:] = maxDist
    fuzzCloud = fuzzCloud.T
    outDf = pd.DataFrame(fuzzCloud, columns=['X',
                                             'Y',
                                             'Z',
                                             'fuzz',
                                             'layers',
                                             'layer distance'])
    outDf.to_csv(outFile,index=False)
    
    return

#------------------------Example------------------------
# cloud_fuzz('./data/test1.las',
#             './data/normals.txt',
#             './data/output4.csv',
#             10,
#             0.5,
#             1,
#             layer_thickness=0.01)