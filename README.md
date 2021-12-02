# cloud_fuzz
Finds the "fuzziness" of a surface in a point cloud, such as that derived from structure from motion photogrammetry.

This is done by first downsampling the point cloud, and finding a vector normal to the surface at each downsampled point.
Then, a cylinder of a given size is projected along the normal vector and centered on the downsampled point. Points in the
full resolution point coud are captured by the cylinder, and the fuzziness is defined by the standard deviation of these
points' positions along the cylinder axis.

Ghosting layers can also be identified by dividing the cylinder into slices, and counting how many points lie wihtin each 
slice. Those with higher point counts than their neighbors indicate a ghosting layer.

It is important to note that if there is more than one layer, the points will have a multimodal distribution and standard
deviation (and thus fuzziness) will no longer be useful.
