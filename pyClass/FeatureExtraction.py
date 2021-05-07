from skimage import feature
import numpy as np

class LocalBinaryPattern:
    def __init__(self, numPoints, radius, method='nri_uniform'):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.method = method
        
    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method=self.method)
        unique = np.unique(lbp.ravel())
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, unique[-1]+2), range=(0, unique[-1]+1))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist