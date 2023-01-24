import numpy as np
import ami.graph_nodes as gn
from ami.flowchart.library.common import CtrlNode
from ami.flowchart.Node import Node
from amitypes import Array1d, Array2d
from numpy import *
from scipy import optimize
import imageio as iio
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops


class DiffractionCent(CtrlNode):
    """
    Returns the center of mass of array img
    """

    nodeName = "DiffractionCent"

    def __init__(self, name):
        super().__init__(name, terminals={'Image': {'io': 'in', 'ttype': Array2d},
                                          'x': {'io': 'out', 'ttype': float},
                                          'y': {'io': 'out', 'ttype': float}},
                         allowAddInput=True)

    def to_operation(self, inputs, outputs, **kwargs):

        def DiffractionCent(image):
            threshold_value = filters.threshold_otsu(image)
            labeled_foreground = (image > threshold_value).astype(int)
            properties = regionprops(labeled_foreground, image)
            center_of_mass = properties[0].centroid
            weighted_center_of_mass = properties[0].weighted_centroid
            x = weighted_center_of_mass[0]
            y = weighted_center_of_mass[1]
            return x, y

        return gn.Map(name=self.name() + '_operation', inputs=inputs,
                      outputs=outputs, func=DiffractionCent, **kwargs)


