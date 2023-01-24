import numpy as np
import ami.graph_nodes as gn
from ami.flowchart.library.common import CtrlNode
from ami.flowchart.Node import Node
from amitypes import Array1d, Array2d
from numpy import *
from scipy import optimize


class COM_Cent(CtrlNode):
    """
    Returns the centre of mass of array img sliced to look in
    cenrange_x and _y
    """

    nodeName = "COM_Cent"
    uiTemplate = [('xmin', 'intSpin', {'value': 400, 'min': 1}),
                  ('xmax', 'intSpin', {'value': 600, 'min': 1}),
                  ('ymin', 'intSpin', {'value': 400, 'min': 1}),
                  ('ymax', 'intSpin', {'value': 600, 'min': 1})]

    def __init__(self, name):
        super().__init__(name, terminals={'Image': {'io': 'in', 'ttype': Array2d},
                                          'x': {'io': 'out', 'ttype': float},
                                          'y': {'io': 'out', 'ttype': float}},
                         allowAddInput=True)

    def to_operation(self, inputs, outputs, **kwargs):
        xmin = self.values['xmin']
        xmax = self.values['xmax']
        ymin = self.values['ymin']
        ymax = self.values['ymax']

        def COM_Cent(img):
           cenrange_x = (xmin, xmax)
           cenrange_y = (ymin, ymax)
           img = img[slice(*cenrange_x), slice(*cenrange_y)]
           cum_mass = 0
           x_cum = 0
           y_cum = 0
           for i in range(img.shape[0]):
               for j in range(img.shape[1]):
                   cum_mass += img[i, j]
                   x_cum += img[i, j] * i
                   y_cum += img[i, j] * j
            roi_com = (x_cum/cum_mass, y_cum/cum_mass)
            x = roi_com[1] + cenrange_x[0]
            y = roi_com[0] + cenrange_y[0]


            return x, y

        return gn.Map(name=self.name() + '_operation', inputs=inputs,
                      outputs=outputs, func=CoordinateFind, **kwargs)


