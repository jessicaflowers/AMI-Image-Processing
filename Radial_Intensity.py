import numpy as np
import ami.graph_nodes as gn
from ami.flowchart.library.common import CtrlNode
from ami.flowchart.Node import Node
from amitypes import Array1d, Array2d
from numpy import *
from scipy import optimize
import math
from psana.pyalgos.generic.HBins import HBins
import ami.flowchart.library.UtilsROI as ur

class RadInten(CtrlNode):
    """
        Region of Interest of image shaped as arch (a.k.a. cut-donat).
        """
        nodeName = "RadInten"
        uiTemplate = [('center x',  'intSpin', {'value': 200, 'min': -1000}),
                      ('center y',  'intSpin', {'value': 200, 'min': -1000}),
                      ('radius o',  'intSpin', {'value': 200, 'min': 1}),
                      ('radius i',  'intSpin', {'value': 100, 'min': 1}),
                      ('angdeg o',  'intSpin', {'value':   0, 'min': 0, 'max': 360}),
                      ('angdeg i',  'intSpin', {'value':  60, 'min': 0, 'max': 360}),
                      ('nbins rad', 'intSpin', {'value': 100, 'min': 1}),
                      ('nbins ang', 'intSpin', {'value':   5, 'min': 1})]


    def __init__(self, name):
        super().__init__(name,
                         terminals={'image': {'io': 'in', 'ttype': Array2d},
                                    'RadAngNormIntens': {'io': 'out', 'ttype': Array2d}},
                         global_op=True,
                         viewable=True)


    def to_operation(self, inputs, outputs, **kwargs):
        radedges = np.array((self.values['ri'], self.values['ro']))
        nradbins = self.values['nr']
        phiedges = np.array((self.values['ao'], self.values['ai']))
        nphibins = self.values['na']

        def RadInten(image):
            shape = np.shape(image)
            rows, cols = shape
            xarr1 = np.arange(cols) - cx
            yarr1 = np.arange(rows) - cy
            xarr, yarr = np.meshgrid(xarr1, yarr1)

            rad, phi0 = cart2polar(xarr, yarr)
            shapeflat = (rad.size,)
            rad.shape = shapeflat
            phi0.shape = shapeflat
            phimin = min(phiedges[0], phiedges[-1])
            phi = np.select((phi0 < phimin, phi0 >= phimin), (phi0 + 360., phi0))
            phi1 = phiedges[0]
            phi2 = phiedges[-1]
            rb = _set_rad_bins(radedges, nradbins)
            pb = _set_phi_bins(phiedges, nphibins)

            npbins = pb[1]
            nrbins = rb[1]
            ntbins = npbins * nrbins  # total number of bins in r-phi array

            # bin index phi
            indpmin, indpmax = (-1, npbins)
            factorp = float(npbins) / (phiedges[-1] - phiedges[0])
            npbins1 = npbins - 1
            npparr = (np.array(phi) - phiedges[0]) * factorp
            ind_p = np.array(np.floor(npparr), dtype=np.int32)
            iphi = np.select((ind_p < 0, ind_p > npbins1), (indpmin, indpmax), default=ind_p)

            # bin index rad
            indrmin, indrmax = (-1, nrbins)
            factorr = float(nrbins) / (radedges[-1] - radedges[0])
            nrbins1 = nrbins - 1
            nrparr = (np.array(rad) - radedges[0]) * factorr
            ind_r = np.array(np.floor(nrparr), dtype=np.int32)
            irad = np.select((ind_r < 0, ind_r > nrbins1), (indrmin, indrmax), default=ind_r)

            # condition
            cond = np.logical_and( \
                np.logical_and(irad > -1, irad < nrbins),
                np.logical_and(iphi > -1, iphi < npbins)
            )

            cond = np.logical_and(cond, mask.astype(np.bool_).ravel())

            iseq = np.select((cond,), (iphi * nrbins + rad,), ntbins).ravel()
            iseq = iseq.astype(np.int64)
            npix_per_bin = np.bincount(iseq, weights=None, minlength=ntbins + 1)

            # bin intensity
            bin_intensity = np.bincount(iseq, weights=image.ravel(), minlength=ntbins + 1)

            # bin average
            num = bin_intensity
            den = npix_per_bin
            bin_avrg = divide_protected(num, den, vsub_zero=0)

            # bin avg phi
            arr_rphi = bin_avrg[:-1]  # -1 removes off ROI bin
            arr_rphi.shape = (npbins, nrbins)
            arr_rphi = arr_rphi[~np.isnan(arr_rphi)]

            return arr_rphi

        return gn.Map(name=self.name() + '_operation', inputs=inputs,
                      outputs=outputs, func=RadInten, **kwargs)


