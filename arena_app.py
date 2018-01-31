#!/usr/bin/env python

import argparse
import subprocess
import signal
import cv2
import sys
import numpy as np
import math

#import scipy.signal as sps

from PyQt4 import QtGui, QtCore
#import pyqtgraph as pg
#import arena_app_auto
import random
from collections import OrderedDict
from collections import deque
import time
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy import stats
import matplotlib
import scipy

from six import string_types
#import dm

# Profiling stuff
#import cProfile

class G(object):
    # Camera resolution
    full_arena = True
    if full_arena:
        xsize = 1280
        ysize = 720
        #xsize = 640
        #ysize = 480
    else:
        xsize = 640
        ysize = 360

    # Region of interest, only apply image processing within this area
    # (bottomleft_x, bottomleft_y, topright_x, topright_y)
    roi = (300,0,960,650)

    # Timeout to grab new frame
    grab_interval = 100

    # Camera distortion correction
    camera_matrix = np.array([
        [1.0,   0.0,    0.0],
        [0.0,   1.0,    0.0],
        [0.0,   0.0,    1.0]])
    camera_dist_coeffs = np.array([ 0.0,    0.0,    0.0,    0.0,    0.0])

    # Kilobot detection and tracking
    min_area    = 10
    max_area    = 75
    min_threshold = 100
    max_threshold = 190

    # LED detection for food counting
    led_min_area    = 1
    led_max_area    = 25
    led_min_threshold = 100
    led_max_threshold = 255

    # Number of samples to make for food counting
    # Each sample is at 100ms intervals, each bit
    # of data is at ~500ms intervals, so 100 should
    # give two full byte cycles
    fsample_size    = 200

    # Distance below which an led is counted as belonging to a kilobot
    critical_dist = 10

    # Age above which a trail is ignored
    critical_time = 1

    # Timestep
    dt = 0.1

    #===============================
    # Some UI defines
    buttonxsize = 4
    buttonysize = 4
    buttonstate = ['STATE:idle','STATE:track','STATE:count','STATE:finished']
    button_array = [['ex+',     'sat+',     'min_t+',        'max_t+'],
                    ['ex-',     'sat-',     'min_t-',        'max_t-'],
                    [buttonstate[0],   'f',        'min_a+',        'max_a+'],
                    ['quit',       'j',     'min_a-',        'max_a-']]


# Run command in shell
def cmdp(s):
    return subprocess.Popen(s, shell=True, stdout = subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]

def circles(x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data scale (ie. in data unit)
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or
        RGBA sequence because that is indistinguishable from an array of
        values to be colormapped.  `c` can be a 2-D array in which the
        rows are RGB or RGBA, however.
    ax : Axes object, optional, default: None
        Parent axes of the plot. It uses gca() if not specified.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.  (Note if you pass a `norm` instance, your
        settings for `vmin` and `vmax` will be ignored.)

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Other parameters
    ----------------
    kwargs : `~matplotlib.collections.Collection` properties
        eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    #http://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker/24567352#24567352
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    import pylab as plt
    #import matplotlib.colors as colors

    if ax is None:
        ax = plt.gca()

    if isinstance(c,string_types):
        color = c     # ie. use colors.colorConverter.to_rgba_array(c)
    else:
        color = None  # use cmap, norm after collection is created
    kwargs.update(color=color)

    if np.isscalar(x):
        patches = [Circle((x, y), s),]
    elif np.isscalar(s):
        patches = [Circle((x_,y_), s) for x_,y_ in zip(x,y)]
    else:
        patches = [Circle((x_,y_), s_) for x_,y_,s_ in zip(x,y,s)]
    collection = PatchCollection(patches, **kwargs)

    if color is None:
        collection.set_array(np.asarray(c))
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    return collection


class Param(object):
    def __init__(self, min, max, value):
        self.min    = min
        self.max    = max
        self.value  = value
        self.changed = True
    def inc(self):
        if self.value < self.max:
            self.value += 1
            self.changed = True
    def dec(self):
        if self.value > self.min:
            self.value -= 1
            self.changed = True
    def scaleup(self):
        if self.value*1.2 < self.max:
            self.value = int(self.value * 1.2)
            self.changed = True
    def scaledn(self):
        if self.value/1.2 > self.min:
            self.value = int(self.value / 1.2)
            self.changed = True


class Webcam(object):
    def __init__(self):
        # Set up the webcam parameter data. Put in an orderedDict to guarantee
        # that when we iterate over it, the parameters are always enacted
        # in the same order
        self.params = OrderedDict([
            ('brightness'                       , Param(-64, 54, 16)),
            ('contrast'                         , Param(0, 95, 24)),
            ('saturation'                       , Param(0, 100, 60)),
            ('hue'                              , Param(-180, 180, 0)),
            ('white_balance_temperature_auto'   , Param(0, 1, 0)),
            ('gamma'                            , Param(48, 300, 100)),
            ('power_line_frequency'             , Param(0, 2, 1)),
            ('white_balance_temperature'        , Param(2800, 6500, 4600)),
            ('sharpness'                        , Param(1, 7, 3)),
            ('backlight_compensation'           , Param(0, 4, 0)),
            ('exposure_auto'                    , Param(0, 3, 1)),
            ('exposure_absolute'                , Param(10, 1250, 280))
            ])
        self.frame = np.zeros((G.ysize, G.xsize, 3))
        # Start up the webcam, raw frame data will be available on the stdout pipe
        # This is a gstreamer pipeline that constrains the webcam to output
        # its maximum resolution in YUV422 at the max possible framerate of 10Hz.
        # This is then converted to BGR, suitable for OpenCV
        # The full capability is given by:
        # v4l2-ctl --list-formats-ext
        self.pipe = subprocess.Popen(
            'gst-launch-1.0 -e -q v4l2src !'
            'video/x-raw,width=%s,height=%s,framerate=10/1,format=YUY2 !'
            'videoflip method=rotate-180 !'
            'videoconvert !'
            'video/x-raw,format=BGR  !'
            'filesink location=/dev/stdout sync=false async=false ' % (G.xsize, G.ysize),
            shell=True, stdout = subprocess.PIPE, stderr = None, bufsize=400000)
        # Set the controllable parameters to our desired values
        for k,v in self.params.iteritems():
            print(k, v.value)
            cmdp('v4l2-ctl -c %s=%s' % (k, v.value))
        # It seems to be necessary to wait for a bit after the camera has started
        # before setting the first exposure value, otherwise the picure seems desaturated
        # and more underexposed than would be expected
        time.sleep(2)

    def grabframe(self):
        raw_image = self.pipe.stdout.read(int(G.xsize * G.ysize * 3))
        # transform the byte read into a numpy array
        image =  np.fromstring(raw_image, dtype='uint8')
        self.frame = image.reshape((G.ysize, G.xsize, 3))
        # throw away the data in the pipe's buffer.
        self.pipe.stdout.flush()
        # return the frame in RGB form
        return self.frame[:,:,::-1]

    def setparam(self, k, v):
        try:
            param = self.params[k]
            if v < param.min: v = param.min
            if v > param.max: v = param.max
            param.value = v
            cmdp('v4l2-ctl -c %s=%s' % (k, v))
        except Exception as e:
            print(e)

    def updateparams(self):
        # check if any parameters have changed and apply them
        for k,v in self.params.iteritems():
            print(k, v.value)
            if v.changed:
                cmdp('v4l2-ctl -c %s=%s' % (k, v.value))
                v.changed = False


#=====================================================================
# http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
#=====================================================================
class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=10, height=5.63, dpi=100, imgxsize=0, imgysize=0):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        self.compute_initial_figure(imgxsize, imgysize)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        sizePolicy =QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        FigureCanvas.setSizePolicy(self, sizePolicy)
        FigureCanvas.updateGeometry(self)


    def compute_initial_figure(self, imgxsize, imgysize):
        pass


class MPLImage(MyMplCanvas):
    def compute_initial_figure(self, imgxsize, imgysize):
        # Set up with no axis or surrounding whitespace
        self.axes.set_position([0,0,1,1])
        self.axes.set_frame_on(False)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        # This sets the imshow area up, colormap and interpolation defined
        # here. Susequent updates just change the data contained in the imshow.
        # The colormap, and scaling only applies to non RGB data
        self.im = self.axes.imshow(
            np.random.poisson(1., (imgysize, imgxsize)),
            interpolation='nearest',
            cmap='gray', vmin=0.0, vmax=1.0
            )

    def update_data(self, data):
        self.im.set_data(data)
        self.draw()

class MPLImageAndPlot(MyMplCanvas):
    def compute_initial_figure(self, imgxsize, imgysize):
        # Set up with no axis or surrounding whitespace
        self.axes.set_position([0,0,1,1])
        self.axes.set_frame_on(False)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        # This sets the imshow area up, colormap and interpolation defined
        # here. Susequent updates just change the data contained in the imshow.
        # The colormap, and scaling only applies to non RGB data
        self.im = self.axes.imshow(
            #np.random.poisson(1., (imgysize, imgxsize)),
            np.zeros((imgysize, imgxsize)),
            interpolation='nearest',
            cmap='gray', vmin=0.0, vmax=1.0, origin='upper'
            )

        #pax = self.fig.add_axes([0,0,1,1])
        #pax.set_axis_off()
        #pax.plot(np.sin(np.linspace(0,20)))
        self.axes.set_ylim(imgysize,0)
        self.axes.set_xlim(0,imgxsize)
        self.axes.hold(True)
        self.kiloc = circles(0, 0, s=6,  ax=self.axes, facecolor='none', edgecolor='green')
        self.emitc = circles(0, 0, s=5,  ax=self.axes, facecolor='blue', edgecolor='none')
        self.ledc = circles(0, 0, s=3,  ax=self.axes, facecolor='none', edgecolor='red')

        self.roibox, = self.axes.plot([G.roi[0],G.roi[2],G.roi[2],G.roi[0],G.roi[0]],
                                    [G.roi[1],G.roi[1],G.roi[3],G.roi[3],G.roi[1]])

    def update_data(self, data):
        self.im.set_data(data)
        #self.draw()

    def update_roi(self, p):
        self.roibox.set_data([p[0],p[2],p[2],p[0],p[0]],[p[1],p[1],p[3],p[3],p[1]])

    def update_ledcircles(self, points):
        # Apply the offsets in data coordinate space
        self.ledc.set_offset_position('data')
        self.ledc.set_offsets(points)

    def update_kilocircles(self, points):
        # Apply the offsets in data coordinate space
        self.kiloc.set_offset_position('data')
        self.kiloc.set_offsets(points)

    def update_emitcircles(self, points):
        # Apply the offsets in data coordinate space
        self.emitc.set_offset_position('data')
        #print points, points.shape
        if points.shape[0] == 0:
            # Hacky, not sure why, but a zero length array puts a blob at the origin
            points = np.array([[-1000,-1000]])
        self.emitc.set_offsets(points)

    def redraw(self):
        self.draw()


class UI(object):
    def __init__(self, parent, bhandler):

        self.mainlayout             = QtGui.QVBoxLayout()
        self.vidlayout              = QtGui.QHBoxLayout()
        self.ctrllayout             = QtGui.QHBoxLayout()
        self.ctrlgrid               = QtGui.QGridLayout()
        self.statuslayout           = QtGui.QVBoxLayout()
        self.bannerlayout           = QtGui.QVBoxLayout()
        self.mainlayout.addLayout(self.vidlayout)
        self.mainlayout.addLayout(self.bannerlayout)
        self.mainlayout.addLayout(self.ctrllayout)
        self.ctrllayout.addLayout(self.ctrlgrid)
        self.ctrllayout.addLayout(self.statuslayout)
        parent.setLayout(self.mainlayout)

        # test matplotlib
        #self.img_raw                = MPLImage(imgxsize=G.xsize, imgysize=G.ysize)
        self.img_postproc           = MPLImageAndPlot(imgxsize=G.xsize, imgysize=G.ysize)

        #self.vidlayout.addWidget(self.graphicsview_raw)
        #self.vidlayout.addWidget(self.graphicsview_postproc)
        #self.vidlayout.addWidget(self.img_raw)
        self.vidlayout.addWidget(self.img_postproc)

        self.status = []
        for i in xrange(8):
            s = (QtGui.QLabel(parent))
            s.setText('<pre><font size=0.6>this is label %d</pre>' % i)
            self.statuslayout.addWidget(s)
            self.status.append(s)

        self.banner = (QtGui.QLabel(parent))
        self.updatebanner('Idle                                    ')
        self.bannerlayout.addWidget(self.banner)

        self.buttons = [[0 for i in xrange(G.buttonxsize)] for i in xrange(G.buttonysize)]
        for y in xrange(G.buttonysize):
            for x in xrange(G.buttonxsize):
                b = QtGui.QPushButton(parent)
                b.setText(G.button_array[y][x])
                b.clicked.connect(bhandler)
                self.ctrlgrid.addWidget(b, y, x)
                self.buttons[y][x] = b

    def updatestat(self, i, string):
        self.status[i].setText('<pre><font size=0.6>%s</pre>' % string)

    def updatebanner(self, string):
        #self.banner.setText('<span style=font-family:"Lucida Console";font-size:50%%>%s</span>' % string)
        self.banner.setText('<span style=font-family:monospace;font-size:30pt>%s</span>' % string)


class Extract(object):
    #-----------------------------------------------------------------
    # Process incoming images to locate kilobots and red LEDs
    #-----------------------------------------------------------------
    def __init__(self):
        self.minThreshold = 10
        self.maxThreshold = 100
        self.leds = np.array([])
        self.kilohistory = deque()
        self.timeptr = 0

    def detect_peaks(self, image):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        #http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2,2)

        #apply the local maximum filter; all pixel of maximal value
        #in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood)==image
        #local_max is a mask that contains the peaks we are
        #looking for, but also the background.
        #In order to isolate the peaks we must remove the background from the mask.

        #we create the mask of the background
        background = (image==0)

        #a little technicality: we must erode the background in order to
        #successfully subtract it form local_max, otherwise a line will
        #appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        #we obtain the final mask, containing only peaks,
        #by removing the background from the local_max mask
        detected_peaks = local_max - eroded_background
        return detected_peaks
        #return local_max

    def find_red(self, f):
        # Work in HSV space
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        # Red is centred around zero hue, on a scale that goes from 0-179
        # Have quite a wide range of saturation, but only work on bright objects
        lower_red1 = np.array([0,70,200])
        upper_red1 = np.array([10,255,255])
        lower_red2 = np.array([160,70,200])
        upper_red2 = np.array([179,255,255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = np.float32(mask1 + mask2) / 255.0
        res = cv2.GaussianBlur(mask, ksize=(0,0), sigmaX=1)
        self.peaks = self.detect_peaks(res)

    def find_green(self, f):
        # Work in HSV space
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        # Red is centred around zero hue, on a scale that goes from 0-179
        # Have quite a wide range of saturation, but only work on bright objects
        lower_green = np.array([40,50,230])
        upper_green = np.array([80,255,255])
        self.greenmask = cv2.inRange(hsv, lower_green, upper_green)
        mask = np.float32(self.greenmask) / 255.0
        res = cv2.GaussianBlur(mask, ksize=(0,0), sigmaX=1)
        self.peaks = self.detect_peaks(res)

    def get_leds(self):
        # Find a numpy array of points in x which are non-zero
        pointsy, pointsx =  np.where(self.peaks)
        self.leds = np.hstack((pointsx[np.newaxis].T, pointsy[np.newaxis].T)).astype(np.float32)
        #print self.points


    def run_detectredleds(self, f):
        self.find_red(f)
        self.get_leds()

    def run_detectgreenleds(self, f):
        self.find_green(f)
        self.get_leds()
        print(self.leds)

    def run_detectleds(self, img):
        #http://stackoverflow.com/questions/9860667/writing-robust-color-and-size-invariant-
        #circle-detection-with-opencv-based-on
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # make areas outside roi into black
        grey[0:G.roi[1],        0:]         = 0
        grey[G.roi[1]:G.roi[3], 0:G.roi[0]] = 0
        grey[G.roi[1]:G.roi[3], G.roi[2]:]  = 0
        grey[G.roi[3]:,         0:]         = 0

        thr = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -80)
        #cv2.imshow('test',thr)

        self.gf32 = np.float32(thr)/255.0
        #detector = cv2.FeatureDetector_create('MSER')
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold         = G.led_min_threshold
        params.maxThreshold         = G.led_max_threshold
        params.filterByColor        = 1
        params.blobColor            = 255
        params.filterByArea         = 1
        params.minArea              = G.led_min_area
        params.maxArea              = G.led_max_area
        params.filterByCircularity  = 0
        params.minCircularity       = 0.8
        params.maxCircularity       = 1.1
        params.filterByConvexity    = 0
        params.minConvexity         = 0.5
        params.maxConvexity         = 1.0
        params.filterByInertia      = 0
        params.minInertiaRatio      = 0.8
        params.maxInertiaRatio      = 1.0
        detector = cv2.SimpleBlobDetector(params)
        fs = detector.detect(grey)
        #fs.sort(key=lambda x: -x.size)

        sfs = fs
        self.leds = np.zeros((len(sfs), 2))
        for idx, p in enumerate(sfs):
            self.leds[idx, 0] = p.pt[0]
            self.leds[idx, 1] = p.pt[1]

    def run_detectkilobots(self, img):
        #http://stackoverflow.com/questions/9860667/writing-robust-color-and-size-invariant-
        #circle-detection-with-opencv-based-on
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # make areas outside roi into black
        grey[0:G.roi[1],        0:]         = 0
        grey[G.roi[1]:G.roi[3], 0:G.roi[0]] = 0
        grey[G.roi[1]:G.roi[3], G.roi[2]:]  = 0
        grey[G.roi[3]:,         0:]         = 0




        self.gf32 = np.float32(grey)/255.0
        #detector = cv2.FeatureDetector_create('MSER')
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold         = G.min_threshold
        params.maxThreshold         = G.max_threshold
        params.filterByColor        = 1
        params.blobColor            = 255
        params.filterByArea         = 1
        params.minArea              = G.min_area
        params.maxArea              = G.max_area
        params.filterByCircularity  = 0
        params.minCircularity       = 0.8
        params.maxCircularity       = 1.1
        params.filterByConvexity    = 0
        params.minConvexity         = 0.5
        params.maxConvexity         = 1.0
        params.filterByInertia      = 0
        params.minInertiaRatio      = 0.8
        params.maxInertiaRatio      = 1.0
        detector = cv2.SimpleBlobDetector(params)
        fs = detector.detect(grey)
        #fs.sort(key=lambda x: -x.size)

        sfs = fs
        self.kilobots = np.zeros((len(sfs), 2))
        for idx, p in enumerate(sfs):
            self.kilobots[idx, 0] = p.pt[0]
            self.kilobots[idx, 1] = p.pt[1]

    def transformtoworld(self):
        if self.kilobots.shape[0] > 0:
            self.kiloworld = np.squeeze(cv2.undistortPoints(self.kilobots.reshape((1,-1,2)), G.camera_matrix, G.camera_dist_coeffs), 0)
        else:
            self.kiloworld = np.array([])
        if self.leds.shape[0] > 0:
            self.ledworld = np.squeeze(cv2.undistortPoints(self.leds.reshape((1,-1,2)), G.camera_matrix, G.camera_dist_coeffs), 0)
        else:
            self.ledworld = np.array([])
        self.gf32world = cv2.undistort(self.gf32, G.camera_matrix, G.camera_dist_coeffs)

    def classify(self):
        # find the distances between all the detected LEDs and all the detected kilobots
        # and sort
        #print self.kiloworld.shape, self.ledworld.shape
        # Don't do anything unless there is something in both arrays
        if self.kiloworld.shape[0] == 0 or self.ledworld.shape[0] == 0:
            self.emitters = np.array([])
            return
        # Broadcast each array across a different axis, (leaving the 0 axis
        # with the x and y coordinates in alone) and subtract. This gives a matrix
        # of all the x and y differences
        diff = self.kiloworld[:,None,:] - self.ledworld[None,:,:]
        # Calculate the cartesian distance of each point
        dists = np.hypot(diff[:,:,0], diff[:,:,1])
        # Select from all the kilobots those that have an led within the critical
        # distance. These are out emitters
        self.emitters = self.kiloworld[np.sum(dists < G.critical_dist, 1) > 0, :]

    def run(self, f):
        # Find salient features in the raw image and transform to world coordinate
        # by applying the camera correction
        #self.run_detectredleds(f)
        self.run_detectkilobots(f)
        self.transformtoworld()

        # Add to history list, pop stuff off the end to keep the length
        # no greater than 10
        self.kilohistory.append(self.kiloworld)
        if len(self.kilohistory) > 10:
            self.kilohistory.popleft()


        #print '%s' % (self.kiloworld)

    # def calc_recent_positions(self):
    #     # Run dbscan cluster on history of positions
    #     data = np.concatenate(self.kilohistory)
    #     if len(data) == 0:
    #         self.kilorecent = np.array([])
    #         return
    #     db = skc.DBSCAN(eps=5, min_samples=5).fit(data)
    #     labels = db.labels_
    #     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #     self.kilorecent = np.zeros((n_clusters,2))
    #     for c in xrange(n_clusters):
    #         pos = np.mean(data[labels==c],0)
    #         self.kilorecent[c] = pos


    def init_food_count(self):
        # Start the food count process.
        self.timeptr = 0
        self.onleds = []

    def countfood(self, f):
        # Count food by looking for leds and decoding the binary values of each kilobot
        #
        # Grab frames until we have enough. Each frame, detect the LEDs that are lit
        # by:
        #   1) turning down the exposure so that the leds are not clipping
        #   2) using adaptive thresholding to binarise the image
        #   3) using blob detect on the binarised image to find the locations
        #   4) save the position data of on leds in a list of arrays
        # Once all the frames have been captured, analyse the data by:
        #   1) cluster using DBSCAN, works well in the presence of noise
        #   2) convert each cluster into a time sequence of zeros and ones
        #   3) decode the sequence into a list of numbers for each cluster
        #   4) sum the separate cluster values if, within a cluster, they are the same
        #   to give a total amount of food. If a cluster has no decodeable
        #   number, it is ignored. If a cluster has different values, it is ignored.
        #   This will underestimate the value but should be relatively robust
        #   against corrupt and nonsense values.
        self.run_detectleds(f)
        self.transformtoworld()
        #print '%f %s' % (time.time(), self.ledworld)
        match = []
        if self.ledworld.shape[0] > 0 and self.timeptr < G.fsample_size:
            np.savetxt('led%05d.txt'%self.timeptr, self.ledworld)
            # Add an extra column with the sample index
            print(self.ledworld.shape)
            #print self.ledworld
            n = np.concatenate((self.ledworld, np.ones((self.ledworld.shape[0], 1)) * self.timeptr), 1)
            self.onleds.append(n)
        self.timeptr += 1

        if self.timeptr >= G.fsample_size:
            if len(self.onleds) == 0:
                return True, 0, 0
            else:
                dc = np.concatenate(self.onleds)
                #self.allfood, self.coords, self.goodcoords, msg = dm.totalfood(dc)
                self.valfood = [i for i in self.allfood if i >= 0]
                food = np.sum(self.valfood)
                return True, food, 0
        return False, 0, G.fsample_size - self.timeptr



#------------------------------------------------------------------
# Main app and UI
#------------------------------------------------------------------
class Arena(QtGui.QWidget):
    def __init__(self, args):
        super(Arena, self).__init__()

        # Remember the cmd line args
        self.args = args
        self.state = G.buttonstate[0]

        # Construct the UI
        self.ui = UI(self, self.bhandler)

        # Start the application
        # Set up the webcam and an extraction class, then
        # start a timer which will repreatedly try and grab a frame from the
        # webcam and process it. The timer is set to be faster than the framerate
        # so the app thread will be waiting in grabframe
        self.webcam = Webcam()
        self.extract = Extract()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.grab)
        self.timer.start(G.grab_interval)

        self.last_time = 0
        self.cur_time = 0

    def bhandler(self):
        b = str(self.sender().text())
        {   'ex+'       : self.exup,
            'ex-'       : self.exdn,
            'sat+'      : self.satup,
            'sat-'      : self.satdn,
            'min_t+'    : self.mintup,
            'min_t-'    : self.mintdn,
            'max_t+'    : self.maxtup,
            'max_t-'    : self.maxtdn,
            'min_a+'    : self.minaup,
            'min_a-'    : self.minadn,
            'max_a+'    : self.maxaup,
            'max_a-'    : self.maxadn,
            G.buttonstate[0]     : self.advstate,
            G.buttonstate[1]     : self.advstate,
            G.buttonstate[2]     : self.advstate,
            G.buttonstate[3]     : self.advstate,
            'quit'      : self.quit,        }[b]()

    def exup(self):
        self.webcam.params['exposure_absolute'].scaleup()
        self.webcam.updateparams()
        print(self.webcam.params['exposure_absolute'].value)
    def exdn(self):
        self.webcam.params['exposure_absolute'].scaledn()
        self.webcam.updateparams()
        print(self.webcam.params['exposure_absolute'].value)
    def satup(self):
        self.webcam.params['saturation'].scaleup()
        self.webcam.updateparams()
        print(self.webcam.params['saturation'].value)
    def satdn(self):
        self.webcam.params['saturation'].scaledn()
        self.webcam.updateparams()
        print(self.webcam.params['saturation'].value)
    def quit(self):
        self.timer.stop()
        QtGui.QApplication.quit()
    def advstate(self):
        if self.state == G.buttonstate[0]:
            # Move to tracking state
            self.state = G.buttonstate[1]
            self.cur_time = time.time()
            self.last_time = self.cur_time
            self.frameidx = 0
        elif self.state == G.buttonstate[1]:
            # Move to the food counting state,
            self.savedval = self.webcam.params['exposure_absolute'].value
            self.webcam.params['exposure_absolute'].value = 83
            self.webcam.params['exposure_absolute'].changed = True
            self.webcam.updateparams()
            self.extract.init_food_count()
            self.state = G.buttonstate[2]
        elif self.state == G.buttonstate[2]:
            self.webcam.params['exposure_absolute'].value = self.savedval
            self.webcam.params['exposure_absolute'].changed = True
            self.webcam.updateparams()
            self.state = G.buttonstate[3]
        elif self.state == G.buttonstate[3]:
            self.savedval = self.webcam.params['exposure_absolute'].value
            self.webcam.params['exposure_absolute'].value = 83
            self.webcam.params['exposure_absolute'].changed = True
            self.webcam.updateparams()
            self.extract.init_food_count()
            self.state = G.buttonstate[2]
        try:
            self.sender().setText(self.state)
        except AttributeError:
            pass


    # blob detector parameters
    def mintup(self):
        G.min_threshold += 1
    def mintdn(self):
        G.min_threshold -= 1
    def maxtup(self):
        G.max_threshold += 1
    def maxtdn(self):
        G.max_threshold -= 1
    def minaup(self):
        G.min_area += 1
    def minadn(self):
        G.min_area -= 1
    def maxaup(self):
        G.max_area += 1
    def maxadn(self):
        G.max_area -= 1



    def grab(self):
        #----------------------------------------------------------------
        # This is the main processing loop, it grabs frames from the
        # webcam and then processes them to extract useful data.
        # It is set called by a QT timer at a faster rate than the frame
        # rate so that it waits for the incoming frame
        #----------------------------------------------------------------

        # Capture the frame and process it
        f   = self.webcam.grabframe()
        #self.ui.img_raw.update_data(f)

        self.cur_time = time.time()

        # Are we measuring positions or counting food?
        if self.state in G.buttonstate[0:2]:
            # Run the image processing to detect kilobots and leds
            self.extract.run(f)

            # Update some status lines
            self.ui.updatestat(0, 'State: %s' % self.state)
            self.ui.updatestat(1, 'Robots:%3i' % self.extract.kiloworld.shape[0])
            self.ui.updatestat(2, 'min threshold: %d' % G.min_threshold)
            self.ui.updatestat(3, 'max threshold: %d' % G.max_threshold)
            self.ui.updatestat(4, 'min area:      %d' % G.min_area)
            self.ui.updatestat(5, 'max area:      %d' % G.max_area)

            # Update with ROI rectangle
            self.ui.img_postproc.update_roi(G.roi)

            # Update with kilobot locations
            self.ui.img_postproc.update_kilocircles(self.extract.kiloworld)

            # Put the monochrome image up as background
            #self.ui.img_postproc.update_data(self.extract.gf32world)
            self.ui.img_postproc.update_data(f)

            if self.state == G.buttonstate[0]:
                self.ui.updatebanner('Idle')

            if self.state == G.buttonstate[1]:

                if self.cur_time > self.last_time + 1.0:
                    self.last_time += 1.0
                    #self.last_time = self.cur_time
                    #self.extract.calc_recent_positions()
                    self.ui.updatebanner('Tracking: %3i' % self.frameidx)
                    np.savetxt('frame_%05i.data' % self.frameidx, self.extract.kilorecent)
                    print(self.extract.kilorecent)
                    self.frameidx += 1



        elif self.state == G.buttonstate[2]:
            # Interpret the flashing leds as foo counts
            finished, self.food, ticks_left = self.extract.countfood(f)
            self.ui.updatebanner('Counting food.. %3i' % ticks_left)
            if finished:
                cmds('echo %i > food.txt' % self.food)
                self.ui.img_postproc.update_kilocircles(self.extract.coords)
                self.ui.img_postproc.update_ledcircles(self.extract.goodcoords)
                self.advstate()

            else:
                self.ui.img_postproc.update_data(f)
                self.ui.img_postproc.update_ledcircles(self.extract.ledworld)

        elif self.state == G.buttonstate[3]:
            self.ui.updatebanner('Finished: Total food %4i (cl:%i val:%i)' % (self.food, len(self.extract.coords), len(self.extract.valfood)))


        # Redraw
        self.ui.img_postproc.redraw()


def sigint_handler(*args):
    QtGui.QApplication.quit()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image',      help='Image to display on projector', default='')

    args = parser.parse_args()
    print(args)


    app = QtGui.QApplication(sys.argv)
    print(app.desktop().availableGeometry(0))
    print(app.desktop().availableGeometry(1))
    # Set up a timer to give the python interpreter a go every 100ms
    # Without this, it is not possible to ctrl-c the app
    # http://stackoverflow.com/questions/4938723/what-is-the-correct-way-to-make-my-pyqt-application-quit-when-killed-from-the-co
    signal.signal(signal.SIGINT, sigint_handler)
    t = QtCore.QTimer()
    t.timeout.connect(lambda: None)
    t.start(100)

    p = None
    a = Arena(args=args)
    a.show()
    if p:
        p.show()
    sys.exit(app.exec_())


