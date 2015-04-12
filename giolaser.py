#!/usr/bin/python2
# -*- coding: utf-8 -*-

import inkex

class GioLaser(inkex.Effect):

    def __init__(self):
        inkex.Effect.__init__(self)

        self.OptionParser.add_option("-f", "--filename",
                                     action="store", type="string", dest="filename", default=None,
                                     help="File name.")
        self.OptionParser.add_option("",   "--board",
                                     action="store", type="string", dest="board", default="gio_rumba",
                                     help="Control board.")

        # Scaling and offset
        self.OptionParser.add_option("-u", "--xscale",
                                     action="store", type="float", dest="xscale", default=1.0,
                                     help="Scale factor X.")
        self.OptionParser.add_option("-v", "--yscale",
                                     action="store", type="float", dest="yscale", default=1.0,
                                     help="Scale factor Y.")
        self.OptionParser.add_option("-x", "--xoffset",
                                     action="store", type="float", dest="xoffset", default=0.0,
                                     help="Offset along X.")
        self.OptionParser.add_option("-y", "--yoffset",
                                     action="store", type="float", dest="yoffset", default=0.0,
                                     help="Offset along Y.")

        # Drawing
        self.OptionParser.add_option("-m", "--move-feed",
                                     action="store", type="int", dest="move_feed", default="2000",
                                     help="Default move feedrate in unit/min.")
        self.OptionParser.add_option("-p", "--feed",
                                     action="store", type="int", dest="feed", default="300",
                                     help="Default cut feedrate in unit/min.")
        self.OptionParser.add_option("-l", "--laser",
                                     action="store", type="int", dest="laser", default="255",
                                     help="Default laser intensity (0-255).")
        self.OptionParser.add_option("-b", "--home-before",
                                     action="store", type="inkbool", dest="home_before", default=True,
                                     help="Home all before starting (G28).")
        self.OptionParser.add_option("-a", "--home-after",
                                     action="store", type="inkbool", dest="home_after", default=False,
                                     help="Home X Y at end of the job.")
        self.OptionParser.add_option("",   "--draw-order",
                                     action="store", type="string", dest="draw_order", default="inside_first",
                                     help="Drawing order ('inside-first', 'outside-first' or 'no_sort').")
        self.OptionParser.add_option("",   "--origin",
                                     action="store", type="string", dest="origin", default="topleft",
                                     help="Origin position (topleft or bottomleft).")

        # Tolerances and approximations
        self.OptionParser.add_option("",   "--biarc-tolerance",
                                     action="store", type="float", dest="biarc_tolerance", default="1",
                                     help="Tolerance used when calculating biarc interpolation.")
        self.OptionParser.add_option("",   "--biarc-max-split-depth",
                                     action="store", type="int", dest="biarc_max_split_depth", default="4",
                                     help="Defines maximum depth of splitting while approximating using biarcs.")
        self.OptionParser.add_option("",   "--min-arc-radius",
                                     action="store", type="float", dest="min_arc_radius", default="0.0005",
                                     help="All arc having radius less than minimum will be considered as straight line")

        # Tweaks
        self.OptionParser.add_option("",   "--unit",
                                     action="store", type="string", dest="unit", default="mm",
                                     help="Units")

        self.OptionParser.add_option("",   "--tab",
                                     action="store", type="string", dest="tab", default="",
                                     help="Ignored.")

def main():
    gio_laser = GioLaser()
    gio_laser.affect()

if __name__ == '__main__':
    main()
