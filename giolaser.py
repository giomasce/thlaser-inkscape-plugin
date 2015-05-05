#!/usr/bin/python2
# -*- coding: utf-8 -*-

import pprint
import collections

# Inkscape imports
import inkex
import simpletransform
import cubicsuperpath
import simplepath

# Change to 96.0 after Inkscape 0.91 (see
# http://wiki.inkscape.org/wiki/index.php/Units_In_Inkscape)
DEFAULT_DPI = 90.0

SVG_GROUP_TAG = inkex.addNS("g", "svg")
SVG_PATH_TAG = inkex.addNS('path','svg')
SVG_IMAGE_TAG = inkex.addNS('image', 'svg')
SVG_TEXT_TAG = inkex.addNS('text', 'svg')
SVG_LABEL_TAG = inkex.addNS("label", "inkscape")
SVG_GROUPMODE_TAG = inkex.addNS("groupmode", "inkscape")

DrawItem = collections.namedtuple('DrawItem', ['type', 'id', 'data'])

def clamp(val, inf, sup):
    return min(sup, max(inf, val))

class GioRumba:

    def __init__(self, foutput):
        self.foutput = foutput

    def home(self):
        self.foutput.write("G28 ; home\n")

    def goto_home(self):
        self.foutput.write("G0 X0 Y0 ; go to home\n")

    def select_mm(self):
        self.foutput.write("G21 ; select mm\n")

    def select_in(self):
        self.foutput.write("G20 ; select in\n")

    def laser_on(self, power):
        self.foutput.write("; laser on\nG4 P0\nM42 P8 S%d\n" % (clamp(int(256 * power)), 0, 255))

    def laser_off(self):
        self.foutput.write("; laser off\nG4 P0\nM42 P8 S0\n")

    def write_comment(self, comment):
        self.foutput.write(''.join(['; ' + x + '\n' for x in comment.splitlines()]))

BOARDS = {
    'gio_rumba': GioRumba,
}

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
                                     action="store", type="float", dest="move_feed", default="2000.0",
                                     help="Default move feedrate in unit/min.")
        self.OptionParser.add_option("-p", "--feed",
                                     action="store", type="float", dest="feed", default="300.0",
                                     help="Default cut feedrate in unit/min.")
        self.OptionParser.add_option("-l", "--laser",
                                     action="store", type="float", dest="laser", default="1.0",
                                     help="Default laser intensity (0.0-1.0).")
        self.OptionParser.add_option("-b", "--home-before",
                                     action="store", type="inkbool", dest="home_before", default=True,
                                     help="Home all before starting.")
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
                                     help="Defines maximum depth of splitting while approximating using biarc.")
        self.OptionParser.add_option("",   "--min-arc-radius",
                                     action="store", type="float", dest="min_arc_radius", default="0.0005",
                                     help="All arc having radius less than minimum will be considered as straight line")

        # Tweaks
        self.OptionParser.add_option("",   "--unit",
                                     action="store", type="string", dest="unit", default="mm",
                                     help="Units (mm or in).")

        self.OptionParser.add_option("",   "--tab",
                                     action="store", type="string", dest="tab", default="",
                                     help="Ignored.")

    def write_header(self):
        self.board.write_comment("Generated by Gio Laser Inkscape extension\n")

        # Just to be sure: shut down the laser
        self.board.laser_off()

        # Select working unit
        if self.options.unit == 'mm':
            self.board.select_mm()
            self.unit_scale = 25.4 / DEFAULT_DPI
        elif self.options.unit == 'in':
            self.board.select_in()
            self.unit_scale = 1.0 / DEFAULT_DPI

        # Possibly home device
        if self.options.home_before:
            self.board.home()

    def write_footer(self):
        # Just to be sure: shut down the laser
        self.board.laser_off()

        # Possibly home device
        if self.options.home_after:
            self.board.goto_home()

    def list_layers(self):
        for node in self.document.getroot().iterchildren():
            if node.tag == SVG_GROUP_TAG and node.get(SVG_GROUPMODE_TAG) == "layer":
                yield node

    def process_document(self):
        for layer in reversed(list(self.list_layers())):
            self.process_layer(layer)

    def list_deep_paths(self, node, trans=None):
        if trans is None:
            trans = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        trans = simpletransform.composeTransform(
            trans, simpletransform.parseTransform(node.get('transform', None)))

        if node.tag == SVG_PATH_TAG:
            csp = cubicsuperpath.parsePath(node.get("d"))
            simpletransform.applyTransformToPath(trans, csp)
            yield DrawItem('vector', node.get('id'), csp)

        elif node.tag == SVG_GROUP_TAG:
            for child in node.iterchildren():
                for path in self.list_deep_paths(child, trans=trans):
                    yield path

        else:
            inkex.errormsg("Cannot parse node with tag %s" % (node.tag))

    def process_layer(self, layer):
        # If the layer's label begins with "#", then we ignore it
        label = layer.get(SVG_LABEL_TAG).strip()
        if label[0] == '#':
            return

        # Parse parameters in layer's label
        params = {}
        try:
            layer_name, params_str = label.split('[', 1)
        except ValueError:
            name = label.strip()
        else:
            name = layer_name.strip()
            params_str = params_str.split(']')[0]
            params = dict([x.split('=', 1) for x in params_str.split(',')])

        # Interpret some well-known parameters
        if 'feed' in params:
            params['feed'] = float(params['feed'])
        if 'move-feed' in params:
            params['move-feed'] = float(params['move-feed'])
        if 'laser' in params:
            params['laser'] = float(params['laser'])

        # Setup a transform to account for measure units (mm or inch)
        trans = [[self.unit_scale, 0.0, 0.0], [0.0, self.unit_scale, 0.0]]

        # Retrieve the list of the paths to draw
        for path_info in self.list_deep_paths(layer, trans):
            path = path_info.data
            self.board.write_comment('\nPath with id: %s\n\n' % (path_info.id))
            for subpath in path:
                self.board.write_comment('New subpath\n')
                self.board.write_comment(pprint.pformat(subpath))

    def effect(self):
        with open(self.options.filename, 'w') as self.foutput:
            self.board = BOARDS[self.options.board](self.foutput)
            self.write_header()
            self.process_document()
            self.write_footer()

def main():
    gio_laser = GioLaser()
    gio_laser.affect()

if __name__ == '__main__':
    main()
