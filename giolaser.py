#!/usr/bin/python2
# -*- coding: utf-8 -*-

import math
import pprint
import collections

# Inkscape imports
import inkex
import simpletransform
import cubicsuperpath
import simplepath
import bezmisc

# Change to 96.0 after Inkscape 0.91 (see
# http://wiki.inkscape.org/wiki/index.php/Units_In_Inkscape)
DEFAULT_DPI = 90.0

CONTINUITY_TOLERANCE = 0.01

SVG_GROUP_TAG = inkex.addNS("g", "svg")
SVG_PATH_TAG = inkex.addNS('path','svg')
SVG_IMAGE_TAG = inkex.addNS('image', 'svg')
SVG_TEXT_TAG = inkex.addNS('text', 'svg')
SVG_LABEL_TAG = inkex.addNS("label", "inkscape")
SVG_GROUPMODE_TAG = inkex.addNS("groupmode", "inkscape")

DrawItem = collections.namedtuple('DrawItem', ['type', 'id', 'data', 'transform'])

def clamp(val, inf, sup):
    return min(sup, max(inf, val))

class GioRumba:

    def __init__(self, foutput):
        self.foutput = foutput
        self.time = 0.0
        self.position = (0.0, 0.0)

    def write_header(self):
        self.foutput.write("M106 S255\n")

    def write_footer(self):
        self.foutput.write("M106 S0\n")

    def home(self):
        self.foutput.write("G28\n")

    def select_mm(self):
        self.foutput.write("G21\n")

    def select_in(self):
        self.foutput.write("G20\n")

    def laser_on(self, power):
        self.foutput.write("G4 P0\nM42 P8 S%d\n" % (clamp(int(256 * power), 0, 255)))

    def laser_off(self):
        self.foutput.write("G4 P0\nM42 P8 S0\n")

    def rapid_move(self, x, y, f):
        self.foutput.write("G0 X%f Y%f F%f\n" % (x, y, f))
        self.time += abs(complex(*self.position) - complex(x, y)) / f
        self.position = (x, y)

    def move(self, x, y, f):
        self.foutput.write("G1 X%f Y%f F%f\n" % (x, y, f))
        self.time += abs(complex(*self.position) - complex(x, y)) / f
        self.position = (x, y)

    def cw_arc(self, x, y, i, j, f):
        self.foutput.write("G2 X%f Y%f I%f J%f F%f\n" % (x, y, i, j, f))
        # TODO: update time

    def ccw_arc(self, x, y, i, j, f):
        self.foutput.write("G3 X%f Y%f I%f J%f F%f\n" % (x, y, i, j, f))
        # TODO: update time

    def write_comment(self, comment):
        self.foutput.write(''.join(['; ' + x + '\n' for x in comment.splitlines()]))

    def get_time(self):
        return self.time

BOARDS = {
    'gio_rumba': GioRumba,
}

def scalar(c1, c2):
    return (c1 * c2.conjugate()).real

def cross(c1, c2):
    return (c1 * c2.conjugare()).imag

def simple_biarc(p1, t1, p2, t2):
    """Compute the biarc interpolation between two points.

    Use the algorithm in
    http://www.ryanjuckett.com/programming/biarc-interpolation/
    """
    p1 = complex(*p1)
    p2 = complex(*p2)
    t1 = complex(*t1)
    t1 /= abs(t1)
    t2 = complex(*t2)
    t2 /= abs(t2)

    v = p2 - p1
    t = t1 + t2

    # Compute d, set as d = d2 = d1 (as in "Choosing d1")
    sc_vt = scalar(v, t)
    sc_tt = scalar(t1, t2)
    discr = sc_vt ** 2 + 2 * (1 - sc_tt) * scalar(v, v)
    assert discr >= 0.0
    if sc_tt != 1.0:
        # Case 1
        d = (discr ** 0.5 - sc_vt) / (2 * (1 - sc_tt))
    else:
        sc_vt2 = scal(v, t2)
        if sc_vt2 != 0.0:
            # Case 2
            d = sc_vv / (4 * sc_vt2)
        else:
            # Case 3
            raise NotImplementedError()

    # Compute pm (as in "Finding the Connection")
    q1 = p1 + d * t1
    q2 = p2 + d * t2
    pm = 0.5 * (q1 + q2)

    # Compute the centers and radii (as in "Finding the Center")
    n1 = 1j * t1
    n2 = 1j * t2
    s1 = 0.5 * scal(pm - p1, pm - p1) / scal(n1, pm - p1)
    s2 = 0.5 * scal(pm - p2, pm - p2) / scal(n2, pm - p2)
    c1 = p1 + s1 * n1
    c2 = p2 + s2 * n2
    r1 = abs(s1)
    r2 = abs(s2)

    # TODO

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
        self.board.write_header()

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
            self.board.move(0.0, 0.0, self.options.move_feed)

        time = self.board.get_time()
        if time is not None:
            self.board.write_comment('Total time elapsed: %f minutes (not including initial and final homing)' % (time))

        self.board.write_footer()

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
            path = simplepath.parsePath(node.get("d"))
            #simpletransform.applyTransformToPath(trans, path)
            yield DrawItem('vector', node.get('id'), path, trans)

        elif node.tag == SVG_GROUP_TAG:
            for child in node.iterchildren():
                for path in self.list_deep_paths(child, trans=trans):
                    yield path

        else:
            inkex.errormsg("Cannot parse node with tag %s" % (node.tag))

    def generate_gcurves_cubic(self, gcurves, old, first, second, new):
        """Compute the recursive biarc interpolation that describes a cubic curve.

        Points are assumed to have already been transformed according
        to current transformation.

        """
        # STUB
        gcurves.append(('line', old, new))

    def generate_gcurves(self, path):
        gcurves = []
        mat = path.transform
        def trans(p):
            return (mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2],
                    mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2])
        current = (0.0, 0.0)
        beginning = None
        for op in path.data:
            assert len(op) == 2
            code = op[0]
            if code == 'M':
                assert len(op[1]) == 2
                current = tuple(op[1])
                beginning = current
            elif code == 'L':
                assert len(op[1]) == 2
                new = tuple(op[1])
                gcurves.append(('line', trans(current), trans(new)))
                current = new
            elif code == 'Z':
                assert len(op[1]) == 0
                gcurves.append(('line', trans(current), trans(beginning)))
                current = beginning
            elif code == 'C':
                assert len(op[1]) == 6
                first = tuple(op[1][0:2])
                second = tuple(op[1][2:4])
                new = tuple(op[1][4:6])
                self.generate_gcurves_cubic(gcurves, trans(current),trans(first),
                                            trans(second), trans(new))
                current = new
            else:
                inkex.errormsg("Code %s not supported (so far...)" % (code))
                break
        return gcurves

    def sort_gcurves(self, gcurves):
        pass

    def generate_gcode(self, gcurves, params):
        current = None
        for curve in gcurves:
            code = curve[0]
            if code == 'line':
                assert len(curve) == 3
                if current is None or abs(complex(*current) - complex(*curve[1])) > CONTINUITY_TOLERANCE:
                    self.board.laser_off()
                    self.board.rapid_move(curve[1][0], curve[1][1], params['move-feed'])
                    self.board.laser_on(params['laser'])
                self.board.move(curve[2][0], curve[2][1], params['feed'])
                current = tuple(curve[2])
            else:
                raise NotImplementedError()

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

        self.board.write_comment("Layer %s" % (name))
        self.board.write_comment(pprint.pformat(params))

        # Interpret some well-known parameters
        params['feed'] = float(params['feed']) if 'feed' in params else self.options.feed
        params['move-feed'] = float(params['move-feed']) if 'move-feed' in params else self.options.move_feed
        params['laser'] = float(params['laser']) if 'laser' in params else self.options.laser

        # Setup a transform to account for measure units (mm or inch)
        trans = [[self.unit_scale, 0.0, 0.0], [0.0, self.unit_scale, 0.0]]

        # Retrieve the list of the paths to draw
        gcurves = []
        for path in self.list_deep_paths(layer, trans):
            #self.board.write_comment('\nPath with id: %s\n\n' % (path.id))
            self.board.write_comment(pprint.pformat(path.transform))
            self.board.write_comment(pprint.pformat(path.data))
            gcurves += self.generate_gcurves(path)
        #self.board.write_comment(pprint.pformat(gcurves))

        # Decide best order to draw
        if self.options.draw_order in ['inside_first', 'outside_first']:
            self.sort_gcurves(gcurves)
            if self.options.draw_order == 'inside_first':
                gcurves.reverse()

        # Actually draw
        self.generate_gcode(gcurves, params)

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
