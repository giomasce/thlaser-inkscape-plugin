#!/usr/bin/python2
# -*- coding: utf-8 -*-

import math
import cmath
import pprint
import collections
import Queue
import itertools
import copy

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
        self.cutting_time = 0.0
        self.position = (0.0, 0.0)

    def close_file(self):
        self.foutput.close()

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
        self.cutting_time += abs(complex(*self.position) - complex(x, y)) / f
        self.position = (x, y)

    def cw_arc(self, x, y, i, j, f):
        self.foutput.write("G2 X%f Y%f I%f J%f F%f\n" % (x, y, i, j, f))
        # TODO: update time

    def ccw_arc(self, x, y, i, j, f):
        self.foutput.write("G3 X%f Y%f I%f J%f F%f\n" % (x, y, i, j, f))
        # TODO: update time

    def cubic(self, x, y, i, j, p, q, f):
        self.foutput.write("G5 X%f Y%f I%f J%f P%f Q%f F%f\n" % (x, y, i, j, p, q, f))
        # TODO: update time

    def write_comment(self, comment):
        self.foutput.write(''.join(['; ' + x + '\n' for x in comment.splitlines()]))

    def get_time(self):
        return self.time, self.cutting_time

BOARDS = {
    'gio_rumba': GioRumba,
}

def scalar(c1, c2):
    return (c1 * c2.conjugate()).real

def cross(c1, c2):
    return -(c1 * c2.conjugate()).imag

def cpx_to_tup(c):
    return (c.real, c.imag)

def evaluate_casteljau(bezier, t):
    t2 = 1.0 - t
    bezier = copy.deepcopy(bezier)
    for i in xrange(len(bezier) - 1, 0, -1):
        for j in xrange(i):
            bezier[j] = t2 * bezier[j] + t * bezier[j+1]
    return bezier[0]

def simple_biarc(p1, t1, p2, t2):
    """Compute the biarc interpolation between two points.

    Input and output points are taken as complex numbers.

    Use the algorithm in
    http://www.ryanjuckett.com/programming/biarc-interpolation/

    """
    #inkex.errormsg(str((p1, t1, p2, t2)))
    # Normalize tangents and compute useful values
    t1_zero = False
    t2_zero = False
    # TODO - Magic constants
    if abs(t1) < 0.01:
        t1_zero = True
    else:
        t1 /= abs(t1)
    if abs(t2) < 0.01:
        t2_zero = True
    else:
        t2 /= abs(t2)
    v = p2 - p1
    t = t1 + t2

    # Handle degenerate cases: when both derivatives are zero, do
    # linear interpolation; when just one is zero, ignore it and do a
    # simple arc interpolation using the other one
    if t1_zero and t2_zero:
        return None, None, None
    if t1_zero or t2_zero:
        if t1_zero:
            n = t2 * 1j
        else:
            n = t1 * 1j
        #inkex.errormsg("v = %s, n = %s, scalar = %s" % (v, n, scalar(v, n)))
        try:
            k = 0.5 * abs(p1 - p2) ** 2 / scalar(v, n)
        except ZeroDivisionError:
            return None, None, None
        if t1_zero:
            c2 = p2 + k * n
            if abs(c2) >= 10000.0:
                raise Exception("Bad things happening")
            return None, p1, c2
        else:
            c1 = p1 + k * n
            if abs(c1) >= 10000.0:
                raise Exception("Bad things happening")
            return c1, p2, None

    # Compute d, set as d = d2 = d1 (as in "Choosing d1")
    sc_vt = scalar(v, t)
    sc_tt = scalar(t1, t2)
    discr = sc_vt ** 2 + 2 * (1 - sc_tt) * scalar(v, v)
    assert discr >= 0.0
    if sc_tt != 1.0:
        # Case 1
        d = (discr ** 0.5 - sc_vt) / (2 * (1 - sc_tt))
    else:
        sc_vt2 = scalar(v, t2)
        sc_vv = scalar(v, v)
        if sc_vt2 != 0.0:
            # Case 2
            d = sc_vv / (4 * sc_vt2)
        else:
            # Case 3
            pm = p1 + 0.5 * v
            c1 = p1 + 0.25 * v
            c2 = p1 + 0.75 * v
            return c1, pm, c2

    # Compute pm (as in "Finding the Connection")
    q1 = p1 + d * t1
    q2 = p2 - d * t2
    pm = 0.5 * (q1 + q2)

    # Compute the centers and radii (as in "Finding the Center")
    n1 = 1j * t1
    n2 = 1j * t2
    try:
        s1 = 0.5 * scalar(pm - p1, pm - p1) / scalar(n1, pm - p1)
    except ZeroDivisionError:
        c1 = None
    else:
        c1 = p1 + s1 * n1
        if abs(c1) >= 1000.0:
            c1 = None
    try:
        s2 = 0.5 * scalar(pm - p2, pm - p2) / scalar(n2, pm - p2)
    except ZeroDivisionError:
        c2 = None
    else:
        c2 = p2 + s2 * n2
        if abs(c2) >= 1000.0:
            c2 = None

    return c1, pm, c2

def recursive_biarc(cubic, tolerance, niter, derivative=None, t1=None, t2=None):
    # Checks and initialization
    assert len(cubic) == 4

    # As per http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
    if derivative is None:
        derivative = [3 * (cubic[1] - cubic[0]),
                      3 * (cubic[2] - cubic[1]),
                      3 * (cubic[3] - cubic[2])]
    if t1 is None:
        t1 = 0.0
        t2 = 1.0

    # Compute simple biarc
    p1 = evaluate_casteljau(cubic, t1)
    p2 = evaluate_casteljau(cubic, t2)
    tg1 = evaluate_casteljau(derivative, t1)
    #if p1 == p2:
    #    inkex.errormsg(repr(cubic))
    tg2 = evaluate_casteljau(derivative, t2)
    c1, pm, c2 = simple_biarc(p1, tg1, p2, tg2)

    # Decide if precision is enough (TODO: the algorithm has large
    # space for enhancement)
    if niter == 0:
        stop = True
    else:
        tgm = evaluate_casteljau(derivative, 0.5 * (t1 + t2))
        if tg2 != tg1:
            error = abs(tgm - 0.5 * (tg1 + tg2)) / abs(tg2 - tg1)
            #inkex.errormsg(str(error))
            stop = error < tolerance
        else:
            stop = False

    # If precision is enough or maximum iteration number has been
    # reached, return computed biarc
    if stop:
        #if (p1, c1, tg1, pm, c2, tg2, p2) == ((77.75394056000002+130.55184282388888j), (3170.984836277948-3029.8311549501805j), (-19.367597366666647-18.956072916666663j), (71.40089397111113+124.32125916j), None, 0j, (71.40089397111113+124.32125916j)):
        #    raise Exception(cubic, tolerance, niter, derivative, t1, t2)
        yield (p1, c1, tg1, pm, c2, tg2, p2)

    # Otherwise make recursive call
    else:
        for ret in recursive_biarc(cubic, tolerance, niter - 1, derivative, t1, 0.5 * (t1 + t2)):
            yield ret
        for ret in recursive_biarc(cubic, tolerance, niter - 1, derivative, 0.5 * (t1 + t2), t2):
            #if ret == ((77.75394056000002+130.55184282388888j), (3170.984836277948-3029.8311549501805j), (-19.367597366666647-18.956072916666663j), (71.40089397111113+124.32125916j), None, 0j, (71.40089397111113+124.32125916j)):
            #    raise Exception(repr((cubic, tolerance, niter - 1, derivative, 0.5 * (t1 + t2), t2)))
            yield ret

def gcurve_len(gcurve):
    code = gcurve[0]
    if code == 'line':
        return abs(complex(*gcurve[1]) - complex(*gcurve[2]))
    elif code in ['cw_arc', 'ccw_arc']:
        first = complex(*gcurve[1])
        second = complex(*gcurve[2])
        center = complex(*gcurve[3])
        angle = cmath.phase((first - center) / (second - center))
        # Double modulo is to get sure that norm_angle is actually in
        # [0.0, 2*math.pi). TODO: check numerical stability (when
        # angle is small and could easily wrap around)
        norm_angle = angle % (2 * math.pi) % (2 * math.pi)
        assert 0.0 <= norm_angle < (2 * math.pi), (angle, norm_angle)
        if code == 'ccw_arc':
            return norm_angle * abs(first - center)
        else:
            return (2 * math.pi - norm_angle) * abs(first - center)
    elif code == 'cubic':
        # TODO: implement
        return 10.0
    else:
        raise Exception("Should not arrive here, code = %s" % (code))

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
        self.OptionParser.add_option("",   "--use-arcs",
                                     action="store", type="inkbool", dest="use_arcs", default=False,
                                     help="Use arc commands")
        self.OptionParser.add_option("",   "--biarc-tolerance",
                                     action="store", type="float", dest="biarc_tolerance", default="0.1",
                                     help="Tolerance used when calculating biarc interpolation.")
        self.OptionParser.add_option("",   "--biarc-max-split-depth",
                                     action="store", type="int", dest="biarc_max_split_depth", default="4",
                                     help="Defines maximum depth of splitting while approximating using biarc.")
        self.OptionParser.add_option("",   "--min-arc-radius",
                                     action="store", type="float", dest="min_arc_radius", default="0.0005",
                                     help="All arc having radius less than minimum will be considered as straight line")
        self.OptionParser.add_option("",   "--curves-as-segments",
                                     action="store", type="inkbool", dest="curves_as_segments", default=False,
                                     help="Treat all curves as segments")

        # Tweaks
        self.OptionParser.add_option("",   "--unit",
                                     action="store", type="string", dest="unit", default="mm",
                                     help="Units (mm or in).")

        self.OptionParser.add_option("",   "--tab",
                                     action="store", type="string", dest="tab", default="",
                                     help="Ignored.")

    def write_header(self, board):
        board.write_comment("Generated by Gio Laser Inkscape extension\n")
        board.write_header()

        # Just to be sure: shut down the laser
        board.laser_off()

        # Select working unit
        if self.options.unit == 'mm':
            board.select_mm()
        elif self.options.unit == 'in':
            board.select_in()

        # Possibly home device
        if self.options.home_before:
            board.home()

    def write_footer(self, board):
        # Just to be sure: shut down the laser
        board.laser_off()

        # Possibly home device
        if self.options.home_after:
            board.move(0.0, 0.0, self.options.move_feed)

        time, cutting_time = board.get_time()
        if time is not None:
            board.write_comment('Total time elapsed: %f minutes (not including initial and final homing)' % (time))
        if cutting_time is not None:
            board.write_comment('Real cutting time: %f minutes' % (cutting_time))

        board.write_footer()

    def list_layers(self):
        for node in self.document.getroot().iterchildren():
            if node.tag == SVG_GROUP_TAG and node.get(SVG_GROUPMODE_TAG) == "layer":
                yield node

    def process_document(self):
        # Select working unit
        if self.options.unit == 'mm':
            self.unit_scale = 25.4 / DEFAULT_DPI
        elif self.options.unit == 'in':
            self.unit_scale = 1.0 / DEFAULT_DPI

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

    def snap_to_grid(self, p):
        def snap(x):
            return round(x / CONTINUITY_TOLERANCE) * CONTINUITY_TOLERANCE
        return (snap(p[0]), snap(p[1]))

    def add_gcurve(self, gcurves, gcurve):
        if gcurve_len(gcurve) > 10000.0:
            # TODO: this fails without credible reasons
            #raise Exception("Bad gcurve: %r" % ((gcurve, gcurve_len(gcurve)),))
            pass
        #inkex.errormsg("adding curve: %r" % ((gcurve, gcurve_len(gcurve)),))
        gcurves.append(gcurve)

    def interp_cubic_gcurve(self, gcurves, old, first, second, new):
        """Compute the recursive biarc interpolation that describes a cubic curve.

        Points are assumed to have already been transformed according
        to current transformation.

        """
        old = complex(*old)
        first = complex(*first)
        second = complex(*second)
        new = complex(*new)
        t1 = first - old
        t2 = new - second
        biarcs = recursive_biarc([old, first, second, new],
                                 self.options.biarc_tolerance,
                                 self.options.biarc_max_split_depth)

        # Test straight line approximation
        #gcurves.append(('line', self.snap_to_grid(cpx_to_tup(old)), self.snap_to_grid(cpx_to_tup(new))))

        # Write curves
        for p1, c1, tg1, pm, c2, tg2, p2 in biarcs:
            if pm is None:
                self.add_gcurve(gcurves, ('line', self.snap_to_grid(cpx_to_tup(p1)), self.snap_to_grid(cpx_to_tup(p2))))
            else:
                if c1 is not None:
                    try:
                        self.add_gcurve(gcurves, ('ccw_arc' if cross(p1 - c1, tg1) > 0 else 'cw_arc',
                                                  self.snap_to_grid(cpx_to_tup(p1)), self.snap_to_grid(cpx_to_tup(pm)), cpx_to_tup(c1)))
                    except:
                        inkex.errormsg("Error while considering: %r" % ((p1, c1, tg1, pm, c2, tg2, p2),))
                        raise
                else:
                    self.add_gcurve(gcurves, ('line', self.snap_to_grid(cpx_to_tup(p1)), self.snap_to_grid(cpx_to_tup(pm))))
                if c2 is not None:
                    self.add_gcurve(gcurves, ('ccw_arc' if cross(p2 - c2, tg2) > 0 else 'cw_arc',
                                              self.snap_to_grid(cpx_to_tup(pm)), self.snap_to_grid(cpx_to_tup(p2)), cpx_to_tup(c2)))
                else:
                    self.add_gcurve(gcurves, ('line', self.snap_to_grid(cpx_to_tup(pm)), self.snap_to_grid(cpx_to_tup(p2))))

    def generate_gcurves(self, path, params):
        gcurves = []
        mat = path.transform
        def trans(p):
            return (mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2],
                    mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2])
        current = (0.0, 0.0)
        beginning = None
        #inkex.errormsg("path data: %s" % (path.data))
        for op in path.data:
            assert len(op) == 2
            code = op[0]
            if code == 'M':
                assert len(op[1]) == 2
                current = tuple(op[1])
                beginning = current
            elif code == 'Z':
                assert len(op[1]) == 0
                self.add_gcurve(gcurves, ('line', self.snap_to_grid(trans(current)), self.snap_to_grid(trans(beginning))))
                current = beginning
            elif code == 'L' or params['curves-as-segments']:
                assert len(op[1]) == 2 or params['curves-as-segments']
                new = tuple(op[1])
                self.add_gcurve(gcurves, ('line', self.snap_to_grid(trans(current)), self.snap_to_grid(trans(new))))
                current = new
            elif code == 'C' or code == 'Q':
                if code == 'C':
                    assert len(op[1]) == 6
                    first = tuple(op[1][0:2])
                    second = tuple(op[1][2:4])
                    new = tuple(op[1][4:6])
                else:
                    # See https://fontforge.github.io/bezier.html
                    # ("Converting TrueType to PostScript")
                    assert len(op[1]) == 4
                    tmp = tuple(op[1][0:2])
                    new = tuple(op[1][2:4])
                    first = (current[0] + 2.0 / 3.0 * (tmp[0] - current[0]),
                             current[1] + 2.0 / 3.0 + (tmp[1] - current[1]))
                    second = (new[0] + 2.0 / 3.0 * (tmp[0] - new[0]),
                              new[1] + 2.0 / 3.0 * (tmp[1] - new[1]))
                if self.options.use_arcs:
                    try:
                        self.interp_cubic_gcurve(gcurves, trans(current),trans(first),
                                                 trans(second), trans(new))
                    except:
                        inkex.errormsg("Exception while processing cubic curve %r" %
                                       ((trans(current),trans(first),
                                         trans(second), trans(new)),))
                        raise
                else:
                    self.add_gcurve(gcurves, ('cubic', self.snap_to_grid(trans(current)), self.snap_to_grid(trans(new)), trans(first), trans(second)))
                current = new
            else:
                inkex.errormsg("Code %s not supported (so far...)" % (code))
                break
        gcurves = [gcurve for gcurve in gcurves if gcurve_len(gcurve) > 0]
        #inkex.errormsg("gcurves:\n%s" % (pprint.pformat(gcurves)))
        return gcurves

    def invert_gcurve(self, gcurve):
        code = gcurve[0]
        if code == 'line':
            return ('line', gcurve[2], gcurve[1])
        elif code == 'cw_arc':
            return ('ccw_arc', gcurve[2], gcurve[1], gcurve[3])
        elif code == 'ccw_arc':
            return ('cw_arc', gcurve[2], gcurve[1], gcurve[3])
        else:
            raise Exception("Should not arrive here, code = %s" % (code))

    def compute_gcurve_tangent(self, gcurve):
        def phase(x):
            if abs(x) == 0.0:
                raise ValueError("Cannot compute phase of 0.0")
            else:
                return cmath.phase(x)
        code = gcurve[0]
        if code == 'line':
            res = phase(complex(*gcurve[2]) - complex(*gcurve[1]))
        elif code == 'cw_arc':
            res = phase(complex(*gcurve[3]) - complex(*gcurve[1])) + 0.5 * math.pi
        elif code == 'ccw_arc':
            res = phase(complex(*gcurve[3]) - complex(*gcurve[1])) - 0.5 * math.pi
        else:
            raise Exception("Should not arrive here, code = %s" % (code))
        res = res % (2 * math.pi)
        if res < 0.0:
            res += 2 * math.pi
        return res

    def build_graph(self, gcurves):
        graph = {}
        for direct in gcurves:
            inverse = self.invert_gcurve(direct)
            for gcurve, gcurve2 in [(direct, inverse), (inverse, direct)]:
                orig = gcurve[1]
                dest = gcurve[2]
                for point in graph:
                    if abs(complex(*point) - complex(*orig)) <= CONTINUITY_TOLERANCE:
                        break
                else:
                    point = orig
                    graph[point] = []
                for point2 in graph:
                    if abs(complex(*point2) - complex(*dest)) <= CONTINUITY_TOLERANCE:
                        break
                else:
                    point2 = dest
                    graph[point2] = []
                graph[point].append((self.compute_gcurve_tangent(gcurve),
                                     [point2, self.compute_gcurve_tangent(gcurve2),
                                      gcurve, gcurve2,
                                      False]))
        for point in graph:
            assert len(graph[point]) == len(set(x[0] for x in graph[point])), pprint.pformat(graph[point])
            graph[point].sort()
        return graph

    def walk_cell(self, graph, edge):
        if edge[4]:
            return [], 0.0
        edge[4] = True
        point = edge[0]
        angle = edge[1]
        new_edges = [x for x in graph[point] if x[0] > angle]
        if len(new_edges) > 0:
            new_edge = new_edges[0]
        else:
            new_edge = graph[point][0]
        new_angle = new_edge[0]
        rot = angle - new_angle
        rot %= 2 * math.pi
        if rot < 0.0:
            rot += 2 * math.pi
        rot -= math.pi
        prev_cell, prev_rot = self.walk_cell(graph, new_edge[1])
        prev_cell.append((edge[2], edge[3]))
        return prev_cell, prev_rot + rot

    def build_cells(self, graph):
        cells = []
        rev_cells = {}
        for point, edges in graph.iteritems():
            for edge in edges:
                cell = self.walk_cell(graph, edge[1])
                if cell[0] != []:
                    assert abs(cell[1]) - 2 * math.pi <= 0.001, cell[1]
                    cell_list = cell[0]
                    cell_list.reverse()
                    cells.append([cell_list, cell[1], None])
                    for direct, _ in cell_list:
                        rev_cells[direct] = cells[-1]
        return cells, rev_cells

    def compute_cell_distance(self, main_cell, rev_cells):
        queue = Queue.Queue()
        queue.put((main_cell, 0))
        while True:
            try:
                cell, dist = queue.get(block=False)
            except Queue.Empty:
                break
            if cell[2] is not None:
                continue
            cell[2] = dist
            for edge in cell[0]:
                new_cell = rev_cells[edge[1]]
                queue.put((new_cell, dist + 1))

    # FIXME: highly optimizable!
    def optimize_gcurves(self, gcurves, pos):
        new_gcurves = []
        while len(gcurves) > 0:
            for_min = [(abs(complex(*gcurve[1]) - complex(*pos)), gcurve, gcurve) for gcurve in gcurves]
            for_min += [(abs(complex(*gcurve[2]) - complex(*pos)), self.invert_gcurve(gcurve), gcurve) for gcurve in gcurves]
            min_el = min(for_min)
            gcurves.remove(min_el[2])
            pos = min_el[1][2]
            new_gcurves.append(min_el[1])
        return new_gcurves, pos

    def sorted_gcurves(self, gcurves):
        #inkex.errormsg("initial gcurves:\n%s" % (pprint.pformat(gcurves)))
        graph = self.build_graph(gcurves)
        cells, rev_cells = self.build_cells(graph)
        main_cells = [cell for cell in cells if cell[1] < 0.0]
        for main_cell in main_cells:
            self.compute_cell_distance(main_cell, rev_cells)
        assert all(x[2] is not None for x in cells)
        max_depth = max(x[2] for x in cells)
        boundary = [[] for _ in xrange(max_depth)]
        internal = [[] for _ in xrange(max_depth)]
        for direct in gcurves:
            reverse = self.invert_gcurve(direct)
            dir_depth = rev_cells[direct][2]
            rev_depth = rev_cells[reverse][2]
            min_depth = min(dir_depth, rev_depth)
            max_depth = max(dir_depth, rev_depth)
            if min_depth == max_depth:
                internal[min_depth - 1].append(direct)
            else:
                boundary[min_depth].append(direct)
        #inkex.errormsg(pprint.pformat(cells))
        #inkex.errormsg(repr(rev_cells))
        #inkex.errormsg(pprint.pformat((boundary, internal)))
        pos = (0.0, 0.0)
        new_gcurves, pos = self.optimize_gcurves(list(itertools.chain(*internal)), pos)
        #inkex.errormsg("internal gcurves:\n%s" % (pprint.pformat(new_gcurves)))
        for gcurve in new_gcurves:
            yield gcurve
        boundary.reverse()
        for bucket in boundary:
            new_gcurves, pos = self.optimize_gcurves(bucket, pos)
            #inkex.errormsg("boundary gcurves:\n%s" % (pprint.pformat(new_gcurves)))
            for gcurve in new_gcurves:
                yield gcurve

    def generate_gcode(self, gcurves, params, board):
        current = None
        for curve in gcurves:
            code = curve[0]
            orig = curve[1]
            dest = curve[2]
            if current is None or abs(complex(*current) - complex(*orig)) > CONTINUITY_TOLERANCE:
                board.laser_off()
                board.rapid_move(orig[0], orig[1], params['move-feed'])
                board.laser_on(params['laser'])
            if code == 'line':
                assert len(curve) == 3
                board.move(dest[0], dest[1], params['feed'])
            elif code == 'cw_arc':
                assert len(curve) == 4
                center = curve[3]
                board.cw_arc(dest[0], dest[1], center[0] - orig[0], center[1] - orig[1], params['feed'])
            elif code == 'ccw_arc':
                assert len(curve) == 4
                center = curve[3]
                board.ccw_arc(dest[0], dest[1], center[0] - orig[0], center[1] - orig[1], params['feed'])
            elif code == 'cubic':
                assert len(curve) == 5
                first = curve[3]
                second = curve[4]
                board.cubic(dest[0], dest[1], first[0] - orig[0], first[1] - orig[1], second[0] - dest[0], second[1] - dest[1], params['feed'])
            else:
                raise Exception("Should not arrive here")
            current = tuple(dest)

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

        # Select board
        board = self.get_board(params.get('tag', None))
        board.write_comment("Layer %s" % (name))
        board.write_comment("with parameters: %s" % (params))

        # Interpret some well-known parameters
        params['feed'] = float(params['feed']) if 'feed' in params else self.options.feed
        params['move-feed'] = float(params['move-feed']) if 'move-feed' in params else self.options.move_feed
        params['laser'] = float(params['laser']) if 'laser' in params else self.options.laser
        params['curves-as-segments'] = params['curves-as-segments'] == 'true' if 'curves-as-segments' in params else self.options.curves_as_segments

        # Setup a transform to account for measure units (mm or inch)
        trans = [[self.unit_scale * self.options.xscale, 0.0, self.options.xoffset],
                 [0.0, self.unit_scale * self.options.yscale, self.options.yoffset]]

        # Retrieve the list of the paths to draw
        gcurves = []
        for path in self.list_deep_paths(layer, trans):
            #board.write_comment('\nPath with id: %s\n\n' % (path.id))
            #board.write_comment(pprint.pformat(path.transform))
            #board.write_comment(pprint.pformat(path.data))
            gcurves += self.generate_gcurves(path, params)
        #board.write_comment(pprint.pformat(gcurves))

        # Decide best order to draw
        draw_order = params['draw-order'] if 'draw-order' in params else self.options.draw_order
        if draw_order in ['inside_first', 'outside_first']:
            gcurves = list(self.sorted_gcurves(gcurves))
            if self.options.draw_order == 'outside_first':
                gcurves.reverse()

        # Actually draw
        self.generate_gcode(gcurves, params, board)

    def get_board(self, tag):
        if self.single_board or tag is None:
            tag = ''
        if tag not in self.boards:
            filename = self.options.filename.replace("%s", tag)
            fout = open(filename, 'w')
            self.boards[tag] = self.board_class(fout)
            self.write_header(self.boards[tag])
        return self.boards[tag]

    def effect(self):
        self.boards = {}
        self.board_class = BOARDS[self.options.board]
        self.single_board = (self.options.filename.find("%s") == -1)
        self.process_document()
        for board in self.boards.itervalues():
            self.write_footer(board)
            board.close_file()

def main():
    gio_laser = GioLaser()
    gio_laser.affect()

if __name__ == '__main__':
    main()
