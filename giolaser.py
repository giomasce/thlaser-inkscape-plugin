#!/usr/bin/python2
# -*- coding: utf-8 -*-

import math
import cmath
import pprint
import collections
import Queue
import itertools

# Inkscape imports
import inkex
import simpletransform
import cubicsuperpath
import simplepath
import bezmisc

# Change to 96.0 after Inkscape 0.91 (see
# http://wiki.inkscape.org/wiki/index.php/Units_In_Inkscape)
DEFAULT_DPI = 90.0

CONTINUITY_TOLERANCE = 0.001

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

        time, cutting_time = self.board.get_time()
        if time is not None:
            self.board.write_comment('Total time elapsed: %f minutes (not including initial and final homing)' % (time))
        if cutting_time is not None:
            self.board.write_comment('Real cutting time: %f minutes' % (cutting_time))

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
        code = gcurve[0]
        if code == 'line':
            res = cmath.phase(complex(*gcurve[2]) - complex(*gcurve[1]))
        elif code == 'cw_arc':
            res = cmath.phase(complex(*gcurve[3]) - complex(*gcurve[1])) + 0.5 * math.pi
        elif code == 'ccw_arc':
            res = cmath.phase(complex(*gcurve[3]) - complex(*gcurve[1])) - 0.5 * math.pi
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
            assert len(graph[point]) == len(set(x[0] for x in graph[point]))
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
        for gcurve in new_gcurves:
            yield gcurve
        boundary.reverse()
        for bucket in boundary:
            new_gcurves, pos = self.optimize_gcurves(bucket, pos)
            for gcurve in new_gcurves:
                yield gcurve

    def generate_gcode(self, gcurves, params):
        current = None
        for curve in gcurves:
            code = curve[0]
            if code == 'line':
                assert len(curve) == 3
                orig = curve[1]
                dest = curve[2]
                if current is None or abs(complex(*current) - complex(*orig)) > CONTINUITY_TOLERANCE:
                    self.board.laser_off()
                    self.board.rapid_move(orig[0], orig[1], params['move-feed'])
                    self.board.laser_on(params['laser'])
                self.board.move(dest[0], dest[1], params['feed'])
                current = tuple(dest)
            elif code in ['cw_arc', 'ccw_arc']:
                assert len(curve) == 4
                orig = curve[1]
                dest = curve[2]
                center = curve[3]
                raise NotImplementedError()
            else:
                raise Exception("Should not arrive here")

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

        # Interpret some well-known parameters
        params['feed'] = float(params['feed']) if 'feed' in params else self.options.feed
        params['move-feed'] = float(params['move-feed']) if 'move-feed' in params else self.options.move_feed
        params['laser'] = float(params['laser']) if 'laser' in params else self.options.laser

        # Setup a transform to account for measure units (mm or inch)
        trans = [[self.unit_scale * self.options.xscale, 0.0, self.options.xoffset], [0.0, self.unit_scale * self.options.yscale, self.options.yoffset]]

        # Retrieve the list of the paths to draw
        gcurves = []
        for path in self.list_deep_paths(layer, trans):
            #self.board.write_comment('\nPath with id: %s\n\n' % (path.id))
            #self.board.write_comment(pprint.pformat(path.transform))
            #self.board.write_comment(pprint.pformat(path.data))
            gcurves += self.generate_gcurves(path)
        #self.board.write_comment(pprint.pformat(gcurves))

        # Decide best order to draw
        if self.options.draw_order in ['inside_first', 'outside_first']:
            gcurves = list(self.sorted_gcurves(gcurves))
            if self.options.draw_order == 'outside_first':
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
