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

STRAIGHT_TOLERANCE = 0.0001
STRAIGHT_DISTANCE_TOLERANCE = 0.0001

class P:
    def __init__(self, x, y=None):
        if not y==None:
            self.x, self.y = float(x), float(y)
        else:
            self.x, self.y = float(x[0]), float(x[1])
    def __add__(self, other): return P(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return P(self.x - other.x, self.y - other.y)
    def __neg__(self): return P(-self.x, -self.y)
    def __mul__(self, other):
        if isinstance(other, P):
            return self.x * other.x + self.y * other.y
        return P(self.x * other, self.y * other)
    __rmul__ = __mul__
    def __div__(self, other): return P(self.x / other, self.y / other)
    def mag(self): return math.hypot(self.x, self.y)
    def unit(self):
        h = self.mag()
        if h: return self / h
        else: return P(0,0)
    def dot(self, other): return self.x * other.x + self.y * other.y
    def rot(self, theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return P(self.x * c - self.y * s,  self.x * s + self.y * c)
    def angle(self): return math.atan2(self.y, self.x)
    def __repr__(self): return '%f,%f' % (self.x, self.y)
    def pr(self): return "%.2f,%.2f" % (self.x, self.y)
    def to_list(self): return [self.x, self.y]

def cspbezsplit(sp1, sp2, t = 0.5):
    s1,s2 = bezmisc.beziersplitatt((sp1[1],sp1[2],sp2[0],sp2[1]),t)
    return [ [sp1[0][:], sp1[1][:], list(s1[1])], [list(s1[2]), list(s1[3]), list(s2[1])], [list(s2[2]), sp2[1][:], sp2[2][:]] ]

def cspseglength(sp1,sp2, tolerance = 0.001):
    bez = (sp1[1][:],sp1[2][:],sp2[0][:],sp2[1][:])
    return bezmisc.bezierlength(bez, tolerance)

def biarc(sp1, sp2, z1, z2, depth=0,):
    def biarc_split(sp1,sp2, z1, z2, depth):
        if depth<options.biarc_max_split_depth:
            sp1,sp2,sp3 = cspbezsplit(sp1,sp2)
            l1, l2 = cspseglength(sp1,sp2), cspseglength(sp2,sp3)
            if l1+l2 == 0 : zm = z1
            else : zm = z1+(z2-z1)*l1/(l1+l2)
            return biarc(sp1,sp2,depth+1,z1,zm)+biarc(sp2,sp3,depth+1,z1,zm)
        else: return [ [sp1[1],'line', 0, 0, sp2[1], [z1,z2]] ]

    P0, P4 = P(sp1[1]), P(sp2[1])
    TS, TE, v = (P(sp1[2])-P0), -(P(sp2[0])-P4), P0 - P4
    tsa, tea, va = TS.angle(), TE.angle(), v.angle()
    if TE.mag()<STRAIGHT_DISTANCE_TOLERANCE and TS.mag()<STRAIGHT_DISTANCE_TOLERANCE:
        # Both tangents are zerro - line straight
        return [ [sp1[1],'line', 0, 0, sp2[1], [z1,z2]] ]
    if TE.mag() < STRAIGHT_DISTANCE_TOLERANCE:
        TE = -(TS+v).unit()
        r = TS.mag()/v.mag()*2
    elif TS.mag() < STRAIGHT_DISTANCE_TOLERANCE:
        TS = -(TE+v).unit()
        r = 1/( TE.mag()/v.mag()*2 )
    else:
        r=TS.mag()/TE.mag()
    TS, TE = TS.unit(), TE.unit()
    tang_are_parallel = ((tsa-tea)%math.pi<STRAIGHT_TOLERANCE or math.pi-(tsa-tea)%math.pi<STRAIGHT_TOLERANCE )
    if ( tang_are_parallel  and
                ((v.mag()<STRAIGHT_DISTANCE_TOLERANCE or TE.mag()<STRAIGHT_DISTANCE_TOLERANCE or TS.mag()<STRAIGHT_DISTANCE_TOLERANCE) or
                    1-abs(TS*v/(TS.mag()*v.mag()))<STRAIGHT_TOLERANCE)    ):
                # Both tangents are parallel and start and end are the same - line straight
                # or one of tangents still smaller then tollerance

                # Both tangents and v are parallel - line straight
        return [ [sp1[1],'line', 0, 0, sp2[1], [z1,z2]] ]

    c,b,a = v*v, 2*v*(r*TS+TE), 2*r*(TS*TE-1)
    if v.mag()==0:
        return biarc_split(sp1, sp2, z1, z2, depth)
    asmall, bsmall, csmall = abs(a)<10**-10,abs(b)<10**-10,abs(c)<10**-10
    if         asmall and b!=0:    beta = -c/b
    elif     csmall and a!=0:    beta = -b/a
    elif not asmall:
        discr = b*b-4*a*c
        if discr < 0:    raise ValueError, (a,b,c,discr)
        disq = discr**.5
        beta1 = (-b - disq) / 2 / a
        beta2 = (-b + disq) / 2 / a
        if beta1*beta2 > 0 :    raise ValueError, (a,b,c,disq,beta1,beta2)
        beta = max(beta1, beta2)
    elif    asmall and bsmall:
        return biarc_split(sp1, sp2, z1, z2, depth)
    alpha = beta * r
    ab = alpha + beta
    P1 = P0 + alpha * TS
    P3 = P4 - beta * TE
    P2 = (beta / ab)  * P1 + (alpha / ab) * P3

    def calculate_arc_params(P0,P1,P2):
        D = (P0+P2)/2
        if (D-P1).mag()==0: return None, None
        R = D - ( (D-P0).mag()**2/(D-P1).mag() )*(P1-D).unit()
        p0a, p1a, p2a = (P0-R).angle()%(2*math.pi), (P1-R).angle()%(2*math.pi), (P2-R).angle()%(2*math.pi)
        alpha =  (p2a - p0a) % (2*math.pi)
        if (p0a<p2a and  (p1a<p0a or p2a<p1a))    or    (p2a<p1a<p0a) :
            alpha = -2*math.pi+alpha
        if abs(R.x)>1000000 or abs(R.y)>1000000  or (R-P0).mag<options.min_arc_radius :
            return None, None
        else :
            return  R, alpha
    R1,a1 = calculate_arc_params(P0,P1,P2)
    R2,a2 = calculate_arc_params(P2,P3,P4)
    if R1==None or R2==None or (R1-P0).mag()<STRAIGHT_TOLERANCE or (R2-P2).mag()<STRAIGHT_TOLERANCE    : return [ [sp1[1],'line', 0, 0, sp2[1], [z1,z2]] ]

    d = get_distance_from_csp_to_arc(sp1,sp2, [P0,P2,R1,a1],[P2,P4,R2,a2])
    if d > options.biarc_tolerance and depth<options.biarc_max_split_depth     : return biarc_split(sp1, sp2, z1, z2, depth)
    else:
        if R2.mag()*a2 == 0 : zm = z2
        else : zm  = z1 + (z2-z1)*(R1.mag()*a1)/(R2.mag()*a2+R1.mag()*a1)
        return [    [ sp1[1], 'arc', [R1.x,R1.y], a1, [P2.x,P2.y], [z1,zm] ], [ [P2.x,P2.y], 'arc', [R2.x,R2.y], a2, [P4.x,P4.y], [zm,z2] ]        ]

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
                assert len(subpath) >= 2
                self.board.write_comment('New subpath\n')
                self.board.write_comment(pprint.pformat(subpath))
                data = []
                data.append([subpath[0][1], 'move', 0, 0])
                for i in range(1, len(subpath)):
                    data += biarc(subpath[i-1], subpath[i], 0, 0)
                data.append([subpath[-1][1], 'end', 0, 0])
                self.board.write_comment(pprint.pformat(data))

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
