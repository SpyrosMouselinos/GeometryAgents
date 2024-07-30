PACK_ORDERING = {'Alpha': 0,
                 'Beta': 1,
                 'Delta': 2,
                 'Epsilon': 3,
                 'Eta': 4,
                 'Gamma': 5,
                 'Iota': 6,
                 'Kappa': 7,
                 'Lambda': 8,
                 'Theta': 9,
                 'Zeta': 10}

FILTERED_EUCLID = ["\nThe ends of a line are points, meaning that a line segment is defined by two points."
                   "\nA plane angle is formed when two lines in a plane meet at a point, and it represents the inclination or deviation from a straight line."
                   "\nA right-angle is a 90-degree angle formed when two lines intersect perpendicularly."
                   "\nAn obtuse angle is an angle greater than 90 degrees."
                   "\nAn acute angle is an angle less than 90 degrees."
                   "\nA circle is a specific type of figure that is defined by a single line called the circumference, with all lines radiating from a point inside "
                   "the circle being equal."
                   "\nThe center of a circle is the point from which all lines radiate."
                   "\nA diameter is a straight-line passing through the "
                   "center of a circle and terminating at the circumference, cutting the circle in half."
                   "\nA semi-circle is half of a circle, defined by a diameter."
                   "\nRectilinear figures are shapes defined by straight lines, including trilateral, quadrilateral, and multilateral figures."
                   "\nAny rectilinear figure can be split in halves by a line through the intersection of its diagonals."
                   "\nTrilateral figures include equilateral, isosceles, and scalene triangles."
                   "\nRight-angled, obtuse-angled, and acute-angled triangles are specific types of trilateral figures."
                   "\nQuadrilateral figures include squares, rectangles, rhombi, and rhomboids, with other quadrilateral figures referred to as trapezia."
                   "\nParallel lines are straight lines in the same plane that, when extended infinitely in both directions, do not intersect."
                   "\n\nYou can always:"
                   "\nDraw a straight-line from any point to any other point."
                   "\nExtend a finite straight-line continuously in a straight path."
                   "\nDraw a circle with any center, and radius."
                   "\n\nYou know that the following always hold:"
                   "\nAll right-angles are equal to one another."
                   "\nIf two things are equal to a third thing, they are also equal to each other."
                   "\nAdding equal things to equal things results in equal wholes."
                   "\nThe principle that things that coincide or perfectly overlap are equal."
                   "\nThe whole is greater than any of its parts."]

COMMON_GEOMETRIC_TERMS_AND_THEOREMS = [
    "The center of a circle or a quadrilateral is axiomatically located at the intersection point of perpendicular bisectors of any two chords or diagonals.",
    "The diagonals of a rectangle are congruent and bisect each other.",
    "Angles can be axiomatically constructed by bisecting larger angles or using the relationship between inscribed and central angles in a circle.",
    "The golden section, is a geometric proportion that divides a line segment into two parts in such a way that the ratio of the whole segment to the longer part is equal to the ratio of the longer part to the shorter part.",
    "Thales theorem: In a circle, if a line is drawn from one endpoint of a diameter to any point on the circle, then the angle formed at the point on the circle is always a right angle",
    "Specific lengths, including irrational numbers the geometric mean, and the golden section, can be axiomatically created using intersecting circles, perpendicular lines, the Pythagorean Theorem, properties of similar triangles, and circle-based segment division methods.",
    "An excircle is a circle that is outside a polygon and touches one of its sides and the extensions of the other two sides.",
    "To achieve equal area division of an circle, a concentric circle with a perpendicular bisector of the diameter as the new radius. For quadrilateral, the intersection of their diagonals is the center, and any line passing through this center bisects the rectangle equally.",
    "The Torricelli point is inside or on the triangle where the sum of the distances from this point to the three vertices of the triangle is minimized and can be found by constructing equilateral triangles on each side and intersecting the lines from their vertices to the opposite vertices of the original triangle.",
    "Heron's Formula is used to find the area of a triangle when you know the lengths of all three sides. Area = squrt((s(s - a)(s - b)(s - c))).",
    "Viviani's Theorem: The sum of distances from any point inside an equilateral triangle to its sides is constant.",
    "Morley's Theorem: Trisecting the angles of a triangle and connecting the trisection points forms an equilateral triangle.",
    "Barycentric Coordinates allow expressing any point within a triangle using weighted ratios.",
    "The Butterfly Theorem relates to the intersection of two lines with intersecting chords in a circle, showing proportional segments.",
    "Inscribed Angle Theorem: The measure of an inscribed angle in a circle is half the measure of the central angle that subtends the same arc. This theorem is often used when working with triangles inscribed in circles, as it relates angles formed by the intersection of chords and arcs."
]

FILTERED_TOOLS = [
    "\nLine Tool: Creates a line between two given points."
    "\nCircle Tool: Creates a circle with center, a given point and radius equal to the distance between the first given point and the second given point."
    "\nIntersection Tool: Returns the point where two lines or circles or bisectors intersect. In case of circles they can intersect in one or two points."
    "\nPerpendicular Bisector Tool: Returns a line perpendicular to the midpoint between two points."
    "\nAngle Bisector Tool: Returns a line that splits a given angle in two equal angles. The line has as a start the point of the given angle.",
]

VRP_PROMPT = "You are a helpful assistant that knows how to interpret geometric images and solve geometric problems. Given the image describe all the points, lines shapes and angles you can identify. Analyse the object relations and relative placement towards each other. Do not measure any distances, scales or magnitudes. Do not use arbitrary numbers. Think step-by step and return your results in bullets."

INCEPTION_PROMPT = """
You need to solve geometry problems by suggesting a series of steps.
You know definitions, postulates, and common notions from the book Elements by Euclid.

Definitions:
A point is an entity that has no physical size or dimensions.
A line is a geometric object that has length but no width.
The extremities of a line are points, meaning that a line segment is defined by two points.
A straight-line is a line that is even or continuous, with points lying on it in a straight path.
A surface is a two-dimensional object that has both length and width.
The extremities of a surface are lines, meaning that a surface is defined by the boundaries formed by lines.
A plane surface is a flat surface that lies evenly with straight lines on itself.
A plane angle is formed when two lines in a plane meet at a point, and it represents the inclination or deviation from a straight line.
A rectilinear angle is an angle formed by straight lines.
A right-angle is a 90-degree angle formed when two lines intersect perpendicularly.
An obtuse angle is an angle greater than 90 degrees.
An acute angle is an angle less than 90 degrees.
A boundary is the outermost part of an object.
A figure is a shape that is enclosed by boundaries.
A circle is a specific type of figure that is defined by a single line called the circumference, with all lines radiating from a point inside the circle being equal.
The center of a circle is the point from which all lines radiate.
A diameter is a straight-line passing through the center of a circle and terminating at the circumference, cutting the circle in half.
A semi-circle is half of a circle, defined by a diameter.
Rectilinear figures are shapes defined by straight lines, including trilateral, quadrilateral, and multilateral figures.
Trilateral figures include equilateral, isosceles, and scalene triangles.
Right-angled, obtuse-angled, and acute-angled triangles are specific types of trilateral figures.
Quadrilateral figures include squares, rectangles, rhombi, and rhomboids, with other quadrilateral figures referred to as trapezia.
Parallel lines are straight lines in the same plane that, when extended infinitely in both directions, do not intersect.

Postulates:
The ability to draw a straight-line from any point to any other point.
The ability to extend a finite straight-line continuously in a straight path.
The ability to draw a circle with any center and radius.
The postulate that all right-angles are equal to one another.
The parallel postulate, which deals with the behavior of straight lines that intersect another line.

Common Notions:
The principle that if two things are equal to a third thing, they are also equal to each other.
The principle that adding equal things to equal things results in equal wholes.
The principle that subtracting equal things from equal things results in equal remainders.
The principle that things that coincide or perfectly overlap are equal.
The principle that the whole is greater than any of its parts.

Each problem example has a description and a solution.
For each problem, you must choose from the tools given in the tool list.

Tool Descriptions:
Line Tool: Creates a line between two given points.
Circle Tool: Creates a circle with center, a given point and radius equal to the distance between the first given point and the second given point.
Intersection Tool: Returns the point where two lines or circles or bisectors intersect. In case of circles they can intersect in one or two points.
Perpendicular Bisector Tool: Returns a line perpendicular to the midpoint between two points.
Perpendicular Tool: Returns a line perpendicular to a line at a given point.
Angle Bisector Tool: Returns a line that splits a given angle in two equal angles. The line has as a start the point of the given angle.
"""


def get_ctt():
    return '\n'.join(COMMON_GEOMETRIC_TERMS_AND_THEOREMS)


NL_SOLVER_INCEPTION_PROMPT = [
    "You are an expert mathematician who focuses on euclidean geometry.",
    "We share a common interest in collaborating to successfully solve a problem step by step.",
    "Your main responsibilities include being an reasoner, a planner and a solution designer.",
    f"\nYou base your answers on the following principles: {get_ctt()}"
    "\n\nYou must help me to write a series of steps that appropriately solve the requested task based on your expertise. Think step by step. Return your solutions in bullet points."
]

NL_VALIDATOR_INCEPTION_PROMPT = [
    "You are an expert mathematician who focuses on euclidean geometry.",
    "We share a common interest in collaborating to successfully correct the solutions to a geometric problem step by step.",
    "Your main responsibilities include being an strict reviewer, and an efficient solution designer.",
    "You are provided with a task decription and a series of solution steps.",
    f"\nYou base your answers on the following principles: {get_ctt()}"
    "\n\nTo complete the task, you must help me identify any mistakes and then correct them. Validate step by step. Mark as incorrect any step you belive that needs correction."
]

GT_SOLVER_INCEPTION_PROMPT = [
    "You are an helpful assistant who focuses on geometry and has access to specific geometric tools.",
    "We share a common interest in collaborating to successfully solve a problem step by step.",
    "Your main responsibilities include being an reasoner, a planner and a solution designer.",
    "You are also provided with a task decription and a series of solution steps that will give you an initial idea how to solve the problem.",
    f"\n\nHere is a summary of the tools available to you: {''.join(FILTERED_TOOLS)}"
    "\n\nFor your suggestions you can only use the tools provided in the task tool list. "
    "Do not use any other tools. Do not imagine tools that do not exist. "
    "Do not use arbitrary lengths or points in your solutions.",
    "\n\nYou must help me to write a series of steps that appropriately solve the requested task based on your expertise, the expert steps and tools in the list."
]

GT_VALIDATOR_INCEPTION_PROMPT = [
    "You are an expert mathematician who focuses on euclidean geometry and has access to specific geometric tools.",
    "We share a common interest in collaborating to successfully correct the solutions to a geometric problem step by step.",
    "Your main responsibilities include being an strict reviewer, and an efficient solution designer.",
    "You are provided with a task decription and a series of solution steps using geometric tools.",
    f"\n\nHere is a summary of the tools available to you: {''.join(FILTERED_TOOLS)}"
    "\n\nFor your suggestions you can only use the tools provided in the task tool list. "
    "Do not use any other tools. Do not imagine tools that do not exist. "
    "Do not use arbitrary lengths or points in your solutions.",
    "\n\nTo complete the task, you must help me identify any mistakes and then correct them. Validate step by step. Mark as incorrect any step you belive that needs correction."
]


def get_nl_s_prompt():
    return ''.join(NL_SOLVER_INCEPTION_PROMPT)


def get_nl_v_prompt():
    return ''.join(NL_VALIDATOR_INCEPTION_PROMPT)


def get_gt_s_prompt():
    return ''.join(GT_SOLVER_INCEPTION_PROMPT)


def get_gt_v_prompt():
    return ''.join(GT_VALIDATOR_INCEPTION_PROMPT)
