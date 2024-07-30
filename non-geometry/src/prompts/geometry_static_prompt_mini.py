# ============ Validation in Geometry Dataset ================= #
MINI_VAL = '''
You will be provided with a problem and an answer. 
You need to check if the answer is correct. 
If the answer is not correct, you need to suggest changes to make the answer valid.
Remember to return the correct answer in \\boxed{} format e.g \\boxed{\\pi * 3} for answer \\pi * 3.
Examples:
Question: How much is the hypotenuse of a right angle with sides 3 and 4?
Proposed Answer:
Assume a = 3 and b = 4 then according to Pythagorean Theorem: c = a^2 + b^2. c = 25
\\boxed{25}.
Correct Answer:
#There are mistakes
Assume a = 3 and b = 4 then according to Pythagorean Theorem: c = sqrt(a^2 + b^2). c = 5
\\boxed{5}.

Question: Right triangle ABC has AB=3, BC=4, AC=5.
Square XYZW is inscribed in ABC with X and Y on AC, W on AB, Z on BC.
What is the side length of the square s?
Proposed Answer:
In the right triangle ABC is h = \\frac{3 * 4}{5} = \\frac{12}{5}
Since triangles ABC and WBZ are similar, we have \\frac{h-s}{s} = \\frac{h}{AC} = \\frac{h}{5} 
Solving for s, we get s = \\frac{5h}{5 + h}, substitute h = \\frac{12}{5} into the equation
s = \\frac{5 \\times \\frac{12}{5}}{5 + \\frac{12}{5}} = \\frac{60}{37}
\\boxed{\\frac{60}{37}}
Correct Answer:
#The answer is correct.
\\boxed{\\frac{60}{37}}
'''.strip()
