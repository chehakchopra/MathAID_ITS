import re
import sympy as sp

def generate_hint(question_text, answer):
    """
    Very simple rule-based hint generator.
    It looks at the question text and gives a structured hint.
    You can expand this over time for more topics.
    """
    if not question_text:
        return "Break the problem into smaller steps and identify what is given and what you must find."

    q = question_text.lower()

    if "solve" in q and "x" in q and ("x^2" not in q and "x²" not in q):
        # Try to parse a pattern like: 3x+4=19
        cleaned = q.replace(" ", "")
        m = re.search(r"(-?\d+)x([+-]\d+)=(-?\d+)", cleaned)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            c = int(m.group(3))
            return (
                f"Move {b} to the other side first → {a}x = {c - b}. "
                f"Then divide both sides by {a}."
            )
        return "Isolate x: move constants to the other side, then divide by the coefficient of x."

    if "x^2" in q or "x²" in q:
        if "solve" in q:
            return "Try to factor the quadratic into (x - r1)(x - r2) = 0, then set each bracket to zero."
        return "Look for a common pattern like (x + a)(x + b) that multiplies to the constant term."

    if "root" in q or "roots" in q or "polynomial" in q:
        return "Try small integer values (−3, −2, −1, 0, 1, 2, 3, …) to see which make the polynomial equal to 0."

    if "d/dx" in q or "differentiate" in q or "derivative" in q:
        return "Use the power rule: d/dx (axⁿ) = a·n·xⁿ⁻¹. Differentiate each term separately."

    if "∫" in q or "integral" in q or "integrate" in q:
        if "from" in q or "limits" in q:
            return "First find the antiderivative, then plug in the upper limit minus the lower limit."
        return "Find the antiderivative: for xⁿ, add 1 to the power and divide by the new power."


    if "area" in q and "rectangle" in q:
        return "Area of a rectangle = length × width."

    if "area" in q and ("triangle" in q or "base" in q and "height" in q):
        return "Area of a triangle = 1/2 × base × height."

    if "hypotenuse" in q or "right triangle" in q or "pythagoras" in q:
        return "Use Pythagoras: c² = a² + b², so c = √(a² + b²)."

    if "mean" in q or "average" in q:
        return "Add all the values together and divide by how many values there are."

    if "probability" in q or "chance" in q:
        return "Probability = number of favourable outcomes ÷ total number of possible outcomes."

    # Default generic hint
    return "Identify what is given, what you must find, and try to write an equation that connects them."
