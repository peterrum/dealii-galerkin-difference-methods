import sympy as sym
import sys


def print_all(expressions):
    print(
        "          {{\n"
        + ",\n".join(
            [
                "            {{"
                + ", ".join(
                    ["%.1f / %.1f" % sym.fraction(c) for c in expression.all_coeffs()]
                )
                + "}}"
                for expression in expressions
            ]
        )
        + "\n          }},"
    )


def reference_node(node, c, deg):
    if c == 0:
        return sym.Rational(2 * node, 1)

    if c == deg:
        return sym.Rational(2 * (node - deg) + 1, 1)

    return sym.Rational(node - c, 1) + sym.Rational(1, 2)


def main():
    x = sym.Symbol("x")

    deg = int(sys.argv[1])
    start = int(deg / 2)

    if deg % 2 != 0:
        raise ValueError("DGGD uses even polynomial degree.")

    for c in range(0, deg + 1):
        expressions = []

        # natural stencil for this cell
        stencil = list(range(c - start, c + start + 1))

        # reference coordinates of the stencil nodes
        nodes = [reference_node(stencil[i], c, deg) for i in range(0, deg + 1)]

        # build natural Lagrange basis on this cell
        for i in range(0, deg + 1):
            temp = 1

            for j in range(0, deg + 1):
                if i != j:
                    temp *= (x - nodes[j]) / (nodes[i] - nodes[j])

            expressions = expressions + [sym.Poly(sym.expand(temp), x)]

        # boundary modification / ghost extrapolation
        temp = [0] * (deg + 1)

        for cc in range(0, deg + 1):
            target = stencil[cc]

            for i in range(0, deg + 1):
                factor = 1

                for j in range(0, deg + 1):
                    if i != j:
                        factor *= sym.Rational(target - j, i - j)

                temp[i] += sym.simplify(factor) * expressions[cc].as_expr()

        temp = [sym.Poly(sym.expand(t), x) for t in temp]

        print_all(temp)


if __name__ == "__main__":
    main()
