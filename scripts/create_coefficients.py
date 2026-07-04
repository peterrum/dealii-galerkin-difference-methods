import sympy as sym
import sys
import numpy as np
import matplotlib.pyplot as plt

x = sym.Symbol('x')

def print_all(expressions):
    print("          {{\n" +",\n".join(["            {{" + ", ".join(["%.1f / %.1f" % sym.fraction(c) for c in expression.all_coeffs()]) + "}}" for expression in expressions])  + "\n          }}," )

def plot(expressions):
    xs = np.linspace(0, 1, 500)

    plt.figure(figsize=(6, 4))

    for i, poly in enumerate(expressions):
        f = sym.lambdify(x, poly.as_expr(), "numpy")
        plt.plot(xs, f(xs), label=f"p{i}")

    plt.xlim(0, 1)
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    deg = int(sys.argv[1])
    start = int(deg/2)

    expressions = []

    for i in range(-deg, 1):
        temp = 1
        for j in range(0, deg + 1):
            if i + j != 0:
                temp *= (x+i+j)
        temp = sym.Poly(temp / round(temp.evalf(subs={x: 0})))
        expressions = expressions + [temp]

    print_all(expressions)
    print()
    print()
    print()

    # shift
    expressions = [sym.Poly(expressions[i].subs(x, x - (i-start)), x) for i in range(0, deg + 1)]

    print_all(expressions)
    print()
    print()
    print()

    for c in range(-start, start + 1):
        temp = [0] * (deg + 1)
        for cc in range(0, deg + 1):
            for i in range(0, deg + 1):
                factor = 1

                for j in range(0, deg + 1):
                    if i!=j:
                        factor *= sym.Rational(c+cc-j,i-j)

                temp[i] += factor * expressions[cc]

        if deg % 2 == 0:
            if c == -start:
                temp = [sym.Poly(expression.subs(x, x /2), x) for expression in temp]
            elif c == +start:
                temp = [sym.Poly(expression.subs(x, (x-1) /2), x) for expression in temp]
            else:
                temp = [sym.Poly(expression.subs(x, x - sym.Rational(1, 2)), x) for expression in temp]


        plot(temp)
        print_all(temp)

if __name__ == '__main__':
    main()
