import sympy as sym
import sys

def print_all(expressions):
    print("          {{\n" +",\n".join(["            {{" + ", ".join(["%.1f / %.1f" % sym.fraction(c) for c in expression.all_coeffs()]) + "}}" for expression in expressions])  + "\n          }}," )

def main():
    x = sym.Symbol('x')

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

    # shift
    expressions = [sym.Poly(expressions[i].subs(x, x - (i-start)), x) for i in range(0, deg + 1)]

    for c in range(-start, start + 1):
        temp = [0] * (deg + 1)
        for cc in range(0, deg + 1):
            for i in range(0, deg + 1):
                factor = 1

                for j in range(0, deg + 1):
                    if i!=j:
                        factor *= (c+cc-j)/(i-j)

                temp[i] += round(factor) * expressions[cc]


        print_all(temp)

if __name__ == '__main__':
    main()
