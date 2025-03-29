import sympy

degree = 6

if degree == 1:
  n_stages = 1

  A = sympy.Matrix([[0]])

  b = sympy.Matrix([1])


elif degree == 2:
  n_stages = 2

  A = sympy.Matrix([[0, 0],
                [sympy.Rational(1, 2), 0]])

  b = sympy.Matrix([0, 1])

elif degree == 3:
  n_stages = 3

  A = sympy.Matrix([[0, 0, 0],
                [sympy.Rational(1,2), 0, 0],
                [-1, sympy.Rational(2), 0]])

  b = sympy.Matrix([sympy.Rational(1,6), sympy.Rational(2,3), sympy.Rational(1,6)])


elif degree == 4:
  n_stages = 4

  A = sympy.Matrix([[0, 0, 0, 0],
                [sympy.Rational(1,2), 0, 0, 0],
                [0, sympy.Rational(1,2), 0, 0],
                [0, 0, sympy.Rational(1), 0]])

  b = sympy.Matrix([sympy.Rational(1,6), sympy.Rational(1,3), sympy.Rational(1,3), sympy.Rational(1,6)])

elif degree == 5:
  n_stages = 6

  A = sympy.Matrix([[0, 0, 0, 0, 0, 0],
                [sympy.Rational(1., 4.), 0, 0, 0, 0, 0],
                [sympy.Rational(1,8), sympy.Rational(1,8), 0, 0, 0, 0],
                [sympy.Rational(0), sympy.Rational(-1, 2), sympy.Rational(1), 0, 0, 0],
                [sympy.Rational(3, 16), sympy.Rational(0), sympy.Rational(0), sympy.Rational(9, 16), 0, 0],
                [sympy.Rational(-3, 7), sympy.Rational(2, 7), sympy.Rational(12, 7), sympy.Rational(-12, 7), sympy.Rational(8, 7), 0]
                ])

  b = sympy.Matrix([sympy.Rational(7, 90), sympy.Rational(0), sympy.Rational(32, 90), sympy.Rational(12, 90), sympy.Rational(32, 90), sympy.Rational(7, 90)])

elif degree == 6:
  n_stages = 7

  A = sympy.Matrix([[0, 0, 0, 0, 0, 0, 0],
                [sympy.Rational(1.0, 3.0), 0, 0, 0, 0, 0, 0],
                [sympy.Rational(0.0), sympy.Rational(2.0, 3.0), 0, 0, 0, 0, 0],
                [sympy.Rational(1.0, 12.0), sympy.Rational(1.0, 3.0), sympy.Rational( -1.0, 12.0), 0, 0, 0, 0],
                [sympy.Rational(-1.0, 16.0), sympy.Rational(9.0, 8.0), sympy.Rational(-3.0, 16.0), sympy.Rational(-3.0, 8.0), 0, 0, 0],
                [sympy.Rational(0.0), sympy.Rational(9.0, 8.0), sympy.Rational(-3.0, 8.0), sympy.Rational(-3.0, 4.0), sympy.Rational(1.0, 2.0), 0, 0],
                [sympy.Rational(9.0, 44.0), sympy.Rational(-9.0, 11.0), sympy.Rational(63.0, 44.0), sympy.Rational(18.0, 11.0), sympy.Rational(0.0), sympy.Rational(-16.0, 11.0), 0]
                ])

  b = sympy.Matrix([sympy.Rational(11, 120), sympy.Rational(0), sympy.Rational(27, 40), sympy.Rational(27, 40), sympy.Rational(-4, 15), sympy.Rational(-4, 15), sympy.Rational(11, 120)])

h = sympy.Symbol('h')

I = sympy.eye(n_stages) / h

INV = (I-A)**-1

result = 1
for i in range(n_stages):
    for j in range(n_stages):
        result += b[i]*INV[i,j]
print(result)

x = sympy.Symbol('x', real=True)
y = sympy.Symbol('y', real=True)
z = x + sympy.I * y

expr = result.subs(h, z)

expr = sympy.simplify(expr)

print(sympy.re(expr))
print(sympy.im(expr))
