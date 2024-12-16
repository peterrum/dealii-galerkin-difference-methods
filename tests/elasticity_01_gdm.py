import sympy as sym

a = sym.Symbol('a')
x = sym.Symbol('x')
y = sym.Symbol('y')

u = [sym.sin(a * x) * sym.sin(a * x) * sym.cos(a * y) * sym.sin(a * y), 
    -sym.cos(a * x) * sym.sin(a * x) * sym.sin(a * y) * sym.sin(a * y)]

eps = [[sym.diff(u[0],x), (sym.diff(u[0],y) + sym.diff(u[1],x)) / 2], 
       [(sym.diff(u[0],y) + sym.diff(u[1],x)) / 2, sym.diff(u[1],y)]]

f = [-2*(sym.diff(eps[0][0],x)+ sym.diff(eps[0][1],y)), 
     -2*(sym.diff(eps[1][0],x)+ sym.diff(eps[1][1],y))]

print(f)