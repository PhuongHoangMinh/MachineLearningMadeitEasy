import numpy as np
from math import log, ceil, sqrt, sin, cos


def bisection(f, x1, x2, tol = 1.0e-9):
    f1 = f(x1)
    if f1 == 0.0: return x1
    f2 = f(x2)
    if f2 == 0.0: return x2
    if f1*f2 > 0.0: print("Root is no within the range %f to %f" %(x1,x2))
    n = ceil(log(abs(x2-x1)/tol)/log(2.0))
    for i in range(n):
        x3 = 0.5*(x1+x2); f3 = f(x3)
        print("root by Bisection at iteration: %d is : %f" % (i, x3))
        if f3 == 0.0:return x3
        if f2*f3 < 0.0 : x1 = x3; f1 = f3
        else:            x2 = x3; f2 = f3

    return (x1+x2)/2.0

def newtonRaphson(f, df, a, b, tol = 1.0e-9):
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if fa*fb > 0.0: print("root is not in the range a: %f  to b: %f" %(a,b))
    x = 0.5*(a+b)
    for i in range(30):
        fx = f(x)
        if abs(fx) < tol: return x

        if fa*fx < 0.0:
            b = x
        else:
            a = x

        #Try a newton-raphson step
        dfx = df(x)
        try: dx = -fx/dfx
        except ZeroDivisionError: dx = b - a
        x = x + dx

        if(b-x)*(x-a) < 0.0:
            dx = 0.5*(b-a)
            x  = a + dx

        print("root at iteration: %d is : %f"%(i, x))

        if abs(dx) < tol*max(abs(b),1.0): return x

##module newtonRaphson2
''' soln = newtonRaphson2(f,x,tol = 1.0e-9)
    solves the simultaneous equations f(x) = 0 by
    the Newton-Raphson method
    {f} and {x} are vectors
'''
def jacobian(f,x):
    h = 1.0e-4
    n = len(x)
    jac = np.zeros((n,n))
    f0 = f(x)
    for i in range(n):
        temp = x[i]
        x[i] = temp + h
        f1 = f(x)
        x[i] = temp
        jac[:, i] = (f1- f0)/h
    return jac, f0

def newtonRaphson2(f,x , tol= 1.0e-9):
    for i in range(60):
        jac, f0 = jacobian(f,x)
        if sqrt(np.dot(f0, f0) / len(x)) < tol: return x
        dx =  np.linalg.solve(jac, -f0)
        x  =  x  + dx
        if sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)), 1.0):
            return x


def myFunction(x):
    return x*x*x - 10*x*x + 5

def thoFunction(x):
    return 12*x*x*x - 20*x*x + 21*x - 10

def myDerivative(x):
    return 3*x*x - 20*x

# root = newtonRaphson(myFunction, myDerivative, 0.4, 6.0)
# print("root solution is: ", root)

# roBisection = bisection(myFunction, 0.4, 6.0)
# print("root solution by bisection is: ", roBisection)
#
# print("value of function at root is: ", myFunction(0.7346))


thoRootBisection = bisection(thoFunction, 0.0, 6.0)
print("root solution by bisection is: ", thoRootBisection)

print("value of function at root is: ", thoFunction(0.7879352))

def f(x):
    f = np.zeros(len(x))
    f[0] = sin(x[0]) + x[1]**2 + log(x[2]) - 7.0
    f[1] = 3.0*x[0] + 2.0**x[1] - x[2]**3 + 1.0
    f[2] = x[0] + x[1] + x[2] - 5.0
    return f

def Func(x):
    f = np.zeros(len(x))
    f[0] = sin(x[0]) + 3*cos(x[0]) - 2
    f[1] = cos(x[0]) - sin(x[1]) + 0.2
    return f

def IntersectionCircleLine(x):
    f = np.zeros(len(x))
    f[0] = x[0]**2 + x[1]**2 - 3
    f[1] = x[0]*x[1] -1
    return f

x = np.array([1.0, 1.0, 1.0])
print (newtonRaphson2(f,x))

NoX = np.array([0.5, 1.5])
print(newtonRaphson2(IntersectionCircleLine, NoX))