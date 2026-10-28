# SPDX-License-Identifier: MIT OR Apache-2.0
# Classes: __init__, methods, class attributes, inheritance, super,
# properties, __repr__, __eq__ and operator methods.
import math


class Shape:
    count = 0

    def __init__(self, name):
        self.name = name
        Shape.count += 1

    def area(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, area={self.area():.4f})"


class Rect(Shape):
    def __init__(self, w, h):
        super().__init__("rect")
        self.w, self.h = w, h

    def area(self):
        return self.w * self.h


class Circle(Shape):
    def __init__(self, r):
        super().__init__("circle")
        self.r = r

    def area(self):
        return math.pi * self.r ** 2

    @property
    def diameter(self):
        return 2 * self.r


class Vec:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y)

    def __mul__(self, k):
        return Vec(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __eq__(self, o):
        return isinstance(o, Vec) and (self.x, self.y) == (o.x, o.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Vec({self.x}, {self.y})"


shapes = [Rect(3, 4), Circle(1), Rect(2, 0.5)]
print(shapes, Shape.count)
print(sum(s.area() for s in shapes), shapes[1].diameter)
print(isinstance(shapes[0], Shape), issubclass(Circle, Rect), type(shapes[1]).__mro__)
v = Vec(1, 2) + Vec(3, 4)
print(v, v * 2, 3 * v, v == Vec(4, 6), v != Vec(0, 0), len({Vec(1, 1), Vec(1, 1)}))
try:
    Shape("abstract").area()
except NotImplementedError:
    print("NotImplementedError")
