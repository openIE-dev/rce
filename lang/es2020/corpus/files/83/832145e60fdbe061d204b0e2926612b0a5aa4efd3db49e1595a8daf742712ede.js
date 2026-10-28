// SPDX-License-Identifier: MIT OR Apache-2.0
// Classes: constructors, methods, getters and setters, static methods,
// inheritance, super and instanceof.
class Shape {
  constructor(name) {
    this.name = name;
    Shape.count = (Shape.count || 0) + 1;
  }
  area() {
    throw new Error("abstract");
  }
  toString() {
    return `${this.constructor.name}(${this.name}, area=${this.area().toFixed(4)})`;
  }
  static describe(s) {
    return "shape " + s.name;
  }
}
class Rect extends Shape {
  constructor(w, h) {
    super("rect");
    this.w = w;
    this.h = h;
  }
  area() {
    return this.w * this.h;
  }
}
class Circle extends Shape {
  constructor(r) {
    super("circle");
    this.r = r;
  }
  area() {
    return Math.PI * this.r ** 2;
  }
  get diameter() {
    return 2 * this.r;
  }
  set diameter(d) {
    this.r = d / 2;
  }
  toString() {
    return "round " + super.toString();
  }
}
const shapes = [new Rect(3, 4), new Circle(1), new Rect(2, 0.5)];
console.log(shapes.map(String).join("; "), Shape.count);
const c = shapes[1];
c.diameter = 10;
console.log(c.r, c.diameter, Shape.describe(c), c instanceof Shape, c instanceof Rect, Object.getPrototypeOf(Circle) === Shape);
console.log(typeof Shape, Shape.name, Rect.prototype.constructor === Rect, Object.keys(c));
try {
  new Shape("abstract").area();
} catch (e) {
  console.log(e.name, e.message);
}
console.log(shapes[0]);
