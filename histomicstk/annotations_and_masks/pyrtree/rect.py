"""
# Modified from: https://code.google.com/archive/p/pyrtree/ .

# Copyright Google
# Under The 3-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""

import math


class Rect:
    """
    A rectangle class that stores: .

    an axis aligned rectangle, and: two flags (swapped_x and swapped_y).
    (The flags are stored implicitly via swaps in the order of
    minx/y and maxx/y.)
    """

    __slots__ = ('x', 'y', 'xx', 'yy', 'swapped_x', 'swapped_y')

    def __init__(self, minx, miny, maxx, maxy):
        """Placeholder."""
        self.swapped_x = (maxx < minx)
        self.swapped_y = (maxy < miny)
        self.x = minx
        self.y = miny
        self.xx = maxx
        self.yy = maxy

        if self.swapped_x:
            self.x, self.xx = maxx, minx
        if self.swapped_y:
            self.y, self.yy = maxy, miny

    def coords(self):
        """Placeholder."""
        return self.x, self.y, self.xx, self.yy

    def overlap(self, orect):
        """Placeholder."""
        return self.intersect(orect).area()

    def write_raw_coords(self, toarray, idx):
        """Placeholder."""
        toarray[idx] = self.x
        toarray[idx + 1] = self.y
        toarray[idx + 2] = self.xx
        toarray[idx + 3] = self.yy
        if (self.swapped_x):
            toarray[idx] = self.xx
            toarray[idx + 2] = self.x
        if (self.swapped_y):
            toarray[idx + 1] = self.yy
            toarray[idx + 3] = self.y

    def area(self):
        """Placeholder."""
        w = self.xx - self.x
        h = self.yy - self.y
        return w * h

    def extent(self):
        """Placeholder."""
        x = self.x
        y = self.y
        return (x, y, self.xx - x, self.yy - y)

    def grow(self, amt):
        """Placeholder."""
        a = amt * 0.5
        return Rect(self.x - a, self.y - a, self.xx + a, self.yy + a)

    def intersect(self, o):
        """Placeholder."""
        if self is NullRect:
            return NullRect
        if o is NullRect:
            return NullRect

        nx, ny = max(self.x, o.x), max(self.y, o.y)
        nx2, ny2 = min(self.xx, o.xx), min(self.yy, o.yy)
        w, h = nx2 - nx, ny2 - ny

        if w <= 0 or h <= 0:
            return NullRect

        return Rect(nx, ny, nx2, ny2)

    def does_contain(self, o):
        """Placeholder."""
        return self.does_containpoint((o.x, o.y)) and self.does_containpoint(
            (o.xx, o.yy))

    def does_intersect(self, o):
        """Placeholder."""
        return (self.intersect(o).area() > 0)

    def does_containpoint(self, p):
        """Placeholder."""
        x, y = p
        return (x >= self.x and x <= self.xx and y >= self.y and y <= self.yy)

    def union(self, o):
        """Placeholder."""
        if o is NullRect:
            return Rect(self.x, self.y, self.xx, self.yy)
        if self is NullRect:
            return Rect(o.x, o.y, o.xx, o.yy)

        x = self.x
        y = self.y
        xx = self.xx
        yy = self.yy
        ox = o.x
        oy = o.y
        oxx = o.xx
        oyy = o.yy

        nx = x if x < ox else ox
        ny = y if y < oy else oy
        nx2 = xx if xx > oxx else oxx
        ny2 = yy if yy > oyy else oyy

        res = Rect(nx, ny, nx2, ny2)

        return res

    def union_point(self, o):
        """Placeholder."""
        x, y = o
        return self.union(Rect(x, y, x, y))

    def diagonal_sq(self):
        """Placeholder."""
        if self is NullRect:
            return 0
        w = self.xx - self.x
        h = self.yy - self.y
        return w * w + h * h

    def diagonal(self):
        """Placeholder."""
        return math.sqrt(self.diagonal_sq())


NullRect = Rect(0.0, 0.0, 0.0, 0.0)
NullRect.swapped_x = False
NullRect.swapped_y = False


def union_all(kids):
    """Placeholder."""
    cur = NullRect
    for k in kids:
        cur = cur.union(k.rect)
    assert not cur.swapped_x
    return cur
