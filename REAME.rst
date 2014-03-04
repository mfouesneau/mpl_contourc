Contourc, a low-level contour computation
=========================================
(a.k.a. MATLAB-Like contourc)

I was myself facing the need of a function that could compute the contours of
an image and give me the polygones without the need to make an actual figure
and extract the curves from the contour object.

However Matplotlib does not provide such function. After quite a long search on
the web, I finally discovered that Matlab provide such function called
`contourc` and that an isse was opened in MPL to implement something, even a
simple hack.

https://github.com/matplotlib/matplotlib/issues/367

Digging a bit more into how MPL computes contours (through `contour` and `contourf`)
I found that the contour calculations are coordinated by the method
_get_allsegs_and_allkinds of classes derived from `:class: ContourSet` (especially
`:class: QuadContourSet` and `:class: TriContourSet`). These methods are fairly
self-contained, but do depend on various things being set up correctly before
they are called, which also includes registering axis.

In this above issue thread, @aaren proposes a quick hack that @ianthomas23
quickly discourage because it bypasses "all reams of consistency checks that
are very important".

In the meantime, we're back to making a dummy figure as only "correct"
solution. As a result, I decided to hack something.

I implemented a new class `:class: ContourEngineContourSet` that sticks as much
as possible to the `"class: ContourSet` to offer the robustness while offering
the low level contour interface.

In addition, I implemented a function,  `:func: contourc` function calculates
the contour matrix for the other contour functions. It is a low-level function
that is not called from the command line.

The values in Z determine the heights of the contour lines with
respect to a plane. The contour calculations use a regularly spaced grid
determined by the dimensions of Z.

`C = contourc(Z)` computes the contour matrix from data in matrix Z, where Z
must be at least a 2-by-2 matrix. The contours are isolines in the units of Z.
The number of contour lines and the corresponding values of the contour lines
are chosen automatically.

`C = contourc(Z, n)` computes contours of matrix Z with n contour levels.

`C = contourc(Z, v)` computes contours of matrix Z with contour lines at the
values specified in vector v. The length of v determines the number of contour
levels. To compute a single contour of level i, use contourc(Z,[i i]).

`C = contourc(x,y,Z)`, `C = contourc(x,y,Z,n)`, and `C = contourc(x,y,Z,v)`
compute contours of Z using vectors x and y to determine the x- and y-axis
limits. x and y must be monotonically increasing.

::python

    import numpy as np
    import pylab as plt
    from matplotlib import rcParams
    from itertools import cycle

    # some exciting test function
    x = np.linspace(-3, 3, 100)
    A, B = np.meshgrid(x, x)
    fab = np.sin(A ** 2) + np.cos(B ** 2)

    # the actual call
    r = contourc(A, B, fab)

    plt.imshow(fab, extent=[x.min(), x.max(), x.min(), x.max()], cmap=plt.cm.gray)

    colors = cycle(rcParams['axes.color_cycle'])
    for level, color in zip(r, colors):
        for pk in level:
            plt.plot(pk[:, 0], pk[:, 1], color=color)
