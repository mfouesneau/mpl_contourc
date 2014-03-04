"""
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
"""
from matplotlib.contour import QuadContourSet
from matplotlib._cntr import Cntr
import numpy as np


class ContourEngineContourSet(object):  # QuadContourSet):
    """
    Low-level contour plot computation

    Calculates the contour matrix C used by contour, contourf and offers a
    robust interface, without plotting interference.

    Attributes
    ----------
    levels: [level0, level1, ..., leveln]
        A list of floating point numbers indicating the level curves to draw;
        eg to draw just the zero contour pass ``levels=[0]``

    origin: [ None | 'upper' | 'lower' | 'image' ]
        If None, the first value of *Z* will correspond to the lower left
        corner, location (0,0). If 'image', the rc value for ``image.origin``
        will be used.

        This keyword is not active if *X* and *Y* are specified in
        the call to contour.

    extent: [ None | (x0,x1,y0,y1) ]
        If *origin* is not *None*, then *extent* is interpreted as in
        :func:`matplotlib.pyplot.imshow`: it gives the outer pixel boundaries.
        In this case, the position of Z[0,0] is the center of the pixel, not a
        corner. If *origin* is *None*, then (*x0*, *y0*) is the position of
        Z[0,0], and (*x1*, *y1*) is the position of Z[-1,-1].

        This keyword is not active if *X* and *Y* are specified in the call to
        contour.

    vmin, vmax: [ None | scalar ]
        If not None, either or both of these values will be supplied to the
        :class:`matplotlib.colors.Normalize` instance.
    """
    def __init__(self, *args, **kwargs):
        """
        The arguments and keyword arguments are described in
        QuadContourSet.contour_doc.
        """
        self.levels = kwargs.get('levels', None)
        self.logscale = kwargs.get('logscale', False)

        self.origin = kwargs.get('origin', None)
        self.extent = kwargs.get('extent', None)

        self._process_args(*args, **kwargs)

        #levels are processed when retrieving the polygones
        #self._process_levels()

    def _process_args(self, *args, **kwargs):
        """
        Process args and kwargs.
        """
        if isinstance(args[0], QuadContourSet):
            C = args[0].Cntr
            if self.levels is None:
                self.levels = args[0].levels
            self.zmin = args[0].zmin
            self.zmax = args[0].zmax
        else:
            x, y, z = self._contour_args(args, kwargs)

            _mask = np.ma.getmask(z)
            if _mask is np.ma.nomask:
                _mask = None
            C = Cntr(x, y, z.filled(), _mask)

        self.Cntr = C

    def _get_allsegs_and_allkinds(self):
        """
        Create and return allsegs and allkinds by calling underlying C code.
        """
        allsegs = []
        allkinds = None
        for level in self.levels:
            nlist = self.Cntr.trace(level)
            nseg = len(nlist) // 2
            segs = nlist[:nseg]
            allsegs.append(segs)
        return allsegs, allkinds

    def _contour_args(self, args, kwargs):
        Nargs = len(args)
        if Nargs <= 2:
            z = np.ma.asarray(args[0], dtype=np.float64)
            x, y = self._initialize_x_y(z)
            args = args[1:]
        elif Nargs <= 4:
            x, y, z = self._check_xyz(args[:3], kwargs)
            args = args[3:]
        else:
            raise TypeError("Too many arguments")
        z = np.ma.masked_invalid(z, copy=False)
        self.zmax = np.ma.maximum(z)
        self.zmin = np.ma.minimum(z)
        if self.logscale and self.zmin <= 0:
            z = np.ma.masked_where(z <= 0, z)
            print('Log scale: values of z <= 0 have been masked')
            self.zmin = z.min()
        self._contour_level_args(z, args)
        return (x, y, z)

    def _check_xyz(self, args, kwargs):
        """
        For functions like contour, check that the dimensions
        of the input arrays match; if x and y are 1D, convert
        them to 2D using meshgrid.

        Possible change: I think we should make and use an ArgumentError
        Exception class (here and elsewhere).
        """
        x, y = args[:2]

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.ma.asarray(args[2], dtype=np.float64)

        if z.ndim != 2:
            raise TypeError("Input z must be a 2D array.")
        else:
            Ny, Nx = z.shape

        if x.ndim != y.ndim:
            raise TypeError("Number of dimensions of x and y should match.")

        if x.ndim == 1:

            nx, = x.shape
            ny, = y.shape

            if nx != Nx:
                raise TypeError("Length of x must be number of columns in z.")

            if ny != Ny:
                raise TypeError("Length of y must be number of rows in z.")

            x, y = np.meshgrid(x, y)

        elif x.ndim == 2:

            if x.shape != z.shape:
                raise TypeError("Shape of x does not match that of z: found "
                                "{0} instead of {1}.".format(x.shape, z.shape))

            if y.shape != z.shape:
                raise TypeError("Shape of y does not match that of z: found "
                                "{0} instead of {1}.".format(y.shape, z.shape))
        else:
            raise TypeError("Inputs x and y must be 1D or 2D.")

        return x, y, z

    def _initialize_x_y(self, z):
        """
        Return X, Y arrays such that contour(Z) will match imshow(Z)
        if origin is not None.
        The center of pixel Z[i,j] depends on origin:
        if origin is None, x = j, y = i;
        if origin is 'lower', x = j + 0.5, y = i + 0.5;
        if origin is 'upper', x = j + 0.5, y = Nrows - i - 0.5
        If extent is not None, x and y will be scaled to match,
        as in imshow.
        If origin is None and extent is not None, then extent
        will give the minimum and maximum values of x and y.
        """
        if z.ndim != 2:
            raise TypeError("Input must be a 2D array.")
        else:
            Ny, Nx = z.shape
        if self.origin is None:  # Not for image-matching.
            if self.extent is None:
                return np.meshgrid(np.arange(Nx), np.arange(Ny))
            else:
                x0, x1, y0, y1 = self.extent
                x = np.linspace(x0, x1, Nx)
                y = np.linspace(y0, y1, Ny)
                return np.meshgrid(x, y)
        # Match image behavior:
        if self.extent is None:
            x0, x1, y0, y1 = (0, Nx, 0, Ny)
        else:
            x0, x1, y0, y1 = self.extent
        dx = float(x1 - x0) / Nx
        dy = float(y1 - y0) / Ny
        x = x0 + (np.arange(Nx) + 0.5) * dx
        y = y0 + (np.arange(Ny) + 0.5) * dy
        if self.origin == 'upper':
            y = y[::-1]
        return np.meshgrid(x, y)

    def _contour_level_args(self, z, args):
        """
        Determine the contour levels and store in self.levels.
        """
        if self.levels is None:
            if len(args) == 0:
                lev = np.linspace(z.min(), z.max(), 7)
            else:
                level_arg = args[0]
                try:
                    if type(level_arg) == int:
                        lev = np.linspace(z.min(), z.max(), level_arg)
                    else:
                        lev = np.asarray(level_arg).astype(np.float64)
                except:
                    raise TypeError( "Last contourc arg must give levels")
            self.levels = lev

    def _process_levels(self):
        """
        Assign values to :attr:`layers` based on :attr:`levels`,
        adding extended layers as needed if contours are filled.

        For line contours, layers simply coincide with levels;
        a line is a thin layer.  No extended levels are needed
        with line contours.
        """
        # The following attributes are no longer needed, and
        # should be deprecated and removed to reduce confusion.
        self.vmin = np.amin(self.levels)
        self.vmax = np.amax(self.levels)
        self.layers = self.levels

    def get_polygons(self, levels=None, logscale=False):
        """ returns coordinate arrays of each contour per level
        Parameters
        ----------
        levels: sequence(float)
            A list of floating point numbers indicating the level
            curves to draw; eg to draw just the zero contour pass
            ``levels=[0]``

        Returns
        -------
        r: sequence
            A list of list of ndarrays. each sublist corresponds one level of
            contours, in which one independent contour is given by 2d
            coordinate array.
        """
        if levels is not None:
            self.levels = levels
        if self.logscale != logscale:
            self.logscale = logscale

        self._process_levels()

        r = self._get_allsegs_and_allkinds()[0]
        return r


def contourc(*args, **kwargs):
    """
    The contourc function calculates the contour matrix for the other contour
    functions. It is a low-level function that is not called from the command line.

    The values in Z determine the heights of the contour lines with
    respect to a plane. The contour calculations use a regularly spaced grid
    determined by the dimensions of Z.

    ::
        C = contourc(Z)

    computes the contour matrix from data in matrix Z, where Z must
    be at least a 2-by-2 matrix. The contours are isolines in the units of Z. The
    number of contour lines and the corresponding values of the contour lines are
    chosen automatically.

    ::
        C = contourc(Z, n)

    computes contours of matrix Z with n contour levels.

    ::
        C = contourc(Z, v)

    computes contours of matrix Z with contour lines at the
    values specified in vector v. The length of v determines the number of contour
    levels. To compute a single contour of level i, use contourc(Z,[i i]).

    ::
        C = contourc(x,y,Z)
        C = contourc(x,y,Z,n)
        C= contourc(x,y,Z,v)

    compute contours of Z using vectors x and y to determine the x- and y-axis
    limits. x and y must be monotonically increasing.

    Parameters
    ----------
    levels: [level0, level1, ..., leveln]
        A list of floating point numbers indicating the level curves to draw;
        eg to draw just the zero contour pass ``levels=[0]``

    origin: [ *None* | 'upper' | 'lower' | 'image' ]
        If *None*, the first value of *Z* will correspond to the lower left
        corner, location (0,0). If 'image', the rc value for ``image.origin``
        will be used.

        This keyword is not active if *X* and *Y* are specified in
        the call to contour.

    extent: [ *None* | (x0,x1,y0,y1) ]
        If *origin* is not *None*, then *extent* is interpreted as in
        :func:`matplotlib.pyplot.imshow`: it gives the outer pixel boundaries.
        In this case, the position of Z[0,0] is the center of the pixel, not a
        corner. If *origin* is *None*, then (*x0*, *y0*) is the position of
        Z[0,0], and (*x1*, *y1*) is the position of Z[-1,-1].

        This keyword is not active if *X* and *Y* are specified in the call to
        contour.
    """
    C = ContourEngineContourSet(*args, **kwargs)
    return C.get_polygons()


def example_contourc():
    """ Example Usage """
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
