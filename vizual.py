# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import math
import os.path
import getpass

viz = Visdom()

# contour
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
viz.contour(X=X, opts=dict(colormap='Viridis'))
#
# # surface
# viz.surf(X=X, opts=dict(colormap='Hot'))
#
# # line plots
# viz.line(Y=np.random.rand(10))
#
# Y = np.linspace(-5, 5, 100)
# viz.line(
#     Y=np.column_stack((Y * Y, np.sqrt(Y + 5))),
#     X=np.column_stack((Y, Y)),
#     opts=dict(markers=False),
# )
#
# # line updates
# win = viz.line(
#     X=np.column_stack((np.arange(0, 10), np.arange(0, 10))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
# )
# viz.line(
#     X=np.column_stack((np.arange(10, 20), np.arange(10, 20))),
#     Y=np.column_stack((np.linspace(5, 10, 10), np.linspace(5, 10, 10) + 5)),
#     win=win,
#     update='append'
# )
# viz.updateTrace(
#     X=np.arange(21, 30),
#     Y=np.arange(1, 10),
#     win=win,
#     name='2'
# )
# viz.updateTrace(
#     X=np.arange(1, 10),
#     Y=np.arange(11, 20),
#     win=win,
#     name='4'
# )
#
# Y = np.linspace(0, 4, 200)
# win = viz.line(
#     Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
#     X=np.column_stack((Y, Y)),
#     opts=dict(
#         fillarea=True,
#         legend=False,
#         width=400,
#         height=400,
#         xlabel='Time',
#         ylabel='Volume',
#         ytype='log',
#         title='Stacked area plot',
#         marginleft=30,
#         marginright=30,
#         marginbottom=80,
#         margintop=30,
#     ),
# )
