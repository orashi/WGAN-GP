import sys
from itertools import product
import cv2
import lime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



def demoDoG(img, sig=0.5, tau=0.99, phi=10,
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    xdog_params = {
        'kappa': 4.5, 'sigma': sig, 'tau': tau, 'phi': phi,
        'edgeType': lime.NPR_EDGE_XDOG
    }
    xdog = lime.edgeDoG(gray, xdog_params)

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi = 220)

    ax1.axis('off')
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Input')

    ax2.axis('off')
    ax2.imshow(xdog, cmap=cm.gray)
    ax2.set_title('XDoG')

    plt.show()
