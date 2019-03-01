{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Stats of batch #3:\n",
      "# of Samples: 10000\n",
      "\n",
      "Label Counts of [0](AIRPLANE) : 994\n",
      "Label Counts of [1](AUTOMOBILE) : 1042\n",
      "Label Counts of [2](BIRD) : 965\n",
      "Label Counts of [3](CAT) : 997\n",
      "Label Counts of [4](DEER) : 990\n",
      "Label Counts of [5](DOG) : 1029\n",
      "Label Counts of [6](FROG) : 978\n",
      "Label Counts of [7](HORSE) : 1015\n",
      "Label Counts of [8](SHIP) : 961\n",
      "Label Counts of [9](TRUCK) : 1029\n",
      "\n",
      "Example of Image 9999:\n",
      "Image - Min Value: 3 Max Value: 242\n",
      "Image - Shape: (32, 32, 3)\n",
      "Label - Label Id: 1 Name: automobile\n",
      "Trying to restore last checkpoint ...\n",
      "Failed to restore checkpoint. Initializing variables instead.\n",
      "this also works for us\n"
     ]
    }
   ],
   "source": [
    "from LENETCifarScript import MEFullyConnected1Graphs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plotDiff(ME1, ME2):\n",
    "    x1 = np.arange(len(ME1))\n",
    "    x2 = np.arange(len(ME2))\n",
    "   # x3 = np.arange(len(ME3))\n",
    "    plt.bar(x1, ME1, color = 'black', align = 'center', alpha = 1)\n",
    "    plt.bar(x2, ME2, color = 'red', align = 'center', alpha = 0.7)\n",
    "    #plt.bar(x3, ME3, color = 'green', align = 'center', alpha = 0.6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plotDiff(MEFullyConnected1Graphs[0], MEFullyConnected1Graphs[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
