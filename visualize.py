#!/usr/bin/env python
import math, argparse, json, colorsys
from tkinter import *

WIDTH = 400
HEIGHT = 300

def color(hue):
    (r,g,b) = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    return "#%02x%02x%02x" % (int(256*r), int(256*g), int(256*b))

class Window(Canvas):

    def __init__(self, master, bins, num_bins, bin_capacity):
        Canvas.__init__(self, master)
        self.master = master
        self.bins = bins
        self.num_bins = num_bins
        self.bin_capacity = bin_capacity
        self.init_window()


    #Creation of init_window
    def init_window(self):
        bin_area = WIDTH * HEIGHT / self.num_bins
        # Width and height of drawn bin
        binw = math.floor(WIDTH / math.ceil(math.sqrt(self.num_bins)))
        binh = math.floor(HEIGHT / math.ceil(math.sqrt(self.num_bins)))
        # Number of bins on x and y axes
        xbins = WIDTH // binw; # round down
        ybins = (self.num_bins + xbins - 1) // xbins; # round up

        # changing the title of our master widget
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        hue = 0.0;
        dhue = 0.05;
        for y in range(ybins):
            for x in range(xbins):
                if(y*xbins + x >= self.num_bins):
                    break;

                # Outline
                self.create_rectangle(x*binw, y*binh, (x+1)*binw, (y+1)*binh)

                # Objs
                top = y*binh;
                for o in self.bins[y*xbins + x]:
                    scaleh = binh * o / self.bin_capacity;
                    self.create_rectangle(x*binw, round(top),
                                          (x+1)*binw, round(top + scaleh),
                                          fill = color(hue));
                    top += scaleh;
                    hue += dhue;

def main():
    parser = argparse.ArgumentParser(description="Visualize packed bins")
    parser.add_argument("-f", dest="in_file", metavar="FILE",
                        help="Input file", required=True)
    args = parser.parse_args()

    with open(args.in_file) as infile:
        bindata = json.load(infile)

    bins = bindata["bins"]
    num_bins = bindata["num_bins"]
    bin_capacity = bindata["bin_size"]

    root = Tk()
    #size of the window
    root.geometry("%dx%d" % (WIDTH, HEIGHT))

    app = Window(root, bins, num_bins, bin_capacity)
    root.mainloop()

if(__name__ == "__main__"):
    main()
