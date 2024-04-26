from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
import argparse

def read_wave(src='./henson.wav'):
    sample_rate, data = wavfile.read(src)
    channel1 = data[:,0]
    print("Min:", channel1.min(), "Max:", channel1.max())
    print("Length:", len(channel1))
    print("Sample rate:", sample_rate)

    return channel1, sample_rate
3
def get_circle(radius=100, points=200):
    circle = []
    for i in range(points):
        circle.append([radius*np.cos(2*np.pi*i/points), radius*np.sin(2*np.pi*i/points)])
    circle = np.array(circle)
    return circle

def build_layer(channel1, expansion=1):
    # get parametric curve
    curved_pts = parametric(channel1)
    # get circle points
    circle = get_circle(radius=200, points=int(len(curved_pts)/2))
    # now devate the points by the curved_pts curve from the center of the circle
    for i in range(0, len(circle), 1):
        # add deviation to x and y coords
        # get unit vector pointing from center to the point
        unit_vector = circle[i]/np.linalg.norm(circle[i])
        # scale unit vector
        unit_vector = unit_vector*expansion
        # scale unit vector by the deviation
        circle[i] = circle[i] + unit_vector*curved_pts[i//2][1]

    # connect end and start point
    circle = np.append(circle, [circle[0]], axis=0)

    # plot and exit
    #plt.plot(circle[:,0], circle[:,1])
    #plt.plot(curved_pts[:,0], curved_pts[:,1])
    #plt.show()
    #exit()
    return circle

def build_model(vase, offset, points_per_layer):
    # vase raw points
    raw = np.concatenate(vase)
    # create list of faces
    faces = []
    # each pair of faces looks like the following
    # face 1 = circle 0: 0, 1, circle 1: 0
    # face 2 = circle 0: 1, circle 1: 0, 1
    # face values are the indices of the points in the raw list
    for i in range(len(vase)-1):
        for j in range(len(vase[i])-1):
            faces.append([i*len(vase[i])+j, i*len(vase[i])+j+1, (i+1)*len(vase[i])+j+1])
            faces.append([i*len(vase[i])+j, (i+1)*len(vase[i])+j+1, (i+1)*len(vase[i])+j])

    # add point in center of bottom layer and top layer
    raw = np.append(raw, [[0,0,0]], axis=0)
    raw = np.append(raw, [[0,0,size-offset]], axis=0)

    # add faces on bottom layer connecting to center point
    for i in range(len(vase[0])-1):
        faces.append([i, len(raw)-2, i+1])
    
    # add faces on top layer connecting to center point
    #for i in range(len(raw)-points_per_layer, len(raw)-1, 1):
    #    faces.append([i, len(raw)-1, i+1])
    
    return faces, raw

def parametric(data):
    # get the first 200 points in data
    line = data[:1000]
    # create 2d parametric curve aling line
    t = np.linspace(0, 1, len(line))
    x = t*10
    y = line
    # make y absolute
    y = np.abs(y)

    # return the curve as one array of [[x,y], ...]
    return np.array([x, y]).T

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Generate 3D model of a vase from an audio file')
    parser.add_argument('-s', type=str, default='', help='Path to audio file')
    parser.add_argument('-o', type=str, default='vase.obj', help='Path to output obj file')

    args = parser.parse_args()
    src = args.s
    out = args.o

    if src == '':
        print("No audio file specified, see --help for more info")
        exit()

    points_per_layer = 2000
    size = points_per_layer * 4
    data, sample_rate = read_wave(src)
    start = int(32.78*sample_rate)
    data = data[start:size+start]
    vase = []
    offset = 400
    expansion = 0.01
    layers = (points_per_layer * 2) // offset
    print("Layers:", layers)
    # calc seconds from data length and sample rate
    seconds = len(data) / sample_rate
    seconds_start = start / sample_rate
    print("Seconds:", seconds_start, "to", seconds_start+seconds)

    for i in range(0, len(data), offset):
        circle = build_layer(data[i:i+offset+1], expansion)
        # convert circle from 2d to 3d by adding z coord being i / 200
        circle = np.append(circle, np.zeros((len(circle),1))+i/8, axis=1)
        vase.append(circle)
    
    # build 3d model
    faces, raw = build_model(vase, offset, points_per_layer)

    # write vertices and faces to obj file
    with open(out, 'w') as f:
        f.write("o vase.1\n")
        for vertex in raw:
            f.write("v " + " ".join([str(x) for x in vertex]) + "\n")
        f.write("\n")
        for face in faces:
            f.write("f " + " ".join([str(x+1) for x in face]) + "\n")

