from asyncio import constants
from re import A
import numpy as np
import pygame as pg
from math import sin, cos, tan

pi = 3.1415927
white = (255,255,255)
red = (255,0,0)

surface = pg.display.set_mode((1000,600))
clock = pg.time.Clock()

cameraPos = [0, -100, 0]
cameraRot = [0, 0, 0]   #pitch, roll, yaw

verticalFOV = 60
vFOVrad = verticalFOV * pi/180  #convert to radians
hFOVrad = np.arctan((np.tan(vFOVrad/2) * pg.display.get_surface().get_size()[0] / 2) / (pg.display.get_surface().get_size()[1] / 2)) * 2
cameraToProjectionPlane = 10    #puts prjection plane this far infront of camera
zoom = 40


#calculate normals of the 4 walls of the fov
extremesOfFOV = [[np.tan(hFOVrad/2) + np.sin(cameraRot[2]), 1, np.tan(vFOVrad/2) + np.sin(cameraRot[0])], [np.tan(hFOVrad/2)  + np.sin(cameraRot[2]), 1, -np.tan(vFOVrad/2) + np.sin(cameraRot[0])], [-np.tan(hFOVrad/2)  + np.sin(cameraRot[2]), 1, -np.tan(vFOVrad/2) + np.sin(cameraRot[0])], [-np.tan(hFOVrad/2)  + np.sin(cameraRot[2]), 1, np.tan(vFOVrad/2) + np.sin(cameraRot[0])]]
normalsList = []
for i in range(4):
    normals = np.cross(extremesOfFOV[i], extremesOfFOV[(i + 1) % 4])
    normalsList.append(normals)

nearPlaneNormal = [cameraToProjectionPlane * np.tan(cameraRot[2]), cameraToProjectionPlane, cameraToProjectionPlane * np.tan(cameraRot[0])]
'''
x - right (+) and left (-)
y - forwards (+) and backwards (-)
z - up (+) and down (-)

theta is rotation around z axis (pointing left and right)
phi is rotation in the x axis (pointing up and down)
'''

def plane(size, pos = (0,0,-30), rot = (0.2,0,0)):    #size is the top right corner, pos is where its centred, angleX is rotated on x-axis, angleY is rotation on y-axis
    #planeCorners = [(size + pos[0], size + pos[1], pos[2]),(size + pos[0], -size + pos[1], pos[2]),(-size + pos[0], -size + pos[1], pos[2]),(-size + pos[0], size + pos[1], pos[2])]    #define a sqare plane
    #planeCorners = [[size + pos[0],size + pos[1],pos[2]], [size + pos[0],-size + pos[1],pos[2]], [-size + pos[0],-size + pos[1],pos[2]], [-size + pos[0],size + pos[1],pos[2]]]
    planeCorners = [[[1],[1],[0]], [[1],[-1],[0]], [[-1],[-1],[0]], [[-1],[1],[0]]]
    pointOrder = [[0,1],[1,2],[2,3],[3,0]]
    planeCorners2 = []
    planeCorners4 = []
    xRot = [[cos(rot[0]), -sin(rot[0]), 0],
            [sin(rot[0]), cos(rot[0]),  0],
            [0,           0,            1]]
    for i in range(len(planeCorners)):
        for j in range(3):
            planeCorners[i][j][0] *= size
        planeCorners1 = matrixAdd(planeCorners[i], [[pos[0]],[pos[1]],[pos[2]]])     
        planeCorners2.append(planeCorners1) 
        planeCorners3 = matrixMult(xRot, planeCorners2[i])
        planeCorners4.append(planeCorners3)
    print(planeCorners4)
    #print(planeCorners2)

    
    for i in range(len(planeCorners4)):
        lineEndPoints = [planeCorners4[pointOrder[i][0]], planeCorners4[pointOrder[i][1]]]
        #print(lineEndPoints)
        screenPointList = projection(lineEndPoints)
        if screenPointList != []:
            pg.draw.line(surface, white, screenPointList[0],screenPointList[1], 1) 


def projection(point): #takes in a list of tuples of theta and phi and returns the list as coordinates
    screenPointList = []
    for i in range(len(point)):        
        projectedPoint = cameraToProjectionPlane * (point[i][0][0] - cameraPos[0]) / (cameraToProjectionPlane + point[i][1][0] - cameraPos[1]) , cameraToProjectionPlane * (point[i][2][0] - cameraPos[2]) / (cameraToProjectionPlane + point[i][1][0] - cameraPos[1])
        screenPoint = (projectedPoint[0]) * zoom + pg.display.get_surface().get_size()[0]/2, (-1*projectedPoint[1]) * zoom + pg.display.get_surface().get_size()[1]/2
        screenPointList.append(screenPoint)
    return screenPointList


def matrixAdd(A, B):
    #A and B are in the form [[x],[y],[z]]
    result = [[A[i][j] + B[i][j]  for j in range(len(A[0]))] for i in range(len(B))]
    resultList = []
    for r in result:
        resultList.append(r)
    return resultList

def matrixMult (A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
      print("Cannot multiply the two matrices. Incorrect dimensions.")
      return
    # Create the result matrix
    # Dimensions would be rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C



mouseX=0
mouseY=0
scrollZoom = 0
count = 0

while True:
    clock.tick(120)
    surface.fill((0,0,0))
    keys = pg.key.get_pressed()
    #cameraRot[2] += 0.01
    #print(cameraRot)
    #for i in range(1000):
    
    plane(50, (0,0,0), (count,0,0))
    #plane(20, (20,0,20))
    #plane(30, (0,0,50))
    count += 0.001

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
        
        elif pg.mouse.get_pressed()[0] != 0:
            mouseX = pg.mouse.get_pos()[0]
            mouseY = pg.mouse.get_pos()[1]
            cameraPos[0] = (-mouseX + pg.display.get_surface().get_size()[0] / 2) / 5
            cameraPos[2] = (mouseY-200) / 5

        elif event.type == pg.MOUSEWHEEL:
            cameraPos[1] += event.y*5

        elif keys[pg.K_w]:
            cameraPos[1] += 2
        elif keys[pg.K_a]:
            cameraPos[0] += 2
        elif keys[pg.K_s]:
            cameraPos[1] -= 2
        elif keys[pg.K_d]:
            cameraPos[0] -= 2
        elif keys[pg.K_SPACE]:
            cameraPos[2] += 2
        elif keys[pg.K_LSHIFT]:
            cameraPos[2] -= 2

    pg.display.update()    