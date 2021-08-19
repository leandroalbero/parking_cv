import xml.dom.minidom

import cv2
import numpy as np
import neural as neur
import csv
import xml.etree.ElementTree as et
import xml.dom.minidom as md

class Parking:
    """
    Representa el estado de un aparcamiento completo

    ...

    Atributos
    _________
    plazas : Plaza

    Métodos
    _______
    insert_coord(x, status)
    """

    def __init__(self, save_name=None):
        self.plazas = []
        if save_name:
            self.load_state(save_name)

    def insert_coord(self, x, status):
        if len(self.plazas) == 0:
            self.plazas.append(Plaza([x], status, len(self.plazas)))
        else:
            if len(self.plazas[-1].coords) < 4:
                self.plazas[-1].coords.append(x)
            else:
                print("Insertando plaza")
                print(self.plazas[-1].coords)
                self.plazas.append(Plaza([x], status, len(self.plazas)))

    def save_state(self, name):
        root = et.Element("parking")
        root.attrib = {"id": name}
        for plaza in self.plazas:
            space = et.Element("space")
            space.attrib = {"id": str(plaza.id), "occupied": str(plaza.status)}
            contour = et.Element("contour")
            for coord in plaza.coords:
                point = et.Element("point")
                point.attrib = {"x": str(coord[0]), "y": str(coord[1])}
                contour.append(point)
            space.append(contour)
            root.append(space)
        tree = et.ElementTree(root)
        f = open("text.xml", "wb")
        tree.write(f)

    def load_state(self, name):
        with open(name, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                id_plaza = row[0]
                if row[1] == 'True':
                    plaza_ocupada = True
                else:
                    plaza_ocupada = False
                plaza_coord = []
                for i in range(2, len(row)):
                    temp = (row[i].strip('][').split(','))
                    plaza_coord.append([int(temp[0]), int(temp[1])])
                self.plazas.append(Plaza(plaza_coord, plaza_ocupada, id_plaza))


class Plaza:
    """
    Representa el estado de una plaza de aparcamiento

    ...

    Atributos
    _________


    Métodos
    _______

    """

    def __init__(self, coords, status, num):
        self.coords = coords
        self.status = status
        self.id = num

    def change_state(self, status):
        self.status = status

    def add_coord(self, coordinates):
        self.coords = coordinates

    def get_coord(self):
        return self.coords


def click_event(event, x, y, z, t):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        c1 = [x, y]
        p1.insert_coord(c1, True)
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        c1 = [x, y]
        p1.insert_coord(c1, False)
    if event == cv2.EVENT_MBUTTONDOWN:
        print("Terminado")
        extract_patches(img_copy, p1.plazas)
        draw_boxes(img, p1.plazas)
        p1.save_state("parking1")


def draw_boxes(_img, plazas):
    if len(plazas) == 0:
        cv2.imshow('lines', _img)
        return _img
    for plaza in plazas:
        np_plaza_coords = np.array(plaza.coords)
        if plaza.status:
            _img = cv2.polylines(_img, np.int32([np_plaza_coords]), True, (0, 0, 255), 2)
        else:
            _img = cv2.polylines(_img, np.int32([np_plaza_coords]), True, (0, 255, 0), 2)
    cv2.imshow('lines', img)
    return img


def extract_patches(_img, plazas):
    for plaza in plazas:
        patch = move_poly(_img, plaza.coords, True)
        resized = cv2.resize(patch, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("plazas/" + str(plaza.id) + ".jpg", resized)
        print()


def calculate_bounding_box(coords):
    minx = np.min(coords, axis=0)[0]
    miny = np.min(coords, axis=0)[1]
    maxx = np.max(coords, axis=0)[0]
    maxy = np.max(coords, axis=0)[1]
    return [maxy - miny, maxx - minx]


def move_poly(_img, coords, square):
    bound_box = calculate_bounding_box(coords)
    offset = np.min(coords, axis=0)
    mask = np.zeros(np.array(_img.shape), dtype=np.uint8)
    cv2.fillPoly(mask, pts=[np.array(coords)], color=(1, 1, 1))
    if square:
        bigger_side = np.max(bound_box)
        newimg = np.zeros([bigger_side, bigger_side, 3], dtype=np.uint8)
    else:
        newimg = np.zeros([bound_box[0], bound_box[1], 3], dtype=np.uint8)
    for i in range(bound_box[0]):
        for j in range(bound_box[1]):
            newimg[i][j] = mask[i + offset[1]][j + offset[0]] * _img[i + offset[1]][j + offset[0]]
    return newimg


if __name__ == "__main__":
    img = cv2.imread("lot.jpg", 1)
    img_copy = cv2.imread("lot.jpg", 1)
    p1 = Parking("parking1")
    draw_boxes(img, p1.plazas)
    cv2.setMouseCallback('lines', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()