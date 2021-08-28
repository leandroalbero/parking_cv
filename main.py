import os
import cv2
import numpy as np
import csv
import xml.etree.ElementTree as et
import train
import predict


class Parking:
    """
    Representa el estado de un aparcamiento completo

    ...

    Atributos
    _________
    save_name : Nombre del archivo .xml en el que se guardará el estado del parking (coordenadas, id y estado).
    id:         Identificador numérico del parking (Se usa para la función que escanea todos los parkings y extrae
    todas sus plazas en recuadros de 150x150.
    plazas :    Contenedor de todas las plazas en el aparcamiento.
    image :     Imagen con la que se ha calculado el estado actual del parking.


    Métodos
    _______
    insert_coord(x, status)
    save_state(name)
    load_state(name)
    extract_patches(_img, plazas, savename=None, folder=None)
    click_event(self, event, x, y, z, t):
    draw_boxes(self, _img=None):
    update_state_from_photo(self, route_to_img):
    create_xml(self, image, savefile):
    """

    def __init__(self, save_name=None, identifier=None, image=None):
        self.save_name = ""
        self.id = identifier
        self.plazas = []
        self.image = image
        if save_name:
            self.save_name = save_name
            if os.path.exists(save_name):
                self.load_state(save_name)
            else:
                if image is None:
                    image = input("Please specify a route to the image: ")
                self.create_xml(image, save_name)

    def insert_coord(self, x, status):
        if len(self.plazas) == 0:
            self.plazas.append(_Plaza([x], status, len(self.plazas)))
        else:
            if len(self.plazas[-1].coords) < 4:
                self.plazas[-1].coords.append(x)
            else:
                print("Insertando plaza")
                print(self.plazas[-1].coords)
                self.plazas.append(_Plaza([x], status, len(self.plazas)))

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
        f = open(self.save_name, "wb")
        tree.write(f)

    def load_state(self, name):
        parser = et.parse(name)
        root = parser.getroot()
        for x in root.findall("space"):
            id_plaza = int(x.attrib.get("id"))
            plaza_ocupada = x.attrib.get("occupied")
            plaza_coord = []
            contour = x.find("contour")
            for y in contour.findall("point"):
                plaza_coord.append([int(y.attrib.get("x")), int(y.attrib.get("y"))])
            self.plazas.append(_Plaza(plaza_coord, plaza_ocupada, id_plaza))

    def extract_patches(self, _img, plazas, savename=None, folder=None):
        if folder is None:
            path_to_folder = "temp/"
            for root, dirs, files in os.walk(path_to_folder):
                for file in files:
                    os.remove(os.path.join(root, file))
        else:
            path_to_folder = folder
        estados = []
        for plaza in plazas:
            if len(plaza.coords) == 0:
                break
            patch = plaza.move_poly(_img, True)
            resized = cv2.resize(patch, (150, 150), interpolation=cv2.INTER_CUBIC)
            estados.append([str(plaza.id), str(plaza.status)])
            cv2.imwrite(f"{path_to_folder}{str(savename)}_{str(plaza.id)}-{str(plaza.status)}.jpg", resized)

    def click_event(self, event, x, y, z, t):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            c1 = [x, y]
            self.insert_coord(c1, True)
        if event == cv2.EVENT_RBUTTONDOWN:
            print(x, ' ', y)
            c1 = [x, y]
            self.insert_coord(c1, False)
        if event == cv2.EVENT_MBUTTONDOWN:
            print("Terminado")
            self.extract_patches(cv2.imread(self.image, 1), self.plazas)
            self.draw_boxes(cv2.imread(self.image, 1))
            self.save_state(self.save_name)

    def draw_boxes(self, _img=None):
        if _img is None:
            _img = cv2.imread(self.image)
        if len(self.plazas) == 0:
            cv2.imshow('lines', _img)
            return _img
        for plaza in self.plazas:
            np_plaza_coords = np.array(plaza.coords)
            if plaza.status == "0":
                _img = cv2.polylines(_img, np.int32([np_plaza_coords]), True, (0, 255, 0), 2)
            else:
                _img = cv2.polylines(_img, np.int32([np_plaza_coords]), True, (0, 0, 255), 2)
        cv2.imshow('lines', _img)
        return _img

    def update_state_from_photo(self, route_to_img):
        self.image = route_to_img
        img = cv2.imread(route_to_img, 1)
        self.extract_patches(img, self.plazas, savename=self.id)
        results = predict.predict_image("temp/")
        header = list(results.keys())[0].split('_')[0]
        for plaza in self.plazas:
            plaza.status = str(results[f"{header}_{plaza.id}-{plaza.status}.jpg"])
        pass

    def create_xml(self, image, savefile):
        img = cv2.imread(image, 1)
        img_copy = cv2.imread(image, 1)
        self.draw_boxes(img)
        cv2.setMouseCallback('lines', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass


    def print_overview(self):
        plazas_ocupadas = 0
        plazas_vacias = 0
        estados = []
        for plaza in self.plazas:
            estados.append(plaza.status)
            if plaza.status == '1':
                plazas_ocupadas += 1
            else:
                plazas_vacias += 1
        print(f"Plazas ocupadas: {plazas_ocupadas}, Plazas vacías: {plazas_vacias}, Plazas totales: {len(self.plazas)}")
        print(estados)


class _Plaza:
    """
    Representa el estado de una plaza de aparcamiento
    723790 plazas en PKlot (~4GB a 150x150)
    ...

    Atributos
    _________
    coords: Coordenadas de los 4 puntos que conforman una plaza.
    status: Estado de la plaza (0: vacío, 1: lleno).
    id:     Identificador numérico de la plaza

    Métodos
    _______
    change_state(self, status):
    add_coord(self, coordinates):
    get_coord(self):
    calculate_bounding_box(self):
    move_poly(self, _img, square):
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

    def calculate_bounding_box(self):
        try:
            minx = np.min(self.coords, axis=0)[0]
            miny = np.min(self.coords, axis=0)[1]
            maxx = np.max(self.coords, axis=0)[0]
            maxy = np.max(self.coords, axis=0)[1]
            return [maxy - miny, maxx - minx]
        except:
            return [1, 1]

    def move_poly(self, _img, square):
        bound_box = self.calculate_bounding_box()
        offset = np.min(self.coords, axis=0)
        mask = np.zeros(np.array(_img.shape), dtype=np.uint8)
        cv2.fillPoly(mask, pts=[np.array(self.coords)], color=(1, 1, 1))
        if square:
            bigger_side = np.max(bound_box)
            newimg = np.zeros([bigger_side, bigger_side, 3], dtype=np.uint8)
        else:
            newimg = np.zeros([bound_box[0], bound_box[1], 3], dtype=np.uint8)
        for i in range(bound_box[0]):
            for j in range(bound_box[1]):
                newimg[i][j] = mask[i + offset[1]][j + offset[0]] * _img[i + offset[1]][j + offset[0]]
        return newimg


def traverse_and_segment(root_dir):
    count = 0
    routes_xml = []
    for dirName, subdirList, fileList in os.walk(root_dir):
        for fname in fileList:
            if str(fname).find(".xml") != -1:
                routes_xml.append(dirName + '/' + fname)

    parkings = []
    for route_xml in routes_xml:
        parkings.append(Parking(route_xml, count))
        count += 1
    print(f"Finished loading parkings: {len(parkings)}")
    for parking in parkings:
        ruta_img = str(parking.save_name)[:len(str(parking.save_name)) - 4] + ".jpg"
        img = cv2.imread(ruta_img, 1)
        if img is None:
            break
        parking.extract_patches(img, parking.plazas, savename=parking.id, folder="plazas/")


if __name__ == "__main__":
    # traverse_and_segment('./PKLot/PKLot')
    # train.start()
    p1 = Parking("PKLot/PKLot/UFPR05/Sunny/2013-03-12/2013-03-12_07_30_01.xml", image="PKLot/PKLot/UFPR05/Sunny/2013-03-12/2013-03-12_07_30_01.jpg")
    p1.update_state_from_photo("PKLot/PKLot/UFPR05/Sunny/2013-03-06/2013-03-06_07_45_02.jpg")
    p1.draw_boxes()
    p1.print_overview()
    cv2.waitKey(0)
