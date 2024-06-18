import os
import pyvista as pv
import xarray as xr
import numpy as np
import rasterio as rio
import utm
import math
import pandas as pd


class Dem:
    def __init__(self, map_path: str, logger=None):
        self.path = map_path
        self.map = rio.open(map_path)
        self.height_array = self.map.read(1).astype('float64')
        self.utm_x = np.linspace(self.map.bounds.left, self.map.bounds.right,
                                 int((self.map.bounds.right - self.map.bounds.left) / self.map.res[0]))
        self.utm_y = np.linspace(self.map.bounds.bottom, self.map.bounds.top,
                                 int((self.map.bounds.top - self.map.bounds.bottom) / self.map.res[1]))
        self.logger = logger
        if self.logger is not None:
            self.logger.log_info(f'Connected to DEM.')

    def tif_to_mesh(self,):
        data = xr.open_rasterio(self.path)
        values = np.asarray(data)
        nans = values == data.nodatavals
        if np.any(nans):
            values[nans] = np.nan
        xx, yy = np.meshgrid(data['x'], data['y'])
        zz = values.reshape(xx.shape)
        mesh = pv.StructuredGrid(xx, yy, zz)
        mesh['data'] = values.ravel(order='F')
        return mesh

    @staticmethod
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def check_bound(self, utm):
        if (self.map.bounds.left < utm[0] < self.map.bounds.right) and \
                (self.map.bounds.bottom < utm[1] < self.map.bounds.top):
            return True
        return False

    def calculate_height(self, lat, long):
        utm_ = utm.from_latlon(lat, long)
        if self.check_bound(utm_):
            _, x_ind = self.find_nearest(self.utm_x, utm_[0])
            _, y_ind = self.find_nearest(self.utm_y, utm_[1])
            return self.height_array[-y_ind - 1][x_ind]
        else:
            if self.logger is not None:
                self.logger.log_error(f'Geo Coordinate {lat, long}, is out of boundary. Returning -1')
            return -1

    def plot(self, plot_2D: bool = False):
        cpos = None
        if plot_2D:
            cpos = 'xy'
        topo = self.tif_to_mesh()
        # terrain = topo.warp_by_scalar()
        plotter = pv.Plotter()
        plotter.add_mesh(topo)
        plotter.show(cpos=cpos)


class Cloud:
    def __init__(self, path):
        self.path = path
        self.cloud = pv.read(path)

    def plot(self,):
        plotter = pv.Plotter()
        plotter.add_mesh(self.cloud)
        plotter.show()


class Compare:
    def __init__(self, dem: Dem, cloud: Cloud):
        self.dem = dem
        self.cloud = cloud
        self.check_bound()

    def check_bound(self,):
        if (self.dem.map.bounds.left < self.cloud.cloud.bounds[0] < self.dem.map.bounds.right) and \
                (self.dem.map.bounds.left < self.cloud.cloud.bounds[1] < self.dem.map.bounds.right) and \
                (self.dem.map.bounds.bottom < self.cloud.cloud.bounds[2] < self.dem.map.bounds.top) and \
                (self.dem.map.bounds.bottom < self.cloud.cloud.bounds[3] < self.dem.map.bounds.top):
            pass
        else:
            raise ValueError('The Point Cloud is not in the area of this DEM map')

    def calculate_error(self, print_log: bool = False):
        dem_height = []
        cloud_height = []
        for i in range(len(self.cloud.cloud.points)-2):  # Two last points are cameras position
            x_val, x_ind = self.dem.find_nearest(self.dem.utm_x, self.cloud.cloud.points[i][0])
            y_val, y_ind = self.dem.find_nearest(self.dem.utm_y, self.cloud.cloud.points[i][1])
            dem_height.append(self.dem.height_array[-y_ind-1][x_ind])
            cloud_height.append(self.cloud.cloud.points[i][2])
            if print_log:
                print(f'UTM coordinates => dem: ({x_val}, {y_val}), cloud: {self.cloud.cloud.points[i][0:2]}')
                print(f'Height => dem height: {self.dem.height_array[-y_ind-1][x_ind]}, cloud height: {self.cloud.cloud.points[i][2]}')
        errors = np.subtract(dem_height, cloud_height)
        print('RMSE: ', np.sqrt(np.square(errors).mean()))
        print('Mean: ', np.mean(errors), ', Std: ', np.std(errors))

    def plot(self, save_cloud_area: bool = False):
        topo = self.dem.tif_to_mesh()
        x_dist = self.cloud.cloud.bounds[1] - self.cloud.cloud.bounds[0]
        y_dist = self.cloud.cloud.bounds[3] - self.cloud.cloud.bounds[2]
        border_size = 1
        bounds = [self.cloud.cloud.bounds[0] - (x_dist*border_size), self.cloud.cloud.bounds[1] + (x_dist*border_size),
                  self.cloud.cloud.bounds[2] - (y_dist*border_size), self.cloud.cloud.bounds[3] + (y_dist*border_size), 0, 5000]
        clipped = topo.clip_box(bounds, invert=False, progress_bar=True)
        if save_cloud_area:
            point_cloud = pv.PolyData(clipped.points)
            point_cloud.save('clipped.ply')
        plotter = pv.Plotter()
        plotter.add_mesh(clipped)
        plotter.add_mesh(self.cloud.cloud)
        plotter.show()

    @staticmethod
    def rotate_matrix(x, y, angle, x_shift=0, y_shift=0, units="DEGREES"):
        x = x - x_shift
        y = y - y_shift
        if units == "DEGREES":
            angle = math.radians(angle)
        xr = (x * math.cos(angle)) - (y * math.sin(angle)) + x_shift
        yr = (x * math.sin(angle)) + (y * math.cos(angle)) + y_shift
        return (xr, yr)

    def calculate_bound_points(self, sensor_data, fovx, fovy):
        utm_first = utm.from_latlon(sensor_data['lat'], sensor_data['long'])

        w = int(2 * sensor_data['altimeter'] * np.tan(np.radians(fovx / 2)))
        h = int(2 * sensor_data['altimeter'] * np.tan(np.radians(fovy / 2)))

        point_1 = (utm_first[0] - (h / 2), utm_first[1] - (w / 2))
        point_2 = (utm_first[0] + (h / 2), utm_first[1] - (w / 2))
        point_3 = (utm_first[0] - (h / 2), utm_first[1] + (w / 2))
        point_4 = (utm_first[0] + (h / 2), utm_first[1] + (w / 2))

        point_1 = self.rotate_matrix(point_1[0] - utm_first[0], point_1[1] - utm_first[1], sensor_data['yaw'])
        point_2 = self.rotate_matrix(point_2[0] - utm_first[0], point_2[1] - utm_first[1], sensor_data['yaw'])
        point_3 = self.rotate_matrix(point_3[0] - utm_first[0], point_3[1] - utm_first[1], sensor_data['yaw'])
        point_4 = self.rotate_matrix(point_4[0] - utm_first[0], point_4[1] - utm_first[1], sensor_data['yaw'])

        point_1 = (point_1[0] + utm_first[0], point_1[1] + utm_first[1])
        point_2 = (point_2[0] + utm_first[0], point_2[1] + utm_first[1])
        point_3 = (point_3[0] + utm_first[0], point_3[1] + utm_first[1])
        point_4 = (point_4[0] + utm_first[0], point_4[1] + utm_first[1])

        _, z1_x = self.dem.find_nearest(self.dem.utm_x, point_1[0])
        _, z1_y = self.dem.find_nearest(self.dem.utm_y, point_1[1])
        z1 = self.dem.height_array[-z1_y - 1][z1_x]

        _, z2_x = self.dem.find_nearest(self.dem.utm_x, point_2[0])
        _, z2_y = self.dem.find_nearest(self.dem.utm_y, point_2[1])
        z2 = self.dem.height_array[-z2_y - 1][z2_x]

        _, z3_x = self.dem.find_nearest(self.dem.utm_x, point_3[0])
        _, z3_y = self.dem.find_nearest(self.dem.utm_y, point_3[1])
        z3 = self.dem.height_array[-z3_y - 1][z3_x]

        _, z4_x = self.dem.find_nearest(self.dem.utm_x, point_4[0])
        _, z4_y = self.dem.find_nearest(self.dem.utm_y, point_4[1])
        z4 = self.dem.height_array[-z4_y - 1][z4_x]

        points = np.array([[point_1[0], point_1[1], z1 + 5], [point_2[0], point_2[1], z2 + 5],
                           [point_3[0], point_3[1], z3 + 5], [point_4[0], point_4[1], z4 + 5]])

        return points

        def plot_bound(self, fovx, fovy, sensor_data=None, sensor_data_2=None, meshroom=False):
        if meshroom is True:
            data_path = os.path.dirname(os.path.dirname(self.cloud.path))
            data = pd.read_csv(os.path.join(data_path, 'sensordata.csv'), header=None)

            sensor_data_first = {'lat': data.iloc[0][1], 'long': data.iloc[0][2], 'altimeter': data.iloc[0][3], 'yaw': data.iloc[0][7]}
            sensor_data_2 = {'lat': data.iloc[1][1], 'long': data.iloc[1][2], 'altimeter': data.iloc[1][3], 'yaw': data.iloc[1][7]}
            sensor_data_last = {'lat': data.iloc[-1][1], 'long': data.iloc[-1][2], 'altimeter': data.iloc[-1][3], 'yaw': data.iloc[-1][7]}

            points_first = self.calculate_bound_points(sensor_data_first, fovx, fovy)
            points_2 = self.calculate_bound_points(sensor_data_2, fovx, fovy)
            points_last = self.calculate_bound_points(sensor_data_last, fovx, fovy)

            topo = self.dem.tif_to_mesh()
            x_dist = self.cloud.cloud.bounds[1] - self.cloud.cloud.bounds[0]
            y_dist = self.cloud.cloud.bounds[3] - self.cloud.cloud.bounds[2]
            border_size = 1
            bounds = [self.cloud.cloud.bounds[0] - (x_dist*border_size), self.cloud.cloud.bounds[1] + (x_dist*border_size),
                      self.cloud.cloud.bounds[2] - (y_dist*border_size), self.cloud.cloud.bounds[3] + (y_dist*border_size), 0, 5000]
            clipped = topo.clip_box(bounds, invert=False, progress_bar=True)
            plotter = pv.Plotter()
            plotter.add_mesh(clipped)
            plotter.add_mesh(self.cloud.cloud)
            plotter.add_points(points_first, label='image_1')
            plotter.add_points(points_2, color='yellow', label='image_2')
            plotter.add_points(points_last, color='red', label='image_3')
            plotter.add_legend(bcolor='w', face=None, size=(0.1, 0.1))
            plotter.show()
        else:
            points_first = self.calculate_bound_points(sensor_data, fovx, fovy)
            if sensor_data_2 is not None:
                points_last = self.calculate_bound_points(sensor_data_2, fovx, fovy)

            topo = self.dem.tif_to_mesh()
            x_dist = self.cloud.cloud.bounds[1] - self.cloud.cloud.bounds[0]
            y_dist = self.cloud.cloud.bounds[3] - self.cloud.cloud.bounds[2]
            border_size = 1
            bounds = [self.cloud.cloud.bounds[0] - (x_dist * border_size),
                      self.cloud.cloud.bounds[1] + (x_dist * border_size),
                      self.cloud.cloud.bounds[2] - (y_dist * border_size),
                      self.cloud.cloud.bounds[3] + (y_dist * border_size), 0, 5000]
            clipped = topo.clip_box(bounds, invert=False, progress_bar=True)
            plotter = pv.Plotter()
            plotter.add_mesh(clipped)
            plotter.add_mesh(self.cloud.cloud)
            plotter.add_points(points_first, label='image_1')
            if sensor_data_2 is not None:
                plotter.add_points(points_last, color='red', label='image_2')
            plotter.add_legend(bcolor='w', face=None, size=(0.1, 0.1))
            plotter.show()



if __name__ == '__main__':
    dem = Dem(r'F:\Data\DEM_map\DemMap_47_part1.tif')
    cloud = Cloud(r'F:\Data\DEM_map\cloud_1.ply')
    compare = Compare(dem, cloud)
    compare.plot_bound(fovx=69, fovy=39)
