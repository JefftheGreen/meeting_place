# /usr/bin/env python
# -*- coding: utf-8 -*-

import numbers
import sys

import cma
import googlemaps
import numpy as np

import new_math as math
from utilities.variables import setter

EARTH_RADIUS = 6371
KEY = 'AIzaSyD5WVuhfdPzGVvq2VXCSAxDw4EYL_7pjsA'

class SearchAgent:

    EARTH_RADIUS = 6371

    def __init__(self, key, *locations, **weighted_locations):
        self.last_fitness = 0
        self.fitness_queries = 0
        self.granularity = 1
        self.client = googlemaps.Client(key)
        self.need_to_calc = {}
        self.changed = True
        self.geocode_fitness = {}
        self.coordinate_geocodes = {}
        self._points = []
        self._direction_matrix = None
        self._geom_median = None
        self._drive_median = None
        for l in locations:
            self.add_location(l)
        for l in weighted_locations:
            self.add_location(l, weighted_locations[l])

    @setter
    def changed(self, value):
        if value:
            self.need_to_calc={var: True for var in ['geom_median',
                                                     'drive_median',
                                                     'direction_matrix']}

    @property
    def geom_median(self):
        if self.need_to_calc['geom_median']:
            self.get_geom_median()
        return self._geom_median

    @property
    def geom_median_total_distance(self):
        dist = 0
        for l in self.locations:
            args = l + self.geom_median + (EARTH_RADIUS,) + (True,)
            dist += math.great_circle_dist(*args)
        return dist

    @property
    def geom_median_weihted_distance(self):
        dist = 0
        for l in range(len(self.locations)):
            args = self.locations[l] + self.geom_median + \
                   (EARTH_RADIUS,) + (True,)
            dist += math.great_circle_dist(*args) * self.weights[l]
        return dist


    @property
    def drive_median(self):
        if self.need_to_calc['drive_median']:
            self.get_drive_median()
        return self._drive_median

    @property
    def locations(self):
        locations = []
        for l in self._points:
            locations.append(l['location'])
        return locations

    @property
    def weights(self):
        weights = []
        for l in self._points:
            weights.append(l['weight'])
        return weights

    @property
    def direction_matrix(self):
        if self.need_to_calc['direction_matrix']:
            self.get_direction_matrix()
        return self._direction_matrix

    def add_location(self, location, weight=1, meta=None):
        meta = {} if meta is None else meta
        coords = self.get_coords(location)
        self._points.append({'location': coords,
                             'weight': weight,
                             'meta': meta})
        self.changed = True
        return len(self._points) - 1

    def remove_location(self, index):
        self._points.pop(index)
        self.changed = True

    def get_geom_median(self):
        self._geom_median = (math.spherical_weiszfeld(*self.locations, radius=1)
                             if len(self.locations) > 0 else None)

    def get_direction_matrix(self):
        matrix = []
        for l1 in self.locations:
            directions = []
            for l2 in self.locations:
                if l1 == l2:
                    directions.append(None)
                else:
                    routes = self.client.directions(l1, l2)
                    route = routes[min(len(routes), 0)]['overview_polyline']
                    route = googlemaps.convert.decode_polyline(route['points'])
                    route = [(i['lat'], i['lng']) for i in route]
                    directions.append(route)
            matrix.append(directions)
        self._direction_matrix = matrix

    def overlap(self, source1, source2, sink):
        source1= self.locations[source1]
        source2 = self.locations[source2]
        sink = self.locations[sink]
        from pprint import pprint as print
        try:
            poly1 = self.client.directions(source1,
                                           sink)[0]['overview_polyline']

            poly2 = self.client.directions(source2,
                                           sink)[0]['overview_polyline']
        except IndexError:
            return []
        poly1 = googlemaps.convert.decode_polyline(poly1['points'])
        print(poly1)
        poly2 = googlemaps.convert.decode_polyline(poly2['points'])
        overlap = [(i['lat'], i['lng']) for i in poly1 if i in poly2]
        return overlap

    def overlap_distance(self, source1, source2, sink):
        overlap = self.overlap(source1, source2, sink)
        if len(overlap) < 2:
            return 0
        def wrap(i, overlap):
            return math.great_circle_dist(overlap[i], overlap[i+1])
        return math.summation(0, len(overlap) - 1, wrap, overlap)

    def sigma0(self):
        max_dist = 0
        for i in self.locations:
            i = np.array(i)
            for j in self.locations:
                j = np.array(j)
                dist = sum((i-j)**2)**0.5
                if dist > max_dist:
                    max_dist = dist
        print(max_dist)
        return 1 / 30 * max_dist

    def get_drive_median(self):
        if len(self.locations) == 0:
            self._drive_median = None
        elif len(self.locations) == 1:
            self._drive_median = self.locations[0]
        elif len(self.locations) == 2:
            self._drive_median = self.path_midpoint(*self.locations)
        else:
            self._drive_median = cma.fmin(self.solution_fitness, list(self.geom_median), self.sigma0(), args=(True,), options={'popsize':2, 'tolfun':1000, 'tolfunhist': 1000})

    def get_coords(self, location):
        if isinstance(location, tuple):
            if len(location) == 2 and all([isinstance(i, numbers.Real)
                                           for i in location]):
                return location
        search = self.client.geocode(location)
        if len(search) > 0:
            return (search[0]['geometry']['location']['lat'],
                    search[0]['geometry']['location']['lng'])
        else:
            return

    def solution_fitness(self, solution, weighted=False):
        solution = [int(s * self.granularity) / self.granularity
                    for s in solution]
        solution = tuple(solution)
        print(solution)
        if solution in self.coordinate_geocodes:
            print('solution {0} in coordinates'.format(solution))
            geocode = self.coordinate_geocodes[solution]
        else:
            address = self.client.reverse_geocode(solution)
            if len(address) > 0:
                geocode = address[0]['formatted_address']
            else:
                print('invalid solution {0}'.format(solution))
                return np.NaN
        print('geocode:', geocode)
        if geocode in self.geocode_fitness:
            print('geocode {0} already found'.format(geocode))
            return self.geocode_fitness[geocode]
        if self.fitness_queries > 1000:
            sys.exit()
        matrix = self.client.distance_matrix(self.locations, [solution])
        distances = []
        for r in matrix['rows']:
            distances.append(r['elements'][0]['distance']['value'])
        fitness = (sum(np.array(distances) * np.array(self.weights))
                   if weighted else sum(distances))
        print('fitness:', fitness)
        print('fitness difference:', fitness - self.last_fitness)
        print('number of queries:', self.fitness_queries)
        self.fitness_queries += 1
        self.last_fitness = fitness
        self.geocode_fitness[geocode] = fitness
        return fitness

    def path_midpoint(self, start, end, route_index=0):
        start, end = self.locations[start], self.locations[end]
        if start == end:
            return self.locations[start]
        routes = self.client.directions(start, end)
        route = routes[min(len(routes), route_index)]['legs'][0]
        steps = route['steps']
        print(route.keys())
        route_length = route['distance']['value']
        distance_traveled = 0
        steps_taken = [{'distance':{'value': 0}}]
        while distance_traveled + steps_taken[-1]['distance']['value'] \
                < route_length / 2:
            distance_traveled += steps_taken[-1]['distance']['value']
            steps_taken.append(steps.pop())
            print(distance_traveled, route_length, len(steps))
        mid_step = steps_taken[-1]
        polyline = googlemaps.convert.decode_polyline(mid_step['polyline']
                                                      ['points'])
        polyline_points = [(p['lat'], p['lng']) for p in polyline]
        traveled_points = []
        while distance_traveled < route_length / 2:
            traveled_points.append(polyline_points.pop())
            if len(traveled_points) > 1:
                arg = traveled_points[-1] + traveled_points[-2] \
                      + (EARTH_RADIUS,) + (True,)
                distance_traveled += math.great_circle_dist(*arg) * 1000
        args = traveled_points[-1] + traveled_points[-2] + (0.5,) + (True,)
        return math.great_circle_fraction(*args)





def test():
    locations = ['Salem, OR', 'Banks, OR', 'Portland, OR']
    searcher = SearchAgent(KEY, *locations)
    searcher.granularity = 20
    print(searcher.overlap(1,2,0))
    '''print(searcher.geom_median)
    print(searcher.path_midpoint(0, 1))
    print(searcher.drive_median)'''


if __name__ == '__main__':
    test()