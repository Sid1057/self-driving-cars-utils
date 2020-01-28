from autopilot import AutoPilot

from utils import create_camera
import carla

import cv2 as cv
import numpy as np

import queue

import time


class CarSync:
    def __init__(self, world, autopilot):
        self.world = world
        self.autopilot = autopilot

        blueprint_library = self.world.get_blueprint_library()

        self.vehicle = None

        i = 0
        while True:
            try:
                self.vehicle = world.spawn_actor(
                    # blueprint_library.find('vehicle.tesla.model3'),
                    blueprint_library.find('vehicle.ford.mustang'),
                    self.world.get_map().get_spawn_points()[i])
                break
            except RuntimeError:
                i += 1

        self.vehicle.set_simulate_physics(True)

        left_transform = carla.Transform(
            carla.Location(x=2.25, y=-0.15, z=0.9),
            carla.Rotation(pitch=-0))
        right_transform = carla.Transform(
            carla.Location(x=2.25, y=+0.15, z=0.9),
            carla.Rotation(pitch=-0))
        size = (1024, 720)
        fov = 45

        self.sensors = {
            'left': create_camera(
                self.vehicle,
                size, fov, left_transform),
            'right': create_camera(
                self.vehicle,
                size, fov, right_transform),
            'semantic': create_camera(
                self.vehicle,
                size, fov, left_transform,
                'sensor.camera.semantic_segmentation'),
            'depth': create_camera(
                self.vehicle,
                size, fov, left_transform,
                'sensor.camera.depth'),
        }

        self.frame = None
        self.delta_seconds = 1.0
        self._queues = []
        self._settings = None

        bp = self.world.get_blueprint_library().find(
            'sensor.other.collision')
        self.collision_sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self.vehicle)

        self.collision_intensity = 0

        def collision_handler(event):
            impulse = event.normal_impulse
            self.collision_intensity = (impulse.x**2 + impulse.y**2 + impulse.z**2) ** 0.5

        self.collision_sensor.listen(collision_handler)

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors.values():
            make_queue(sensor.listen)

        return self

    def __iter__(self):
        while True:
            timeout = 1.0
            self.frame = self.world.tick()
            data = [self._retrieve_data(q, timeout) for q in self._queues]
            assert all(x.frame == self.frame for x in data[:3])

            control = self.autopilot(data)
            self.vehicle.apply_control(control)

            yield self.collision_intensity

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        print('destroying actors.')
        self.vehicle.destroy()
        for actor in self.sensors.values():
            actor.destroy()

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()

autopilot = AutoPilot()

try:
    for i in range(10):
        with CarSync(world, autopilot) as mustang:
            for collision in mustang:
                if collision > 1:
                    print('collision with intensity {}'.format(collision))
                    break
finally:
    print('done.')
