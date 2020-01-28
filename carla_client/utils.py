import numpy as np
import carla

def create_camera(
        actor,
        size, fov,
        transform,
        sensor_type='sensor.camera.rgb'):

    bp = actor.get_world().get_blueprint_library().find(sensor_type)
    bp.set_attribute('image_size_x', str(size[0]))
    bp.set_attribute('image_size_y', str(size[1]))
    bp.set_attribute('fov', str(fov))
    # self.bp.set_attribute('sensor_tick', str(period))

    transform = transform

    camera = actor.get_world().spawn_actor(
        bp,
        transform,
        attach_to=actor,
        attachment_type=carla.AttachmentType.Rigid)

    return camera


def cv_from_carla_image(image):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (image.height, image.width, 4))
    img = img[:, :, :3]

    return img


def cv_from_depth_image(image):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (image.height, image.width, 4)).astype(np.float32)

    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    normalized = (R + G * 256 + B * 256 ** 2) / (256**3 - 1)
    in_meters = 1000 * normalized

    return in_meters


def cv_from_semantic_image(image):
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (image.height, image.width, 4))
    img = img[:, :, 2]

    return img
