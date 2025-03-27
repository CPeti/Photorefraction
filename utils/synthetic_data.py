import numpy as np
from skspatial.objects import Line
from cv2 import GaussianBlur

l_eye = 1/60  # distance from lens to retina

def length(v):
    return np.linalg.norm(v)

def normalize(v):
    return v / length(v)

class Ray():
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)
        self.dir = self.end - self.start
        self.dir = normalize(self.dir)

    def intersect(self, ray: 'Ray'):
        line_a = Line(point=self.start, direction=self.dir)
        line_b = Line(point=ray.start, direction=ray.dir)
        # occasionally fails to find intersection even when it exists - TODO fix this
        intersection = line_a.intersect_line(line_b)
        return intersection
    
class Plane():
    def __init__(self, point, normal):
        self.normal = normalize(np.array(normal))
        self.point = point

    @classmethod
    def init_from_points(cls, point1, point2, point3):
        v1 = point2 - point1
        v2 = point3 - point1
        normal = np.cross(v1, v2)
        point = point1
        return cls(point, normal)

    def intersect(self, ray: Ray):
        # check if the ray is parallel to the plane
        if np.dot(ray.dir, self.normal) == 0:
            return None
        # calculate intersection point
        t = np.dot(self.point - ray.start, self.normal) / np.dot(ray.dir, self.normal)
        return ray.start + t * ray.dir
    
class Sphere():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray, mode='first'):
        # calculate intersection point
        a = np.dot(ray.dir, ray.dir)
        b = 2 * np.dot(ray.dir, ray.start - self.center)
        c = np.dot(ray.start - self.center, ray.start - self.center) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        if mode == 'first':
            return ray.start + min(t1, t2) * ray.dir
        elif mode == 'second':
            return ray.start + max(t1, t2) * ray.dir
        else:
            return None
        
class Lens():
    """
    Thin lens model - only accurate for small angles
    """
    def __init__(self, center, normal, R, D_cyl, axis, D):
        self.center = np.array(center)
        self.normal = normalize(np.array(normal))
        self.f_n = 1/(1/l_eye + 1/D)

        self.R = R          # spherical refractive error
        self.D_cyl = D_cyl  # astigmatic error
        self.axis = axis    # axis of astigmatism

        # an astigmatic lens produces two focal lines instead of a single focal point - called sturm lines
        # once we have calculated the location of the sturm lines,
        # we can use them to refract rays more accurately than with the paraxial approximation of the astigmatic lens
        self.sturm_line1_1 = None
        self.sturm_line1_2 = None
        self.sturm_line2_1 = None
        self.sturm_line2_2 = None

    def set_sturms(self, sturm_line1_1, sturm_line1_2, sturm_line2_1, sturm_line2_2):
        self.sturm_line1_1 = sturm_line1_1
        self.sturm_line1_2 = sturm_line1_2
        self.sturm_line2_1 = sturm_line2_1
        self.sturm_line2_2 = sturm_line2_2

    def intersect(self, ray: Ray):
        # calculate intersection point of ray and lens
        t = np.dot(self.center - ray.start, self.normal) / np.dot(ray.dir, self.normal)
        intersection = ray.start + t * ray.dir
        return intersection
    
    def refract(self, ray: Ray):
        if self.sturm_line1_1 is not None and self.D_cyl != 0: # we assume all sturms are set
            return self.refract_with_sturms(ray)

        hit = self.intersect(ray)
        # calculate refracted ray
        hit_dir = hit - self.center
        axis_vector = np.array([np.cos(self.axis), np.sin(self.axis), 0])
        axis_vector = normalize(axis_vector)
        angle = np.arccos(np.dot(hit_dir, axis_vector) / (length(hit_dir) * length(axis_vector)))

        # Powers in Oblique Meridians Formula
        # D_sum = R + sin(angle)^2 * D_cyl
        focal_length = 1/(1/self.f_n - (self.R + pow(np.sin(angle), 2) * self.D_cyl))
        focal_point = self.center - self.normal * focal_length

        center_ray = Ray(ray.start, self.center)
        orthogonal_ray = Ray(ray.start, ray.start-self.normal)
        if not np.array_equal(orthogonal_ray.dir, center_ray.dir):
            ortho_hit = self.intersect(orthogonal_ray)
            ortho_refract = Ray(ortho_hit, focal_point)
            image_point = center_ray.intersect(ortho_refract)
        else:
            im_distance = 1/(1/focal_length - 1/np.linalg.norm(self.center - orthogonal_ray.start))
            image_point = self.center - self.normal * im_distance
        refracted_ray = Ray(hit, image_point)
        return refracted_ray

    def refract_with_sturms(self, ray):
        # the ray must pass through both sturm lines
        # the hit point and the two lines define the ray that will be refracted
        hit = self.intersect(ray)
        sturm_plane = Plane.init_from_points(hit, self.sturm_line1_1, self.sturm_line1_2)
        sturm_ray = Ray(self.sturm_line2_1, self.sturm_line2_2)
        sturm_intersection = sturm_plane.intersect(sturm_ray)
        refracted_ray = Ray(hit, sturm_intersection)
        return refracted_ray
    
class Camera():
    def __init__(self, loc, lookat, up, r_p, windowWidth=100, windowHeight=100):
        """
        loc: camera location
        lookat: point the camera is looking at
        up: up vector
        r_p: pupil radius
        windowWidth: width of the window
        windowHeight: height of the window
        """
        self.loc = loc
        self.lookat = lookat
        self.up = up
        w = lookat - loc
        self.focus = length(w)
        self.right = normalize(np.cross(self.up, w)) * r_p*2
        self.up = normalize(np.cross(w, self.right)) * r_p*2
    
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        
    def getRay(self, X, Y):
        direction = self.lookat + self.right * (2.0 *(X + 0.5) / self.windowWidth - 1) + self.up * (2.0 *(Y + 0.5) / self.windowHeight - 1) - self.loc
        return Ray(self.loc, direction)
        
class Scene():
    def __init__(self, camera_pos, light_pos, lens_pos, retina_pos, R, D_cyl, axis, D, r_p, w_width=100, w_height=100):
        self.lens_object = Lens(lens_pos, np.array((0, 0, 1)), R, D_cyl, axis, D)
        sphere_center = retina_pos + np.array([0, 0, -l_eye/2])
        self.retina_object = Sphere(sphere_center, l_eye/2)

        self.camera = Camera(camera_pos, np.array((0, 0, 0)), np.array((0, 1, 0)), r_p, windowWidth=w_width, windowHeight=w_height)
        self.r_p = r_p
        self.axis = axis
        self.camera_pos = camera_pos
        self.light_pos = light_pos
        self.lens_pos = lens_pos
        self.w_width = w_width
        self.w_height = w_height

        # X and Y coordinates of lens edges
        self.a = np.cos(self.axis) * self.r_p
        self.b = np.sin(self.axis) * self.r_p
        self.c = np.cos(self.axis+np.pi/2) * self.r_p
        self.d = np.sin(self.axis+np.pi/2) * self.r_p

        self.cast_camera_rays()
        self.cast_light_rays()
        self.lens_object.set_sturms(self.sturm_line_cam_1_1, self.sturm_line_cam_1_2, self.sturm_line_cam_2_1, self.sturm_line_cam_2_2)



    def cast_camera_rays(self):
        # cast rays from camera to 4 corners of the lens
        #  - these 4 rays are not at oblique angles, and can be refracted using the thin lens model without approximation
        camera_rays = []
        camera_rays.append(Ray(self.camera_pos, self.lens_pos + np.array((self.a,  self.b,  0))))
        camera_rays.append(Ray(self.camera_pos, self.lens_pos + np.array((-self.a, -self.b, 0)))) # top of crescent
        camera_rays.append(Ray(self.camera_pos, self.lens_pos + np.array((self.c,  self.d,  0))))
        camera_rays.append(Ray(self.camera_pos, self.lens_pos + np.array((-self.c, -self.d, 0))))
        camera_hits = []

        refracted_camera_rays = []
        for ray in camera_rays:
            ray = self.lens_object.refract(ray)
            refracted_camera_rays.append(ray)
            hit = self.retina_object.intersect(ray)
            camera_hits.append(hit)
        camera_ray_focus1 = refracted_camera_rays[0].intersect(refracted_camera_rays[1])
        camera_ray_focus2 = refracted_camera_rays[2].intersect(refracted_camera_rays[3])

        camera_hits = np.array(camera_hits)
        sturm_plane_cam1 = Plane(camera_ray_focus1, normalize(self.camera_pos))
        sturm_plane_cam2 = Plane(camera_ray_focus2, normalize(self.camera_pos))
        self.sturm_line_cam_1_1 = sturm_plane_cam1.intersect(refracted_camera_rays[2])
        self.sturm_line_cam_1_2 = sturm_plane_cam1.intersect(refracted_camera_rays[3])
        self.sturm_line_cam_2_1 = sturm_plane_cam2.intersect(refracted_camera_rays[0])
        self.sturm_line_cam_2_2 = sturm_plane_cam2.intersect(refracted_camera_rays[1])

        self.image_point_cam = np.mean([camera_ray_focus1, camera_ray_focus2], axis=0)

        self.camera_hits = camera_hits

    def cast_light_rays(self):
        # cast rays from light source to 4 corners of the lens
        light_rays = []
        light_rays.append(Ray(self.light_pos, self.lens_pos + np.array((self.a,  self.b,  0))))
        light_rays.append(Ray(self.light_pos, self.lens_pos + np.array((-self.a, -self.b, 0)))) # top of crescent
        light_rays.append(Ray(self.light_pos, self.lens_pos + np.array((self.c,  self.d,  0))))
        light_rays.append(Ray(self.light_pos, self.lens_pos + np.array((-self.c, -self.d, 0))))
        light_hits = []

        refracted_light_rays = []
        for ray in light_rays:
            ray = self.lens_object.refract(ray)
            refracted_light_rays.append(ray)
            hit = self.retina_object.intersect(ray)
            light_hits.append(hit)
        light_ray_focus1 = refracted_light_rays[0].intersect(refracted_light_rays[1])
        light_ray_focus2 = refracted_light_rays[2].intersect(refracted_light_rays[3])
        light_hits = np.array(light_hits)

        sturm_plane1 = Plane(light_ray_focus1, normalize(self.camera_pos))
        sturm_plane2 = Plane(light_ray_focus2, normalize(self.camera_pos))
        self.sturm_line1_1 = sturm_plane1.intersect(refracted_light_rays[2])
        self.sturm_line1_2 = sturm_plane1.intersect(refracted_light_rays[3])
        self.sturm_line2_1 = sturm_plane2.intersect(refracted_light_rays[0])
        self.sturm_line2_2 = sturm_plane2.intersect(refracted_light_rays[1])

        self.image_point = np.mean([light_ray_focus1, light_ray_focus2], axis=0)

        self.light_hits = light_hits

    def trace(self, ray: Ray, fundus_color=(1, 0, 0), crescent_color=(1, 1, 0.5), shade=False):
        fundus_color = np.array(fundus_color)
        crescent_color = np.array(crescent_color)
        lens_hit = self.lens_object.intersect(ray)
            
        if length(lens_hit - self.lens_object.center) > self.r_p: # ray misses lens
            return (0, 0, 0)

        camera_ray = self.lens_object.refract(ray)
        retina_hit = self.retina_object.intersect(camera_ray)

        # calculate light ray that hits retina at the same point as camera ray
        if self.lens_object.D_cyl != 0:
            light_plane = Plane.init_from_points(self.sturm_line1_1, self.sturm_line1_2, retina_hit)
            sturm_line_2 = Ray(self.sturm_line2_1, self.sturm_line2_2)
            sturm_line_2_hit = light_plane.intersect(sturm_line_2)
            light_ray = Ray(sturm_line_2_hit, retina_hit)
        else:
            light_ray = Ray(self.image_point, retina_hit)

        entry_point = self.lens_object.intersect(light_ray)
        distance_to_edge = length(entry_point - self.lens_object.center)

        if distance_to_edge > self.r_p: # ray misses lens - point is not illuminated
            if shade:
                return fundus_color * (self.r_p / distance_to_edge) ** shade
            else:
                return fundus_color        
        return crescent_color
    
    def render(self, fundus_color=(1, 0, 0), crescent_color=(1, 1, 0.5), glow=True, blur=True, shade=5):
        # material properties
        image = np.zeros((self.w_height, self.w_width, 3), dtype=np.uint8)

        for x in range(self.w_width):
            for y in range(self.w_height):
                ray = self.camera.getRay(x, y)
                outRadience = self.trace(ray, crescent_color=crescent_color, fundus_color=fundus_color, shade=shade)
                outRadience = np.clip(outRadience, 0, 1) * 255
                image[y, x] = outRadience.astype(np.uint8)
        if glow:
            image = self.add_glow(image)
        if blur:
            image = self.add_blur(image)
        return image
        
    def add_glow(self, image): # TODO make this more realistic
        # add white glow to center of eye
        center = (int(self.w_width/2), int(self.w_height/2))
        reflection_radius = self.w_width/16 - 1
        for x in range(self.w_width):
            for y in range(self.w_height):
                d = length(np.array([x, y]) - np.array(center))
                if d < reflection_radius:
                    intensity = 1 - (d / reflection_radius)**30
                    c = np.array([255, 255, 255])*intensity
                    c = np.clip(c, 0, 255)
                    image[y, x] = c
        return image
    
    def add_blur(self, image):
        # noise the whole image with a gaussian filter
        noised = GaussianBlur(image, (3, 3), 1)
        # add some noise
        center = (int(self.w_width/2), int(self.w_height/2))
        for x in range(self.w_width):
            for y in range(self.w_height):
                # TODO make this more realistic
                noised[y, x] = np.clip(noised[y, x] + np.random.normal(0.5, 0.5, 3)*0.05 * 255, 0, 255) 
                if length(np.array([x, y]) - np.array(center)) > self.w_width/2:
                    noised[y, x] = (0, 0, 0)

        return noised
    

def generate_crescent(R, D_cyl, axis, e, D, r_p,
                    orientation='V',
                    fundus_color=(1, 0, 0), 
                    crescent_color=(1, 1, 0.5), 
                    shade=0.8, blur=True, glow=True, 
                    size=100):
    """
    Generate a synthetic image of a crescent
    Parameters:
    R: spherical refractive error
    D_cyl: astigmatic error
    axis: axis of astigmatism
    e: distance from light source to camera
    D: distance from camera to lens
    r_p: pupil radius
    fundus_color: color of the fundus: 0-1 RGB tuple
    crescent_color: color of the crescent: 0-1 RGB tuple
    shade: shading factor of fundus pixels based on distance to edge of crescent. Larger values makes the intensity of the non-illuminated pixels decrease faster.
    blur: apply some noise and a gaussian blur to the image
    glow: add a white glow to the center of the image
    size: width and height of the image
    """
    eps = 0
    camera_pos = (0, 0, D)
    if orientation == 'V':
        light_pos = (0, -e, D)
    elif orientation == 'H':
        light_pos = (e, 0, D)
    else:
        raise ValueError('Orientation must be V or H')
    lens_pos = (0, 0, 0)
    retina_pos = (0, 0, -l_eye)
    success = False
    while not success:
        try:
            axis_rad = np.deg2rad(axis-90 + eps) # convert axis to radians measured from x-axis
            scene = Scene(camera_pos, light_pos, lens_pos, retina_pos, R, D_cyl, axis_rad, D, r_p, w_width=size, w_height=size)
            success = True
        except ValueError:
            eps += 1e-6
    return scene.render(fundus_color=fundus_color, crescent_color=crescent_color, shade=shade, blur=blur, glow=glow)
    

        



        