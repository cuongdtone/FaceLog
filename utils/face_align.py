import cv2
import numpy as np
import math
import sys



arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def get_similarity_transform(src, dst=arcface_src, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1
    T = np.eye(dim + 1, dtype=np.double)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0
    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T[0:2, :]

def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M = get_similarity_transform(landmark, arcface_src)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = SimilarityTransform(rotation=rot)
    t4 = SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


def estimate_affine_matrix_3d23d(X, Y):
    X_homo = np.hstack((X, np.ones([X.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y)[0].T # Affine matrix. 3 x 4
    return P


def P2sRt(P):
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz


def _center_and_normalize_points(points):
    centroid = np.mean(points, axis=0)
    rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])
    norm_factor = math.sqrt(2) / rms
    matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
                       [0, norm_factor, -norm_factor * centroid[1]],
                       [0, 0, 1]])
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    new_pointsh = (matrix @ pointsh).T
    new_points = new_pointsh[:, :2]
    new_points[:, 0] /= new_pointsh[:, 2]
    new_points[:, 1] /= new_pointsh[:, 2]
    return matrix, new_points


def get_bound_method_class(m):
    return m.im_class if sys.version < '3' else m.__self__.__class__


class GeometricTransform(object):
    def __call__(self, coords):
        raise NotImplementedError()

    def inverse(self, coords):
        raise NotImplementedError()

    def residuals(self, src, dst):
        return np.sqrt(np.sum((self(src) - dst) ** 2, axis=1))

    def __add__(self, other):
        raise NotImplementedError()


class ProjectiveTransform(GeometricTransform):
    _coeffs = range(8)

    def __init__(self, matrix=None):
        if matrix is None:
            # default to an identity transform
            matrix = np.eye(3)
        if matrix.shape != (3, 3):
            raise ValueError("invalid shape of transformation matrix")
        self.params = matrix

    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)
        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = src.T @ matrix.T
        dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
        # rescale to homogeneous coordinates
        dst[:, :2] /= dst[:, 2:3]
        return dst[:, :2]

    def __call__(self, coords):
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        return self._apply_mat(coords, self._inv_matrix)

    def estimate(self, src, dst):
        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = np.nan * np.empty((3, 3))
            return False

        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # params: a0, a1, a2, b0, b1, b2, c0, c1
        A = np.zeros((rows * 2, 9))
        A[:rows, 0] = xs
        A[:rows, 1] = ys
        A[:rows, 2] = 1
        A[:rows, 6] = - xd * xs
        A[:rows, 7] = - xd * ys
        A[rows:, 3] = xs
        A[rows:, 4] = ys
        A[rows:, 5] = 1
        A[rows:, 6] = - yd * xs
        A[rows:, 7] = - yd * ys
        A[:rows, 8] = xd
        A[rows:, 8] = yd

        A = A[:, list(self._coeffs) + [8]]

        _, _, V = np.linalg.svd(A)
        if np.isclose(V[-1, -1], 0):
            return False

        H = np.zeros((3, 3))
        H.flat[list(self._coeffs) + [8]] = - V[-1, :-1] / V[-1, -1]
        H[2, 2] = 1

        H = np.linalg.inv(dst_matrix) @ H @ src_matrix
        self.params = H
        return True

    def __add__(self, other):
        """Combine this transformation with another.

        """
        if isinstance(other, ProjectiveTransform):
            # combination of the same types result in a transformation of this
            # type again, otherwise use general projective transformation
            if type(self) == type(other):
                tform = self.__class__
            else:
                tform = ProjectiveTransform
            return tform(other.params @ self.params)
        elif (hasattr(other, '__name__')
              and other.__name__ == 'inverse'
              and hasattr(get_bound_method_class(other), '_inv_matrix')):
            return ProjectiveTransform(other.__self__._inv_matrix @ self.params)
        else:
            raise TypeError("Cannot combine transformations of differing "
                            "types.")


class SimilarityTransform(ProjectiveTransform):
    def __init__(self, matrix=None, scale=None, rotation=None,
                 translation=None):
        params = any(param is not None
                     for param in (scale, rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix
        elif params:
            if scale is None:
                scale = 1
            if rotation is None:
                rotation = 0
            if translation is None:
                translation = (0, 0)

            self.params = np.array([
                [math.cos(rotation), - math.sin(rotation), 0],
                [math.sin(rotation),   math.cos(rotation), 0],
                [                 0,                    0, 1]
            ])
            self.params[0:2, 0:2] *= scale
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)