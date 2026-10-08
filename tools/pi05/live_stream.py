"""Send actual MuJoCo geometry and camera frames to the ROS Python process."""
import base64
import json
import hashlib
from pathlib import Path
import socket
import struct
import cv2
import numpy as np


class LiveStream:
    def __init__(self, env, folder, port):
        self.socket = socket.create_connection(('127.0.0.1', port), timeout=10)
        self.sim = env.env.sim
        model = self.sim.model
        folder = Path(folder).resolve() / 'rviz_meshes'
        folder.mkdir(parents=True, exist_ok=True)
        self.geometries = []
        for index in range(model.ngeom):
            # Match robosuite's visible group; group 0 is collision geometry.
            if model.geom_group[index] == 0:
                continue
            color = np.asarray(model.geom_rgba[index]).tolist()
            material = int(model.geom_matid[index])
            if material >= 0:
                color = np.asarray(model.mat_rgba[material]).tolist()
            if color[3] < .01:
                continue
            kind, size = int(model.geom_type[index]), model.geom_size[index].tolist()
            mesh = None
            if kind == 7:
                mid = int(model.geom_dataid[index])
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                vertices = np.asarray(model.mesh_vert[va:va+vn])
                faces = np.asarray(model.mesh_face[fa:fa+fn])
                fingerprint = hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()[:16]
                path = folder / f'mesh_{fingerprint}.stl'
                if not path.exists():
                    triangles = vertices[faces]
                    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
                    records = np.zeros(fn, dtype=[('normal', '<f4', (3,)),
                                                   ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
                    records['normal'], records['vertices'] = normal, triangles
                    path.write_bytes(b'LIBERO live mesh'.ljust(80, b'\0') + struct.pack('<I', fn) + records.tobytes())
                mesh = path.as_uri()
            if kind in (0, 2, 3, 4, 5, 6, 7):
                self.geometries.append(dict(id=index, kind=kind, size=size, color=color, mesh=mesh))

    def publish(self, image, wrist, prompt, step, status):
        data = self.sim.data
        geoms = []
        for geom in self.geometries:
            index = geom['id']
            geoms.append(dict(geom, xyz=data.geom_xpos[index].tolist(),
                              rotation=data.geom_xmat[index].reshape(3, 3).tolist()))
        def encode(rgb):
            ok, encoded = cv2.imencode('.jpg', rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError('Camera encode failed')
            return base64.b64encode(encoded).decode('ascii')
        payload = dict(geometries=geoms, image=encode(image), wrist=encode(wrist),
                       prompt=prompt, step=step, status=status)
        self.socket.sendall(json.dumps(payload).encode() + b'\n')

    def close(self):
        self.socket.close()
