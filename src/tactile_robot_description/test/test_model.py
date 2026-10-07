from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
from ament_index_python.packages import get_package_share_directory


def model(**mappings):
    source = Path(__file__).parents[1] / 'urdf' / 'xarm6_ag95.urdf.xacro'
    return ET.fromstring(xacro.process_file(str(source), mappings=mappings).toxml())


def test_connected_model_and_meshes():
    robot = model()
    links = {link.attrib['name'] for link in robot.findall('link')}
    joints = robot.findall('joint')
    children = {joint.find('child').attrib['link'] for joint in joints}
    assert links - children == {'world'}
    assert len(children) == len(joints) == len(links) - 1
    reached = {'world'}
    while True:
        expanded = reached | {j.find('child').attrib['link'] for j in joints
                              if j.find('parent').attrib['link'] in reached}
        if expanded == reached:
            break
        reached = expanded
    assert reached == links
    independent = {j.attrib['name'] for j in joints
                   if j.attrib['type'] != 'fixed' and j.find('mimic') is None}
    assert independent == {f'joint{i}' for i in range(1, 7)} | {
        'ag95_left_outer_knuckle_joint'}
    mimics = [j.find('mimic') for j in joints if j.find('mimic') is not None]
    assert len(mimics) == 5
    assert all(m.attrib['joint'] == 'ag95_left_outer_knuckle_joint' for m in mimics)
    for mesh in robot.findall('.//mesh'):
        uri = mesh.attrib['filename']
        assert uri.startswith('package://')
        package, relative = uri[len('package://'):].split('/', 1)
        assert (Path(get_package_share_directory(package)) / relative).is_file()


def test_adjustable_mount():
    robot = model(mount_xyz='0 0 0.02', mount_rpy='0 0 1.5708')
    mount = robot.find("joint[@name='ag95_gripper_base_joint']")
    assert mount.find('parent').attrib['link'] == 'link_eef'
    assert mount.find('origin').attrib == {'xyz': '0 0 0.02', 'rpy': '0 0 1.5708'}
