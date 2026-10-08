from pathlib import Path
import xml.etree.ElementTree as ET

import xacro
import pytest
from ament_index_python.packages import get_package_share_directory


def model(**mappings):
    source = Path(__file__).parents[1] / 'urdf' / 'xarm6_ag95.urdf.xacro'
    return ET.fromstring(xacro.process_file(str(source), mappings=mappings).toxml())


@pytest.mark.parametrize('robot_model', ['uf850', 'xarm6'])
def test_connected_model_and_meshes(robot_model):
    robot = model(robot_model=robot_model)
    assert any(f'/meshes/{robot_model}/' in mesh.attrib['filename']
               for mesh in robot.findall('.//mesh'))
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
    materials = {m.attrib['name'] for m in robot.findall('material')}
    for material in robot.findall('link/visual/material'):
        assert len(material) or material.attrib['name'] in materials
    for mesh in robot.findall('.//mesh'):
        uri = mesh.attrib['filename']
        assert uri.startswith('package://')
        package, relative = uri[len('package://'):].split('/', 1)
        assert (Path(get_package_share_directory(package)) / relative).is_file()


def test_default_is_uf850():
    robot = model()
    assert robot.find("joint[@name='joint1']/origin").attrib['xyz'] == '0 0 0.364'
    assert robot.find("joint[@name='joint3']/limit").attrib['upper'] == '0.061087'
    assert '/uf850/' in robot.find("link[@name='link_base']/visual/geometry/mesh").attrib['filename']


def test_adjustable_mount():
    robot = model(mount_xyz='0 0 0.02', mount_rpy='0 0 1.5708')
    mount = robot.find("joint[@name='ag95_gripper_base_joint']")
    assert mount.find('parent').attrib['link'] == 'link_eef'
    assert mount.find('origin').attrib == {'xyz': '0 0 0.02', 'rpy': '0 0 1.5708'}


def test_camera_kit_and_optical_frames():
    robot = model(camera_mount_xyz='0.01 0 0.02', camera_mount_rpy='0 0 0.3')
    mount = robot.find("joint[@name='camera_mount_joint']")
    assert mount.find('parent').attrib['link'] == 'link_eef'
    assert mount.find('origin').attrib == {'xyz': '0.01 0 0.02', 'rpy': '0 0 0.3'}
    kit = robot.find("joint[@name='camera_kit_joint']")
    assert kit.find('parent').attrib['link'] == 'camera_mount_link'
    assert kit.find('child').attrib['link'] == 'd435_link_eef'
    mesh = robot.find("link[@name='d435_link_eef']/visual/geometry/mesh")
    assert mesh.attrib['filename'].endswith('visual/d435_with_cam_stand.stl')
    for sensor in ('depth', 'color', 'left_ir', 'right_ir'):
        joint = robot.find(f"joint[@name='d435_camera_{sensor}_optical_joint']")
        assert joint.attrib['type'] == 'fixed'
        assert joint.find('child').attrib['link'] == f'd435_camera_{sensor}_optical_frame'
        assert joint.find('origin').attrib['rpy'] == '-1.5707963267948966 0 -1.5707963267948966'
    assert not any('imu' in link.attrib['name'] for link in robot.findall('link'))


def test_camera_can_be_disabled():
    robot = model(add_camera='false')
    assert robot.find("link[@name='camera_mount_link']") is None
    assert not any(link.attrib['name'].startswith('d435_') for link in robot.findall('link'))
    assert robot.find("joint[@name='ag95_gripper_base_joint']") is not None
