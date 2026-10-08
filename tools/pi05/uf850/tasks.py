"""Matched scenes: the instruction, rather than a fixed slot, selects the target."""
import numpy as np

OBJECTS = ('black_bowl', 'bottle', 'distractor_block')
PROMPTS = {
    'black_bowl': ('pick up the black bowl and place it on the plate',
                   'put the black bowl on the plate', 'move the black bowl onto the plate'),
    'bottle': ('pick up the green bottle and place it upright on the plate',
               'put the green bottle upright on the plate', 'move the green bottle onto the plate'),
    'distractor_block': ('pick up the red block and place it on the plate',
                         'put the red block on the plate', 'move the red block onto the plate'),
}
ORDERS = {'train': ((0,1,2), (0,2,1), (1,0,2), (2,1,0)),
          'heldout': ((1,2,0), (2,0,1))}


def layout_spec(seed, split='train'):
    if split not in ORDERS:
        raise ValueError('layout split must be train or heldout')
    rng = np.random.default_rng(seed)
    slots = np.array([[.40,.015], [.59,.20], [.59,-.25]])
    slots += rng.uniform([-.025,-.035], [.025,.035], size=(3,2))
    order = ORDERS[split][seed % len(ORDERS[split])]
    positions = {name: slots[order[i]].tolist() for i,name in enumerate(OBJECTS)}
    positions['ramekin'] = [.70,.31]
    positions['target_plate'] = (np.array([.43,-.17])+rng.uniform(-.012,.012,2)).tolist()
    return dict(layout_seed=int(seed), layout_split=split, slot_order=list(order), positions=positions)


def apply_layout(world, spec):
    import pybullet as p
    for obj in world.objects.values():
        if obj['name'] in spec['positions']:
            xyz,quat=p.getBasePositionAndOrientation(obj['body'],physicsClientId=world.client)
            xy=spec['positions'][obj['name']]
            p.resetBasePositionAndOrientation(obj['body'],(*xy,xyz[2]),quat,physicsClientId=world.client)
            p.resetBaseVelocity(obj['body'],[0,0,0],[0,0,0],physicsClientId=world.client)
    for _ in range(30): world.step()
