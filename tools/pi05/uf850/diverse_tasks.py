"""Reproducible color/shape compositions and larger physical pose variation."""
import math
import numpy as np

TARGETS=('target_0','target_1','target_2')
COLORS={'red':(.85,.07,.06,1.),'green':(.10,.62,.18,1.),'blue':(.07,.25,.88,1.),
        'yellow':(.92,.78,.07,1.),'orange':(.95,.40,.07,1.),'purple':(.62,.13,.78,1.),
        'black':(.06,.07,.08,1.),'white':(.91,.92,.90,1.)}
KINDS=('bowl','bottle','cube','cylinder')
NOUNS={'bowl':'bowl','bottle':'bottle','cube':'block','cylinder':'cylinder'}
HELDOUT_PAIRS={('yellow','bottle'),('purple','cube'),('blue','bowl'),('green','cylinder')}
TEMPLATES=('pick up the {label} and place it on the plate','put the {label} on the plate',
           'move the {label} onto the plate','place the {label} on the plate',
           'transfer the {label} to the plate')
ROI={'reference_size':448,'box':[160,224,432,448],'output_size':224,
     'method':'off-axis projection of the original square external camera, then black letterbox'}
PROFILE={'version':'uf850_diverse_v6','colors':list(COLORS),'kinds':list(KINDS),
         'heldout_color_shape_pairs':[list(pair) for pair in sorted(HELDOUT_PAIRS)],
         'position_jitter_m':[.06,.08],'plate_jitter_m':.035,
         'start_position_jitter_m':[.025,.025,.015],'scale_range':[.90,1.10],
         'external_roi':ROI,'fps':20,'gripper_motor_effort_nm':3.,
         'teacher':'Physical IK demonstrations; no object attachment or manipulation-time resets'}


def label(item):return f"{item['color']} {NOUNS[item['kind']]}"


def color_transfer_spec(seed,variant):
    """Evaluation only: identical physics, red versus unseen cyan target."""
    if variant not in ('known','cyan'):raise ValueError('Unknown color-transfer variant')
    spec=layout_spec(seed,'train')
    index=seed-42001;kind=KINDS[(index//3)%len(KINDS)];slot=index%3
    slots={item['slot']:item['xy'] for item in spec['objects']}
    order=[slot]+[s for s in range(3) if s!=slot]
    colors=[('cyan' if variant=='cyan' else 'red'),'blue','yellow']
    palette={**COLORS,'cyan':(.04,.80,.85,1.)}
    for i,item in enumerate(spec['objects']):
        scale=item['scale']/(1.35 if item['kind']=='bowl' else 1.)
        color=colors[i]
        item.update(kind=kind,scale=scale*(1.35 if kind=='bowl' else 1.),
            color=color,rgba=list(palette[color]),slot=order[i],xy=slots[order[i]],
            novel_color=color=='cyan',novel_combination=(color,kind) in HELDOUT_PAIRS or color=='cyan')
    spec.update(slot_order=order,layout_split='color_transfer_evaluation',
        color_transfer_variant=variant,prompt_variant=0,mode='same_shape',
        note='Only target color/name differs between known and cyan variants; no retraining.')
    return spec


def layout_spec(seed,split='train'):
    if split not in ('train','heldout'):raise ValueError('Unknown split')
    rng=np.random.default_rng(seed)
    all_pairs=[(color,kind) for color in COLORS for kind in KINDS]
    train_pairs=[pair for pair in all_pairs if pair not in HELDOUT_PAIRS]
    allowed=train_pairs if split=='train' else all_pairs
    mode=('same_shape','same_color','mixed','mixed')[seed%4]
    if mode=='same_shape':
        kind=KINDS[int(rng.integers(4))];choices=[pair for pair in allowed if pair[1]==kind]
    elif mode=='same_color':
        color=list(COLORS)[int(rng.integers(8))];choices=[pair for pair in allowed if pair[0]==color]
    else:choices=allowed
    indices=rng.choice(len(choices),3,replace=False)
    pairs=[choices[int(i)] for i in indices]
    if split=='heldout' and not any(pair in HELDOUT_PAIRS for pair in pairs):
        novel=sorted(HELDOUT_PAIRS)[seed%len(HELDOUT_PAIRS)]
        if mode=='same_shape':novel=next(pair for pair in sorted(HELDOUT_PAIRS) if pair[1]==pairs[0][1])
        elif mode=='same_color':
            possible=[pair for pair in sorted(HELDOUT_PAIRS) if pair[0]==pairs[0][0]]
            if possible:novel=possible[0]
            else:mode='mixed'
        pairs[0]=novel
    plate=np.array([.43,-.17])+rng.uniform(-.035,.035,2)
    for _ in range(100):
        slots=np.array([[.42,.025],[.58,.20],[.58,-.235]])+rng.uniform([-.06,-.08],[.06,.08],(3,2))
        if min(np.linalg.norm(slots[i]-slots[j]) for i in range(3) for j in range(i))<.125:continue
        if min(np.linalg.norm(xy-plate) for xy in slots)<.11:continue
        break
    else:raise RuntimeError('Cannot sample separated object poses')
    order=rng.permutation(3).tolist()
    objects=[]
    for index,(color,kind) in enumerate(pairs):
        scale=float(rng.uniform(.90,1.10))*(1.35 if kind=='bowl' else 1.)
        rgba=np.asarray(COLORS[color]);rgba[:3]=np.clip(rgba[:3]+rng.uniform(-.035,.035,3),.02,.98)
        objects.append({'name':TARGETS[index],'color':color,'kind':kind,'rgba':rgba.tolist(),
            'scale':scale,'xy':slots[order[index]].tolist(),'slot':order[index],
            'yaw':float(rng.uniform(-math.pi,math.pi)),'mass':float(rng.uniform(.04,.065)),
            'friction':float(rng.uniform(.85,1.10)), 'novel_combination':(color,kind) in HELDOUT_PAIRS})
    return {'layout_seed':int(seed),'layout_split':split,'slot_order':order,'objects':objects,
            'plate_xy':plate.tolist(),'start_xyz':(np.array([.39,0.,1.08])+
                rng.uniform([-.025,-.025,-.015],[.025,.025,.015])).tolist(),
            'mode':mode,'distractor':bool(rng.integers(2)),
            'prompt_variant':int(rng.integers(len(TEMPLATES))), 'external_roi':ROI}


def apply_layout(world,spec):
    import pybullet as p
    from tactile_simulation.objects import geometry
    from generate import solve_pose
    world.clear()
    parts,height=geometry('plate')
    world.add_object('plate',parts,height,spec['plate_xy'],(.88,.89,.90,1.),slot=-1,mass=.12,name='target_plate')
    for index,item in enumerate(spec['objects']):
        parts,height=geometry(item['kind']);scale=item['scale']
        parts=[dict(part,size=tuple(v*scale for v in part['size']),xyz=tuple(v*scale for v in part['xyz'])) for part in parts]
        world.add_object(item['kind'],parts,height*scale,item['xy'],item['rgba'],slot=index,
                         yaw=item['yaw'],mass=item['mass'],name=item['name'])
        body=next(obj['body'] for obj in world.objects.values() if obj['name']==item['name'])
        p.changeDynamics(body,-1,lateralFriction=item['friction'],physicsClientId=world.client)
    if spec['distractor']:
        parts,height=geometry('bowl')
        world.add_object('bowl',parts,height,(.70,.31),(.55,.55,.55,1.),slot=3,mass=.05,name='gray_ramekin')
    world.home=tuple(np.r_[solve_pose(world,spec['start_xyz']),0.])
    world.reset_arm()
    for _ in range(30):world.step()


def render_external(world,roi=ROI):
    """Render the workspace crop directly, keeping the original camera extrinsics."""
    import pybullet as p
    from tactile_simulation import tabletop
    n=roi['reference_size'];left,top,right,bottom=roi['box'];size=roi['output_size']
    near,far=.02,4.;half=near*math.tan(math.radians(58/2))
    projection=p.computeProjectionMatrix(half*(2*left/n-1),half*(2*right/n-1),
        half*(1-2*bottom/n),half*(1-2*top/n),near,far)
    view=p.computeViewMatrix(tabletop.EXTERNAL_EYE,tabletop.EXTERNAL_TARGET,[0.,0.,1.])
    height=round(size*(bottom-top)/(right-left))
    rgba=p.getCameraImage(size,height,view,projection,renderer=p.ER_TINY_RENDERER,physicsClientId=world.client)[2]
    rgb=np.asarray(rgba,np.uint8).reshape(height,size,4)[:,:,:3]
    result=np.zeros((size,size,3),np.uint8);padding=(size-height)//2
    result[padding:padding+height]=rgb
    return result
