from .research_core import *

def press_events(ep,theta):
 x=ep['arrays']['observation.state'];fps=ep['channels']['fps'];g=gaussian_filter1d(x[:,7],1);z=gaussian_filter1d(x[:,10],2)
 high=g>150;edges=np.diff(np.r_[False,high,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
 ans=[];dz=np.gradient(z)*fps
 for a,b in zip(starts,ends):
  if b-a<=fps*.5:continue
  k=a+int(np.argmin(z[a:b]));idx=np.flatnonzero(dz[k:b]>=theta);ans.append(int(k+idx[0] if len(idx) else k))
 return ans

def crop(ep,a,b):return {**ep,'arrays':{k:v[a:b] for k,v in ep['arrays'].items()},'n':b-a}
def hierarchy(source,target,events):
 sa=grip_predict(source,events);ta=grip_predict(target,events)
 if sa is None or ta is None:return None
 sb=[0]+sa+[source['n']];tb=[0]+ta+[target['n']];allb=[s['start_frame'] for s in source['ann']['subtasks'][1:]];result=list(ta)
 for i,pat in enumerate(['right_arm|right_gripper','left_arm|left_gripper','arm_joint|gripper']):
  bs=[b-sb[i] for j,b in enumerate(allb) if j not in [1,4] and sb[i]<b<sb[i+1]]
  cfg={'selectors':[['observation.state',pat]],'smooth_seconds':.12,'stride':2,'derivative_weight':.1,'warp_penalty':.02}
  if bs:
   p,c,_=align(crop(source,sb[i],sb[i+1]),crop(target,tb[i],tb[i+1]),cfg,bs);result += [v+tb[i] for v in p]
 return sorted(result)

def motion_onset(ep,scale):
 x=gaussian_filter1d(ep['arrays']['observation.state'][:,6:9],1,axis=0)
 ref=np.median(x[:max(1,int(ep['channels']['fps']*.5))],axis=0)
 d=np.linalg.norm((x-ref)/np.asarray(scale),axis=1);active=d>.03
 for i in np.flatnonzero(active):
  if i+3<len(active) and active[i:i+3].all():return int(i)
 return None

def aligned(sources,target,cfg,boundaries=None):
 preds=[];costs=[]
 for i,s in enumerate(sources):
  p,c,_=align(s,target,cfg,None if boundaries is None else boundaries[i]);preds.append(p);costs.append(c)
 return np.rint(np.mean(preds,axis=0)).astype(int).tolist(),costs

def infer(sources,target,p):
 method=p['method'];notes=[];costs=[];stage_indices=list(range(len(p['labels'])))
 if method=='observable_suffix':
  start=motion_onset(target,p['onset_scale'])
  if start is None:return [],[None],['no_observable_suffix'],[]
  sc=[];bs=[]
  for s in sources:
   a=motion_onset(s,p['onset_scale'])
   if a is None:raise ValueError('frozen_source_has_no_motion')
   sc.append(crop(s,a,s['n']));bs.append([seg['start_frame']-a for seg in s['ann']['subtasks'][7:]])
  local,costs=aligned(sc,crop(target,start,target['n']),p['alignment_config'],bs)
  b=[start]+[v+start for v in local];labs=[None]+p['labels'][6:]
  return b,labs,['early_stages_unobservable','flat_prefix_not_time_warped','two_exemplar_subject_conflicts'],costs
 if method=='grip_fsm':
  b=grip_predict(target,p['events'])
  if b is not None and 'handoff_gate' in p:
   gate=p['handoff_gate']
   if target['arrays']['observation.state'][b[1],gate['recipient_index']]>gate['maximum_closed_value']:
    b=None;notes.append('receiver_not_closed_at_donor_release')
  if b is None:
   b,costs=aligned(sources,target,p['alignment_config']);notes.append('gripper_events_incomplete_used_physical_path_fallback')
 elif method=='hybrid_press':
  b,costs=aligned(sources,target,p['alignment_config']);press=press_events(target,p['press_velocity_threshold'])
  if len(press)==2 and press[0]<b[1]<press[1]:b[0],b[2]=press
  else:notes.append('press_events_not_unique_used_path_correspondence')
 elif method=='hybrid_card':
  b,costs=aligned(sources,target,p['alignment_config']);event_b=grip_predict(target,p['events'])
  if event_b is not None:
   for ev,v in zip(p['events'],event_b):b[ev['boundary_index']]=v
  else:notes.append('gripper_events_incomplete_used_path_correspondence')
  right_cfg={'selectors':[['observation.state','right_arm|right_gripper']],'smooth_seconds':.08,'stride':2,'warp_penalty':.02}
  right,_=aligned(sources,target,right_cfg);b[3]=right[3]
 elif method=='anchored_wipe':
  hs=[hierarchy(s,target,p['anchors']) for s in sources]
  if all(h is not None and len(h)==7 for h in hs):b=np.rint(np.mean(hs,axis=0)).astype(int).tolist()
  else:b,costs=aligned(sources,target,p['alignment_config']);notes.append('release_anchors_missing_used_arm_path')
  cfg={'selectors':[['observation.state','arm_joint|gripper']],'smooth_seconds':.15,'stride':2,'warp_penalty':.005,'derivative_only':True}
  motion,c=aligned(sources,target,cfg);costs+=c
  for i in [3,6]:b[i]=motion[i]
  notes.append('pickup_and_arm_handoff_boundaries_not_precisely_identifiable')
 else:b,costs=aligned(sources,target,p['alignment_config'])
 return b,p['labels'],notes,costs

def predict(sources,target,p,dataset,eid,run):
 base=dict(dataset=dataset,episode_id=eid,frame_count=target['n'],rule_version=run,status='review',segments=[],review_reasons=[],error=None)
 # A completely flat selected physical record cannot support temporal labels.
 if sources and 'alignment_config' in p:
  tx,_=select(target,p['alignment_config'])
  sx,_=select(sources[0],p['alignment_config'])
  _,scale,active=normalization(sx)
  if active.any() and np.max(np.ptp(tx[:,active],axis=0)/scale[active])<.001:
   base.update(segments=[{'start_frame':0,'end_frame':target['n'],'label':None}],review_reasons=['selected_physical_record_has_no_observable_phase_changes'])
   return base
 b,labels,notes,costs=infer(sources,target,p)
 if not b and labels==[None]:
  base.update(segments=[{'start_frame':0,'end_frame':target['n'],'label':None}],review_reasons=notes+['physical_evidence_insufficient']);return base
 if len(labels)!=len(b)+1 or any(not 0<v<target['n'] for v in b) or any(a>=z for a,z in zip(b,b[1:])):
  # Do not synthesize timestamps/relative-duration boundaries to repair failed correspondence.
  base.update(status='abstain',review_reasons=['physical_correspondence_not_ordered']+notes);return base
 edges=[0]+b+[target['n']]
 base['segments']=[{'start_frame':a,'end_frame':z,'label':lab} for a,z,lab in zip(edges,edges[1:],labels)]
 base['review_reasons']=['labels_transferred_from_exemplar_phase_correspondence_not_object_recognition','two_exemplar_development_only']+notes
 if any(l is None for l in labels):base['review_reasons'].append('partial_semantic_ambiguity')
 if p.get('unresolved'):base['review_reasons'].append('exemplar_boundary_uncertainty')
 return base
