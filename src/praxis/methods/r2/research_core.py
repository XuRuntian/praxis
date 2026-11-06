"""Common physical utilities; task methods are selected with documented exemplar evidence."""
import json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from numba import njit
def select(ep,config):
 blocks=[];spec=[]
 for col,pattern in config['selectors']:
  arr=ep['arrays'][col];names=ep['channels']['features'][col].get('names',[])
  if pattern=='all':idx=list(range(arr.shape[1]))
  else:
   import re
   if len(names)!=arr.shape[1]:raise ValueError('unnamed_dimensions_for_named_selector')
   idx=[i for i,n in enumerate(names) if re.search(pattern,n)]
  if not idx:raise ValueError('empty_feature_selection')
  z=gaussian_filter1d(arr[:,idx],config.get('smooth_seconds',.08)*ep['channels']['fps'],axis=0)
  blocks.append(z);spec += [(col,int(i),names[i] if len(names)==arr.shape[1] else f'component_{i}_unmapped') for i in idx]
 return np.concatenate(blocks,axis=1),spec

def normalization(x):
 lo,hi=np.quantile(x,[.05,.95],axis=0);scale=hi-lo
 active=scale>np.maximum(1e-6,np.max(abs(x),axis=0)*1e-4)
 return lo,np.where(active,scale,1.),active

@njit(cache=True)
def path_dtw(cost,penalty):
 n,m=cost.shape;d=np.full((n+1,m+1),np.inf);back=np.zeros((n,m),np.int8);d[0,0]=0
 for i in range(n):
  for j in range(m):
   diag=d[i,j];up=d[i,j+1]+penalty;left=d[i+1,j]+penalty
   if diag<=up and diag<=left:d[i+1,j+1]=cost[i,j]+diag;back[i,j]=0
   elif up<=left:d[i+1,j+1]=cost[i,j]+up;back[i,j]=1
   else:d[i+1,j+1]=cost[i,j]+left;back[i,j]=2
 i=n-1;j=m-1;pi=[];pj=[]
 while i>=0 and j>=0:
  pi.append(i);pj.append(j);b=back[i,j]
  if b==0:i-=1;j-=1
  elif b==1:i-=1
  else:j-=1
 return np.array(pi[::-1]),np.array(pj[::-1]),d[n,m]/len(pi)

def align(source,target,config,boundaries=None):
 x,spec=select(source,config);y,_=select(target,config)
 if config.get('relative_origin'):
  x=x-np.median(x[:max(1,int(source['channels']['fps']*.5))],axis=0)
  y=y-np.median(y[:max(1,int(target['channels']['fps']*.5))],axis=0)
 lo,sc,mask=normalization(x)
 if not mask.any():raise ValueError('no_physical_variation')
 x=(x[:,mask]-lo[mask])/sc[mask];y=(y[:,mask]-lo[mask])/sc[mask]
 if config.get('derivative_weight',0):
  w=config['derivative_weight'];x=np.concatenate([x,w*np.gradient(x,axis=0)*source['channels']['fps']],axis=1);y=np.concatenate([y,w*np.gradient(y,axis=0)*target['channels']['fps']],axis=1)
 if config.get('derivative_only'):
  x=gaussian_filter1d(np.gradient(x,axis=0)*source['channels']['fps'],2,axis=0)
  y=gaussian_filter1d(np.gradient(y,axis=0)*target['channels']['fps'],2,axis=0)
 stride=config.get('stride',2)
 si=np.unique(np.r_[np.arange(0,len(x),stride),len(x)-1]);ti=np.unique(np.r_[np.arange(0,len(y),stride),len(y)-1])
 c=cdist(x[si],y[ti],'sqeuclidean')/x.shape[1]
 c=np.minimum(c,config.get('cost_cap',25.))
 pi,pj,score=path_dtw(c,config.get('warp_penalty',.02))
 sf=si[pi];tf=ti[pj]
 uniq=np.unique(sf);mapping=np.array([np.median(tf[sf==i]) for i in uniq])
 if boundaries is None:boundaries=[s['start_frame'] for s in source['ann']['subtasks'][1:]]
 pred=np.rint(np.interp(boundaries,uniq,mapping)).astype(int).tolist()
 return pred,float(score),{'selected_features':spec,'path_length':len(pi),'active_dimensions':int(mask.sum())}

def semantic_key(label):
 def act(k):
  a=label.get(k)
  if not a:return None
  return [a.get('subject'),a.get('skill'),{k:v for k,v in a.get('slots',{}).items() if 'object' in k or k in ['interaction_target','destination_subject']}]
 return [act('primary_action'),act('auxiliary_action')]
def labels_for(ex):
 seq=[];conflicts=[]
 for i,(a,b) in enumerate(zip(ex[0]['ann']['subtasks'],ex[1]['ann']['subtasks'])):
  lab={k:v for k,v in a.items() if k not in ['id','start_frame','end_frame']}
  if semantic_key(a)!=semantic_key(b):seq.append(None);conflicts.append(i)
  else:seq.append(lab)
 return seq,conflicts

def crossing_events(y,threshold,direction,hold=1):
 condition=y<=threshold if direction<0 else y>=threshold
 changes=np.flatnonzero(condition & ~np.r_[condition[0],condition[:-1]])
 return [int(i) for i in changes if condition[i:min(len(y),i+hold)].all()]

def learn_grip(ex,event_spec):
 events=[]
 for spec in event_spec:
  idx=spec['index'];col=spec.get('column','observation.state');bi=spec['boundary_index'];vals=[]
  for ep in ex:
   y=gaussian_filter1d(ep['arrays'][col][:,idx],1.)
   b=ep['ann']['subtasks'][bi+1]['start_frame'];vals.append(float(y[b]))
  events.append({**spec,'column':col,'threshold':float(np.mean(vals)),'exemplar_boundary_values':vals})
 return events

def grip_predict(ep,events):
 out=[];prev=0
 for event in events:
  y=gaussian_filter1d(ep['arrays'][event['column']][:,event['index']],1.)
  candidates=[i for i in crossing_events(y,event['threshold'],event['direction'],event.get('hold',2)) if i>prev]
  if not candidates:return None
  b=candidates[-1] if event.get('occurrence')=='last' else candidates[0]
  out.append(b);prev=b
 return out
