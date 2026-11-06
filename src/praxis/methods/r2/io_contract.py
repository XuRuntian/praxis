from .research_core import *
class InputContractError(ValueError):pass

def load_target(path,channels,eid,columns=None):
 if columns is None:columns=[c for c in channels['features'] if c.startswith('observation.state') or c=='action']
 requested=list(dict.fromkeys(columns+['frame_index','timestamp','episode_index']))
 table=pq.read_table(path,columns=requested)
 eids=np.asarray(table['episode_index'].to_pylist())
 if not np.all(eids==eid):
  import pyarrow as pa
  table=table.filter(pa.array(eids==eid))
 frame=np.asarray(table['frame_index'].to_pylist());ts=np.asarray(table['timestamp'].to_pylist(),float)
 if len(frame)<2:raise InputContractError('too_few_frames')
 if not np.array_equal(frame,np.arange(len(frame))):raise InputContractError('original_frame_mapping_not_contiguous')
 if ts.shape!=(len(frame),) or not np.isfinite(ts).all():raise InputContractError('invalid_timestamp')
 arrays={}
 for col in columns:
  x=np.asarray(table[col].to_pylist(),float)
  expected=channels['features'][col]['shape'][0]
  if x.shape!=(len(frame),expected):raise InputContractError('required_channel_shape_mismatch')
  if not np.isfinite(x).all():raise InputContractError('nonfinite_required_physical_values')
  arrays[col]=x
 notes=[]
 if np.any(np.diff(ts)<=0):notes.append('timestamps_nonmonotonic_using_original_frame_and_declared_fps')
 if not np.allclose(np.diff(ts),1/channels['fps'],rtol=.05,atol=1e-5):notes.append('timestamp_spacing_differs_from_declared_fps_feature_derivatives_are_per_frame_rate')
 return {'arrays':arrays,'channels':channels,'eid':eid,'n':len(frame),'timestamps':ts,'input_notes':notes},[{'frame_index':int(f),'timestamp':float(t)} for f,t in zip(frame,ts)]

def load_sources(directory,p):
 result=[]
 with np.load(Path(directory)/'templates.npz',allow_pickle=False) as bundle:
  for i,ann in enumerate(p['source_annotations']):
   arrays={key.split('::',1)[1]:bundle[key] for key in bundle.files if key.startswith(f'{i}::')}
   result.append({'arrays':arrays,'channels':p['channels'],'ann':ann,'eid':ann['episode_id'],'n':ann['frame_count']})
 return result
