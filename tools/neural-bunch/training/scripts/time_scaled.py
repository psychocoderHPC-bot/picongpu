import time, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuralbunch import scaled
from neuralbunch.fast_reference import load_fast_reference
mod=load_fast_reference()
split=sys.argv[1] if len(sys.argv)>1 else 'train'
n=int(sys.argv[2]) if len(sys.argv)>2 else 160
descs=[d for d in scaled.descriptors_for_split(split) if d['kind']=='scenario' and d['direction'] in ('axis_x','random') and d['position_mode'] in ('near','far','subcell')]
print('candidates',len(descs),flush=True)
res=[]
for j,d in enumerate(descs[:n]):
    s=scaled.make_sample(d,split,20261101+{'train':0,'val':1000,'test':2000}[split])
    q=scaled.scaled_quadrature(s)
    t=time.time(); scaled._pack_samples([s],mod); dt=time.time()-t
    dm=scaled._min_distance_cells(s)
    res.append((dt,dm,q['order'],tuple(round(x,3) for x in d['aspects']),d['position_mode'],d['gamma']))
    print(j,'%.3fs'%dt,'dmin=%.2f'%dm,'ord',q['order'],'a',res[-1][3],d['position_mode'],'g%.1f'%d['gamma'],flush=True)
arr=np.array([r[0] for r in res])
print('n',len(res),'total %.2f'%arr.sum(),'mean %.3f'%arr.mean(),'p99 %.3f'%np.percentile(arr,99),'max %.3f'%arr.max())
for r in sorted(res,reverse=True)[:8]: print('SLOW %.3fs dmin=%.2f ord=%d a=%s %s g=%.1f'%r)
