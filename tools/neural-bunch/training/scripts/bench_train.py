import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from neuralbunch.reference_data import load_split
from neuralbunch.train import TrainConfig, Batcher, build_model_from_config, make_optimizer, loss_and_terms
root=sys.argv[1]; dev=torch.device(sys.argv[2] if len(sys.argv)>2 else 'cuda:0')
data=load_split(root,'train')
cfg=TrainConfig(hidden=(512,512),latent=256,batch_tiles=64,max_steps=10,seed=0)
m=build_model_from_config(cfg).to(dev); opt=make_optimizer(m,cfg)
b=Batcher(data,dev); mc=cfg.model_config()
for step in range(5):
    ids=b.next_tiles(64); f,w,ti,di,dw,t,s0,T=b.batch(ids,mc)
    p=m(f,w,ti,di,dw,T); l,_,_=loss_and_terms(p,t,s0,cfg.force_weight); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
torch.cuda.synchronize(); t0=time.time(); n=40
for step in range(n):
    ids=b.next_tiles(64); f,w,ti,di,dw,t,s0,T=b.batch(ids,mc)
    p=m(f,w,ti,di,dw,T); l,_,_=loss_and_terms(p,t,s0,cfg.force_weight); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
torch.cuda.synchronize(); dt=time.time()-t0
print('tiles',data.n_tiles,'steps/s %.1f'%(n/dt),'ms/step %.1f'%(1000*dt/n),'mem MB',torch.cuda.max_memory_allocated()//2**20)
