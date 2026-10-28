"""CPU-only cue raster rules shared by production and replay."""
import numpy as np

def visibility(frames,ids,task,command,common_start=False):
    counts={n:{c:int(np.sum(frames[c+'_segmentation']==sid)) for c in ('body','wrist')} for n,sid in ids.items()}
    def count(n):return max(counts[n].values())
    if task=='M1':required={'target'+str(command):4,'axis%d_0'%command:2,'axis%d_1'%command:2}
    elif task=='M2':required={'obstacle':16,'goal':4}
    else:required={'black_A':4,'white_B':4}
    failures=[n for n,threshold in required.items() if count(n)<threshold]
    centers={};color_checks={}
    if task=='M1':
        for a in (0,1):
            n='axis%d_%d'%(command,a);c=max(counts[n],key=counts[n].get);mask=frames[c+'_segmentation']==ids[n]
            pixels=frames[c][mask];yx=np.argwhere(mask)
            centers[n]=dict(camera=c,xy=np.mean(yx[:,::-1],axis=0).tolist() if len(yx) else None)
            mean=np.mean(pixels,axis=0) if len(pixels) else np.zeros(3)
            color_checks[n]=bool(len(pixels) and mean[a]>max(np.delete(mean,a)))
            if not color_checks[n]:failures.append(n+'_COLOR')
        first,second=[centers['axis%d_%d'%(command,a)] for a in (0,1)]
        if first['camera']==second['camera'] and first['xy'] is not None and second['xy'] is not None and np.linalg.norm(np.array(first['xy'])-second['xy'])<=0:
            failures.append('AXIS_CENTROIDS_IDENTICAL')
    sectors=[any(count('contour'+str(i))>0 for i in (2*s,2*s+1)) for s in range(12)]
    if task=='M3' and common_start and sum(sectors)<8:failures.append('CONTOUR_LT8_SECTORS')
    return dict(pass_necessary=not failures,failures=failures,required=required,counts=counts,
        axis_centroids=centers,axis_color_checks=color_checks,visible_contour_sectors=sectors)


