"""CPU-only plots with explicit coordinates, units and fixed numerical limits."""
import cv2
import numpy as np
from eef_video_workflow import text

COLORS=[(110,190,255),(255,185,105),(165,235,130)]
GREY=(190,190,190)

def plot(canvas,box,series,xlim,ylim,title,xlabel='',ylabel='',cursor=None,threshold=None):
    x,y,w,h=box;left=x+40;top=y+23;right=x+w-8;bottom=y+h-24
    text(canvas,title,x+4,y+14,.4)
    cv2.rectangle(canvas,(left,top),(right,bottom),(80,80,80),1)
    def project(xx,yy):
        return np.rint(np.c_[left+(np.asarray(xx)-xlim[0])/(xlim[1]-xlim[0])*(right-left),bottom-(np.asarray(yy)-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)]).astype(np.int32)
    for tick in np.linspace(ylim[0],ylim[1],3):
        py=int(project([xlim[0]],[tick])[0,1]);cv2.line(canvas,(left,py),(right,py),(45,45,45),1);text(canvas,f'{tick:.3f}'.rstrip('0').rstrip('.') or '0',x,py+4,.31)
    for tick in np.linspace(xlim[0],xlim[1],3):
        px=int(project([tick],[ylim[0]])[0,0]);text(canvas,f'{tick:.3f}'.rstrip('0').rstrip('.') or '0',px-10,bottom+14,.31)
    if threshold is not None:
        py=int(project([xlim[0]],[threshold])[0,1]);cv2.line(canvas,(left,py),(right,py),(100,170,255),1)
    if cursor is not None:
        px=int(project([np.clip(cursor,*xlim)],[ylim[0]])[0,0]);cv2.line(canvas,(px,top),(px,bottom),(120,120,120),1)
    for xx,yy,color in series:
        xx,yy=np.asarray(xx),np.asarray(yy);good=np.isfinite(xx)&np.isfinite(yy)
        pts=project(xx[good],yy[good])
        if len(pts)>1:
            for a,b in zip(pts,pts[1:]):
                ok,p,q=cv2.clipLine((left,top,right-left+1,bottom-top+1),tuple(a),tuple(b))
                if ok:cv2.line(canvas,p,q,color,1,cv2.LINE_AA)
        if len(pts):
            marker=pts[-1] if cursor is None else pts[max(0,min(len(pts)-1,int(np.searchsorted(xx[good],cursor,side='right')-1)))]
            if (cursor is None or cursor>=xx[good][0]) and left<=marker[0]<=right and top<=marker[1]<=bottom:cv2.circle(canvas,tuple(marker),3,color,-1)
    text(canvas,xlabel,right-45,y+h-1,.31)

def padded(values,minimum=.01):
    a=np.asarray(values);lo=float(np.nanmin(a));hi=float(np.nanmax(a));pad=max((hi-lo)*.08,minimum)
    return (lo-pad,hi+pad)
