"""Exact D4 context-completion expert. Standard library; no task IDs.

For each solid single-color occluder, find a completely observed donor patch
under each D4 transform and translation. Rank by the largest rectangular
known context matching exactly. Return only a uniquely best reconstructed
patch, with more observed context cells than occluded cells. The mask color
and output orientation are selected using the provided demonstrations.

Coordinates in diagnostics use half-open context bounds (top,bottom,left,right).
"""
from arc2_d4_tools import d4

def rectangles(g,color):
 pts=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==color]
 if not pts:return None
 r0=min(r for r,c in pts);r1=max(r for r,c in pts);c0=min(c for r,c in pts);c1=max(c for r,c in pts)
 if len(pts)!=(r1-r0+1)*(c1-c0+1):return None
 return r0,r1,c0,c1

def best_context(g,a,dr,dc,bbox,color):
 H,W=len(g),len(g[0]);A,B=len(a),len(a[0]);r0,r1,c0,c1=bbox
 lo=max(0,-dr);hi=min(H,A-dr);left=max(0,-dc);right=min(W,B-dc)
 limits={}
 for r in range(lo,hi):
  src=a[r+dr];row=g[r]
  if not all(src[c+dc]!=color and (row[c]==color or row[c]==src[c+dc]) for c in range(c0,c1+1)):
   continue
  l=c0
  while l>left and src[l-1+dc]!=color and row[l-1]==src[l-1+dc]:l-=1
  u=c1+1
  while u<right and src[u+dc]!=color and row[u]==src[u+dc]:u+=1
  limits[r]=(l,u)
 if any(r not in limits for r in range(r0,r1+1)):return None
 top=r0
 while top>lo and top-1 in limits:top-=1
 bottom=r1+1
 while bottom<hi and bottom in limits:bottom+=1
 best=(-1,None)
 for t in range(top,r0+1):
  l=max(limits[r][0] for r in range(t,r1+1));u=min(limits[r][1] for r in range(t,r1+1))
  for b in range(r1+1,bottom+1):
   if b>r1+1:l=max(l,limits[b-1][0]);u=min(u,limits[b-1][1])
   area=(b-t)*(u-l)
   if area>best[0]:best=area,(t,b,l,u)
 return best

def search(g,color):
 box=rectangles(g,color)
 if box is None:raise ValueError('Not a rectangle')
 r0,r1,c0,c1=box;h=r1-r0+1;w=c1-c0+1; found={}
 for name,a in d4(g):
  H,W=len(a),len(a[0])
  for sr in range(H-h+1):
   for sc in range(W-w+1):
    out=[row[sc:sc+w] for row in a[sr:sr+h]]
    if any(color in row for row in out):continue
    result=best_context(g,a,sr-r0,sc-c0,box,color)
    if result is None:continue
    area,context=result;score=area-h*w
    if score<=0:continue
    key=tuple(map(tuple,out));entry={'score':score,'orientation':name,'donor_row':sr,'donor_col':sc,'context':context,'prediction':out}
    if key not in found or score>found[key]['score']:found[key]=entry
 return sorted(found.values(),key=lambda e:e['score'],reverse=True)



class Abstain(ValueError):
    """The demonstrated structural rule has no unambiguous supported prediction."""


def reconstruct(g,color):
    box=rectangles(g,color)
    if box is None: raise Abstain('Mask color does not form one solid rectangle')
    area=(box[1]-box[0]+1)*(box[3]-box[2]+1)
    ranked=search(g,color)
    if not ranked: raise Abstain('No exact context match')
    best=ranked[0]
    if best['score']<=area:
        raise Abstain('Observed exact context does not exceed unknown area')
    if len(ranked)>1 and ranked[1]['score']==best['score']:
        raise Abstain('Maximal contexts disagree about reconstruction')
    return best['prediction'],{'mask_color':color,'mask_bbox':box,
        'mask_area':area,'best_context_known_cells':best['score'],
        'runner_up_known_cells':ranked[1]['score'] if len(ranked)>1 else 0,
        'donor_orientation':best['orientation'],'donor_row':best['donor_row'],
        'donor_col':best['donor_col'],'context':best['context']}


def fit(train_pairs):
    if not train_pairs: raise Abstain('Demonstrations required')
    colors=set(v for row in train_pairs[0]['input'] for v in row)
    for p in train_pairs:
        colors.intersection_update(v for row in p['input'] for v in row)
    models=[]
    for color in sorted(colors):
        labels=None
        for p in train_pairs:
            box=rectangles(p['input'],color)
            if box is None or any(color in row for row in p['output']):
                labels=set();break
            if (box[1]-box[0]+1)*(box[3]-box[2]+1)!=len(p['output'])*len(p['output'][0]):
                labels=set();break
            try: grid,_=reconstruct(p['input'],color)
            except Abstain: labels=set();break
            matched={name for name,g in d4(grid) if g==p['output']}
            labels=matched if labels is None else labels & matched
            if not labels: break
        if labels:
            models.extend((color,label) for label in sorted(labels))
    if not models: raise Abstain('No supported model reproduces every demonstration')
    return models


def predict_candidates(train_pairs,test_input):
    outputs=[]
    for color,label in fit(train_pairs):
        try: restored,_=reconstruct(test_input,color)
        except Abstain: continue
        out=dict(d4(restored))[label]
        if out not in outputs: outputs.append(out)
    if len(outputs)!=1:
        raise Abstain('Demonstrated models do not yield one supported answer')
    return outputs


def predict(train_pairs,test_input):
    return predict_candidates(train_pairs,test_input)[0]

