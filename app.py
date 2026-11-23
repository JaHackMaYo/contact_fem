import io
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

st.set_page_config(page_title='FEM 2D - Contatto semianello', layout='wide')

# -----------------------------
# FEM: materiale ed elemento CST
# -----------------------------
def constitutive_plane_strain(E, nu):
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, 0.5 * (1 - 2 * nu)]])


def cst_stiffness(xy, D):
    x1, y1 = xy[0]; x2, y2 = xy[1]; x3, y3 = xy[2]
    twoA = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    A = 0.5 * abs(twoA)
    if A <= 1e-14:
        raise ValueError('Elemento degenerato.')
    b = np.array([y2-y3, y3-y1, y1-y2]) / twoA
    c = np.array([x3-x2, x1-x3, x2-x1]) / twoA
    B = np.array([[b[0],0,b[1],0,b[2],0], [0,c[0],0,c[1],0,c[2]],
                  [c[0],b[0],c[1],b[1],c[2],b[2]]])
    return A * (B.T @ D @ B), B, A


def element_dofs(conn):
    d = np.empty(6, dtype=int)
    d[0::2] = 2*conn; d[1::2] = 2*conn+1
    return d

# -----------------------------
# Mesh strutturate scritte da zero
# -----------------------------
def graded_partition(total,h0,growth=1.25,hmax=None,max_cells=300):
    """Lunghezze crescenti da h0, con chiusura esatta del dominio."""
    if total<=0 or h0<=0 or growth<1: raise ValueError('Richiesti total>0, h0>0 e growth>=1.')
    hmax=total if hmax is None else max(h0,hmax); sizes=[]; rem=float(total); hi=min(h0,rem)
    while rem>1e-12*max(1.,total):
        if len(sizes)>=max_cells: raise ValueError('Mesh troppo fitta: aumentare la dimensione al contatto.')
        use=min(hi,hmax,rem)
        if 0<rem-use<0.35*use: use=rem
        sizes.append(use); rem-=use; hi=min(hi*growth,hmax)
    sizes=np.asarray(sizes,float); sizes*=total/sizes.sum(); return sizes

def mesh_half_ring(ri,re,h_contact,growth,hmax_factor):
    hmax=h_contact*hmax_factor
    arcs=graded_partition(.5*np.pi*re,h_contact,growth,hmax)
    theta=-.5*np.pi+np.r_[0.,np.cumsum(arcs)/re]; theta[-1]=0.
    dr=graded_partition(re-ri,h_contact,growth,hmax)
    rs=(re-np.r_[0.,np.cumsum(dr)])[::-1]; rs[0]=ri
    yc=re; nodes=np.array([[r*np.cos(t),yc+r*np.sin(t)] for t in theta for r in rs])
    nr=len(rs)-1; nt=len(theta)-1; elems=[]
    for j in range(nt):
        for i in range(nr):
            n0=j*(nr+1)+i; n1=n0+1; n2=n0+nr+1; n3=n2+1
            elems += [[n0,n1,n3],[n0,n3,n2]] if (i+j)%2==0 else [[n0,n1,n2],[n1,n3,n2]]
    contact=np.array([j*(nr+1)+nr for j in range(nt+1)],int)
    top=np.arange(nt*(nr+1),(nt+1)*(nr+1)); sym=np.arange(nr+1)
    return nodes,np.asarray(elems),contact,top,sym,theta,rs

def mesh_block_from_ring(L,h,rnodes,rcontact,h_contact,growth,hmax_factor):
    hmax=h_contact*hmax_factor
    xc=np.unique(np.round(rnodes[rcontact,0],12)); xc.sort(); xe=xc[-1]
    if L<=xe: raise ValueError('L deve essere maggiore di re.')
    start=max(h_contact,xc[-1]-xc[-2]); tail=graded_partition(L-xe,start,growth,hmax)
    xs=np.r_[xc,xe+np.cumsum(tail)]; xs[-1]=L
    dy=graded_partition(h,h_contact,growth,hmax); ys=-np.r_[0.,np.cumsum(dy)][::-1]; ys[0]=-h; ys[-1]=0.
    nx=len(xs)-1; ny=len(ys)-1; nodes=np.array([[x,y] for y in ys for x in xs]); elems=[]
    for j in range(ny):
        for i in range(nx):
            n0=j*(nx+1)+i; n1=n0+1; n2=n0+nx+1; n3=n2+1
            elems += [[n0,n1,n3],[n0,n3,n2]] if (i+j)%2==0 else [[n0,n1,n2],[n1,n3,n2]]
    top=np.arange(ny*(nx+1),(ny+1)*(nx+1)); bottom=np.arange(nx+1); sym=np.arange(0,(ny+1)*(nx+1),nx+1)
    return nodes,np.asarray(elems),top,bottom,sym,xs,ys

def build_model(p):
    nr,er,rc,rt,rs,theta,radii=mesh_half_ring(p['ri'],p['re'],p['h_contact'],p['mesh_growth'],p['hmax_factor'])
    nb,eb,bt,bb,bs,xs,ys=mesh_block_from_ring(p['L'],p['h'],nr,rc,p['h_contact'],p['mesh_growth'],p['hmax_factor'])
    off=len(nb); nodes=np.vstack([nb,nr]); elems=np.vstack([eb,er+off]); mats=np.r_[np.zeros(len(eb),int),np.ones(len(er),int)]
    return dict(nodes=nodes,elems=elems,mats=mats,nblock=len(nb),block_top=bt,block_bottom=bb,block_sym=bs,
                ring_contact=rc+off,ring_top=rt+off,ring_sym=rs+off,
                mesh_info=dict(nx=len(xs)-1,ny=len(ys)-1,nr=len(radii)-1,nt=len(theta)-1,
                min_dx=np.min(np.diff(xs)),max_dx=np.max(np.diff(xs)),min_dy=np.min(np.diff(ys)),max_dy=np.max(np.diff(ys))))
# -----------------------------
# Assemblaggio e contatto
# -----------------------------
def assemble_bulk(model, Eblock, nublock, Ering, nuring):
    n=len(model['nodes']); K=np.zeros((2*n,2*n)); cache=[]
    Ds=[constitutive_plane_strain(Eblock,nublock), constitutive_plane_strain(Ering,nuring)]
    for e,conn in enumerate(model['elems']):
        ke,B,A=cst_stiffness(model['nodes'][conn],Ds[model['mats'][e]])
        dofs=element_dofs(conn); K[np.ix_(dofs,dofs)] += ke
        cache.append((B,Ds[model['mats'][e]],A))
    return K,cache


def contact_candidates(model, U):
    X=model['nodes'] + U.reshape(-1,2)
    master=model['block_top']; slave=model['ring_contact']
    order=np.argsort(X[master,0]); master=master[order]
    contacts=[]
    xm=X[master,0]
    for s in slave:
        x=X[s,0]
        k=int(np.clip(np.searchsorted(xm,x)-1,0,len(master)-2))
        a,b=master[k],master[k+1]; xa,xb=X[a],X[b]
        t=xb-xa; ll=float(t@t)
        xi=float(np.clip(((X[s]-xa)@t)/ll,0.0,1.0)) if ll>0 else 0.0
        n=np.array([-t[1],t[0]])/np.sqrt(ll)
        if n[1] < 0: n=-n
        proj=(1-xi)*xa+xi*xb
        gap=float((X[s]-proj)@n)
        contacts.append((s,a,b,xi,n,gap))
    return contacts


def contact_matrix_rhs(model, U, kp, active_prev=None):
    ndof=2*len(model['nodes']); Kc=np.zeros((ndof,ndof)); rhs=np.zeros(ndof)
    active=[]; data=[]
    X0=model['nodes']
    for s,a,b,xi,n,gap in contact_candidates(model,U):
        if gap < 0.0:
            dofs=np.array([2*s,2*s+1,2*a,2*a+1,2*b,2*b+1])
            G=np.r_[n, -(1-xi)*n, -xi*n]
            g0=float((X0[s]-((1-xi)*X0[a]+xi*X0[b]))@n)
            Kc[np.ix_(dofs,dofs)] += kp*np.outer(G,G)
            rhs[dofs] += -kp*g0*G
            active.append((s,a,b)); data.append((s,gap,-kp*gap,xi))
    return Kc,rhs,tuple(active),data


def solve_dirichlet(K, rhs, prescribed):
    nd=len(rhs); fixed=np.array(sorted(prescribed),int); vals=np.array([prescribed[i] for i in fixed])
    free=np.setdiff1d(np.arange(nd),fixed)
    U=np.zeros(nd); U[fixed]=vals
    U[free]=np.linalg.solve(K[np.ix_(free,free)], rhs[free]-K[np.ix_(free,fixed)]@vals)
    return U


def solve_model(model,p,progress=None):
    Kbulk,cache=assemble_bulk(model,p['E_block'],p['nu_block'],p['E_ring'],p['nu_ring'])
    nd=Kbulk.shape[0]; U=np.zeros(nd); history=[]; last_contact=[]
    fixed_zero={}
    for n in model['block_bottom']:
        fixed_zero[2*n]=0.; fixed_zero[2*n+1]=0.
    for n in np.unique(np.r_[model['block_sym'],model['ring_sym']]): fixed_zero[2*n]=0.
    for step in range(1,p['n_steps']+1):
        delta=p['delta_max']*step/p['n_steps']; active_old=None; converged=False
        for it in range(1,p['max_iter']+1):
            Kc,rhs,active,cdata=contact_matrix_rhs(model,U,p['kp'])
            prescribed=dict(fixed_zero)
            for n in model['ring_top']: prescribed[2*n+1]=-delta
            Unew=solve_dirichlet(Kbulk+Kc,rhs,prescribed)
            rel=np.linalg.norm(Unew-U)/max(np.linalg.norm(Unew),max(delta,1e-12))
            U=(1-p['relax'])*U+p['relax']*Unew
            if rel < p['tol'] and active==active_old:
                converged=True; break
            active_old=active
        # reazione strutturale ai DOF verticali imposti (contatto è interno)
        reaction=Kbulk@U
        half=-sum(reaction[2*n+1] for n in model['ring_top'])
        history.append([delta,2*half,len(active),it,converged])
        last_contact=cdata
        if progress: progress.progress(step/p['n_steps'], text=f'Incremento {step}/{p["n_steps"]}')
        if not converged: break
    return U,np.array(history,dtype=object),cache,last_contact


def element_results(model,U,cache):
    vm=[]; syy=[]
    for conn,(B,D,A) in zip(model['elems'],cache):
        sig=D@(B@U[element_dofs(conn)])
        sx,sy,t=sig
        # von Mises 3D coerente con plane strain: sigma_z=nu*(sx+sy), nu materiale recuperato indirettamente non disponibile
        vm.append(np.sqrt(max(sx*sx-sx*sy+sy*sy+3*t*t,0.0)))
        syy.append(sy)
    return np.array(vm),np.array(syy)

# -----------------------------
# Solutore didattico incrementale
# -----------------------------
def init_state(model,p):
    K,cache=assemble_bulk(model,p['E_block'],p['nu_block'],p['E_ring'],p['nu_ring'])
    fixed={}
    for n in model['block_bottom']: fixed[2*n]=0.; fixed[2*n+1]=0.
    for n in np.unique(np.r_[model['block_sym'],model['ring_sym']]): fixed[2*n]=0.
    return {'K':K,'cache':cache,'fixed':fixed,'U':np.zeros(K.shape[0]),'step':0,'history':[],'shots':[],'finished':False}

def one_step(model,p,S):
    if S['finished']: return S
    step=S['step']+1; delta=p['delta_max']*step/p['n_steps']; U=S['U'].copy(); old=None; log=[]; converged=False
    for it in range(1,p['max_iter']+1):
        Kc,rhs,active,cdata=contact_matrix_rhs(model,U,p['kp']); bc=dict(S['fixed'])
        for n in model['ring_top']: bc[2*n+1]=-delta
        V=solve_dirichlet(S['K']+Kc,rhs,bc)
        err=np.linalg.norm(V-U)/max(np.linalg.norm(V),delta,1e-12)
        log.append({'iterazione':it,'variazione_relativa':err,'contatti_attivi':len(active),'penetrazione_max':max([-q[1] for q in cdata],default=0.)})
        U=(1-p['relax'])*U+p['relax']*V
        if err<p['tol'] and active==old: converged=True; break
        old=active
    reaction=S['K']@U; force=-2*sum(reaction[2*n+1] for n in model['ring_top'])
    amap={q[0]:q[2] for q in cdata}; diag=[]
    for sn,a,b,xi,n,gap in contact_candidates(model,U):
        diag.append({'slave':sn,'a':a,'b':b,'xi':xi,'gap':gap,'pressure':amap.get(sn,0.),'active':sn in amap})
    row=[delta,force,sum(q['active'] for q in diag),it,converged,min(q['gap'] for q in diag),max(q['pressure'] for q in diag)]
    S['U']=U; S['step']=step; S['history'].append(row); S['shots'].append({'U':U.copy(),'delta':delta,'force':force,'diag':diag,'log':log,'converged':converged})
    S['finished']=step>=p['n_steps'] or not converged
    return S

def signature(p): return tuple((k,float(v)) for k,v in p.items())

def fig_mesh(model, U=None, scale=1.0, mesh=True, title=''):
    import plotly.graph_objects as go

    X = model['nodes'].copy() if U is None else model['nodes'] + scale * U.reshape(-1, 2)
    fig = go.Figure()

    materials = [
        (0, 'rgba(255,40,40,0.42)', 'Blocco'),
        (1, 'rgba(35,35,35,0.78)', 'Semianello'),
    ]

    # Riempimento dei domini. I bordi vengono aggiunti separatamente,
    # così la mesh del semianello rimane visibile sul fondo scuro.
    for mat, fill_color, name in materials:
        first = True
        for tri in model['elems'][model['mats'] == mat]:
            z = X[np.r_[tri, tri[0]]]
            fig.add_trace(go.Scatter(
                x=z[:, 0], y=z[:, 1], mode='lines', fill='toself',
                fillcolor=fill_color,
                line=dict(color=fill_color, width=0.2),
                name=name, legendgroup=name, showlegend=first,
                hoverinfo='skip'
            ))
            first = False

    if mesh:
        mesh_styles = [
            (0, 'rgba(110,0,0,0.82)', 0.8, 'Mesh blocco'),
            (1, 'rgba(255,255,255,0.95)', 1.1, 'Mesh semianello'),
        ]
        for mat, line_color, line_width, name in mesh_styles:
            x_lines, y_lines = [], []
            for tri in model['elems'][model['mats'] == mat]:
                z = X[np.r_[tri, tri[0]]]
                x_lines.extend(z[:, 0].tolist() + [None])
                y_lines.extend(z[:, 1].tolist() + [None])
            fig.add_trace(go.Scatter(
                x=x_lines, y=y_lines, mode='lines',
                line=dict(color=line_color, width=line_width),
                name=name, legendgroup=name, hoverinfo='skip'
            ))

    fig.update_layout(
        title=title, height=590, dragmode='zoom',
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title='x', yaxis_title='y',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0)
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig

def fig_contact(model,shot,width,height,scale):
    import plotly.graph_objects as go
    X=model['nodes']+scale*shot['U'].reshape(-1,2); master=model['block_top']; master=master[np.argsort(X[master,0])]; D={q['slave']:q for q in shot['diag']}; fig=go.Figure()
    fig.add_trace(go.Scatter(x=X[master,0],y=X[master,1],mode='lines+markers',name='Master: blocco',line={'color':'red','width':3}))
    free=[n for n in model['ring_contact'] if not D[n]['active']]; act=[n for n in model['ring_contact'] if D[n]['active']]
    if free: fig.add_trace(go.Scatter(x=X[free,0],y=X[free,1],mode='lines+markers',name='Slave liberi',line={'color':'#455a64'}))
    if act: fig.add_trace(go.Scatter(x=X[act,0],y=X[act,1],mode='markers',name='Contatto attivo',marker={'color':'orange','size':11,'symbol':'diamond'}))
    for n in model['ring_contact']:
        q=D[n]; P=(1-q['xi'])*X[q['a']]+q['xi']*X[q['b']]
        fig.add_trace(go.Scatter(x=[X[n,0],P[0]],y=[X[n,1],P[1]],mode='lines',showlegend=False,line={'dash':'dot','color':'orange' if q['active'] else '#90caf9'},hovertext=f"gap={q['gap']:.4g}<br>p={q['pressure']:.4g}",hoverinfo='text'))
    fig.update_layout(title='Dettaglio contatto: nodi, proiezioni e active set',height=590,dragmode='zoom',xaxis_title='x',yaxis_title='y',margin={'l':20,'r':20,'t':50,'b':20})
    fig.update_xaxes(range=[0,width]); fig.update_yaxes(range=[-height,height],scaleanchor='x',scaleratio=1); return fig

def fig_gap(shot):
    import plotly.graph_objects as go
    i=np.arange(len(shot['diag'])); g=[q['gap'] for q in shot['diag']]; p=[q['pressure'] for q in shot['diag']]
    fig=go.Figure(); fig.add_bar(x=i,y=p,name='Pressione penalty',marker_color='orange'); fig.add_scatter(x=i,y=g,name='Gap',mode='lines+markers',yaxis='y2')
    fig.update_layout(title='Gap e pressione lungo i nodi slave',height=400,dragmode='zoom',yaxis={'title':'Pressione'},yaxis2={'title':'Gap','overlaying':'y','side':'right'},xaxis_title='Indice nodo slave',hovermode='x unified'); return fig

def fig_history(H):
    import plotly.graph_objects as go
    A=np.asarray(H,object); fig=go.Figure(go.Scatter(x=A[:,0].astype(float),y=A[:,1].astype(float),mode='lines+markers'))
    fig.update_layout(title='Curva forza-spostamento fino allo step selezionato',height=400,dragmode='zoom',xaxis_title='δ',yaxis_title='Reazione totale / profondità'); return fig

def csv_history(H):
    out='delta,reazione,contatti,iterazioni,convergenza,gap_minimo,pressione_massima\n'
    return out+''.join(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n' for r in H)

st.title('FEM 2D didattico: contatto semianello-blocco')
st.caption('Esecuzione incremento per incremento con visualizzazione interattiva dell’active set, del gap e della pressione penalty.')
with st.sidebar:
    st.header('Geometria'); ri=st.number_input('Raggio interno ri',.1,value=40.); re=st.number_input('Raggio esterno re',.2,value=60.); L=st.number_input('Semilarghezza L',1.,value=240.); h=st.number_input('Altezza h',1.,value=180.)
    st.header('Materiali'); E_ring=st.number_input('E semianello',1.,value=210000.); nu_ring=st.number_input('nu semianello',0.,.499,value=.3); E_block=st.number_input('E blocco',1.,value=30000.); nu_block=st.number_input('nu blocco',0.,.499,value=.2)
    st.header('Mesh'); h_contact=st.number_input('Elemento al contatto',.05,value=2.,step=.25); mesh_growth=st.slider('Crescita elementi',1.02,1.6,1.22,.01); hmax_factor=st.slider('h max / h contatto',2.,20.,8.,.5)
    st.header('Soluzione'); delta_max=st.number_input('Spostamento massimo',.0001,value=1.); n_steps=st.slider('Incrementi',1,50,12); penalty_factor=st.number_input('Fattore penalty',1.,value=50.); max_iter=st.slider('Iterazioni massime',5,100,35); tol=st.number_input('Tolleranza',value=1e-7,format='%.1e'); relax=st.slider('Rilassamento',.1,1.,.8,.05)
    st.header('Vista'); scale=st.slider('Scala deformata',0.,20.,1.,.5); zoom_w=st.number_input('Ampiezza zoom contatto',1.,value=30.); zoom_h=st.number_input('Semialtezza zoom',.1,value=8.)
if not(re>ri and L>re and h>0): st.error('Richiesto re > ri, L > re, h > 0'); st.stop()
Eeff=2/(1/E_ring+1/E_block); kp=penalty_factor*Eeff/h_contact
p=dict(ri=ri,re=re,L=L,h=h,E_ring=E_ring,nu_ring=nu_ring,E_block=E_block,nu_block=nu_block,h_contact=h_contact,mesh_growth=mesh_growth,hmax_factor=hmax_factor,delta_max=delta_max,n_steps=n_steps,kp=kp,max_iter=max_iter,tol=tol,relax=relax)
try: model=build_model(p)
except Exception as e: st.error(str(e)); st.stop()
config={'scrollZoom':True,'displaylogo':False}; a,b,c=st.columns(3); a.metric('Nodi',len(model['nodes'])); b.metric('Elementi',len(model['elems'])); c.metric('GDL',2*len(model['nodes']))
t1,t2,t3=st.tabs(['Geometria','Mesh zoomabile','Soluzione passo-passo'])
with t1: st.plotly_chart(fig_mesh(model,mesh=False,title='Geometria'),use_container_width=True,config=config)
with t2: st.plotly_chart(fig_mesh(model,title='Mesh interattiva'),use_container_width=True,config=config)
with t3:
    sig=signature(p); cols=st.columns(3)
    if cols[0].button('Inizializza / Reset',use_container_width=True): st.session_state.S=init_state(model,p); st.session_state.sig=sig
    disabled='S' not in st.session_state
    if cols[1].button('Esegui un incremento',type='primary',use_container_width=True,disabled=disabled) and st.session_state.get('sig')==sig: st.session_state.S=one_step(model,p,st.session_state.S)
    if cols[2].button('Esegui fino alla fine',use_container_width=True,disabled=disabled) and st.session_state.get('sig')==sig:
        while not st.session_state.S['finished']: st.session_state.S=one_step(model,p,st.session_state.S)
    if 'S' not in st.session_state: st.info('Inizializzare, poi eseguire un incremento alla volta.')
    elif st.session_state.get('sig')!=sig: st.warning('Parametri modificati: inizializzare nuovamente.')
    elif st.session_state.S['step']==0: st.plotly_chart(fig_mesh(model,title='Stato iniziale'),use_container_width=True,config=config)
    else:
        S = st.session_state.S
        if S['step'] == 1:
            k = 1
            st.caption('Incremento visualizzato: 1. Lo slider sarà disponibile dal secondo incremento.')
        else:
            k = st.slider(
                'Incremento da visualizzare',
                min_value=1,
                max_value=S['step'],
                value=S['step'],
                step=1,
            )
        shot = S['shots'][k-1]
        row = S['history'][k-1]
        m=st.columns(5); m[0].metric('Step',f'{k}/{n_steps}'); m[1].metric('δ',f"{shot['delta']:.5g}"); m[2].metric('Reazione',f"{shot['force']:.5g}"); m[3].metric('Contatti',row[2]); m[4].metric('Iterazioni',row[3])
        st.markdown('**Rombi arancioni:** contatti attivi. **Linee tratteggiate:** proiezione nodo-segmento. Un gap negativo attiva la molla penalty.')
        x,y=st.columns(2)
        with x: st.plotly_chart(fig_mesh(model,shot['U'],scale,True,f'Deformata step {k}, scala ×{scale:g}'),use_container_width=True,config=config)
        with y: st.plotly_chart(fig_contact(model,shot,zoom_w,zoom_h,scale),use_container_width=True,config=config)
        st.plotly_chart(fig_gap(shot),use_container_width=True,config=config); st.plotly_chart(fig_history(S['history'][:k]),use_container_width=True,config=config)
        with st.expander('Iterazioni interne dello step'): st.dataframe(shot['log'],use_container_width=True,hide_index=True)
        st.download_button('Scarica storia CSV',csv_history(S['history']),'storia_incrementale.csv','text/csv')
with st.expander('Uso dello zoom'): st.write('Rotella: zoom. Trascinamento: rettangolo di zoom. Doppio clic: reset. La barra Plotly consente pan, autoscale e salvataggio PNG.')
