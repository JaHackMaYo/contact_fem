import io
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

st.set_page_config(page_title='FEM 2D - Contatto semianello', layout='wide')


# =============================================================
# PARAMETRI DI DEFAULT DELL'INTERFACCIA
# Modificare qui i valori iniziali proposti nella sidebar.
# Usare un sistema di unita coerente, ad esempio N, mm, N/mm2.
# =============================================================

# Geometria
DEFAULT_RI = 40.0
DEFAULT_RE = 60.0
DEFAULT_L = 72.0
DEFAULT_H = 36.0

# Materiali elastici lineari isotropi
DEFAULT_E_RING = 210000.0
DEFAULT_NU_RING = 0.30
DEFAULT_E_BLOCK = 30000.0
DEFAULT_NU_BLOCK = 0.20

# Mesh
DEFAULT_H_CONTACT = 2.0
DEFAULT_MESH_GROWTH = 1.22
DEFAULT_HMAX_FACTOR = 8.0

# Soluzione incrementale e contatto penalty
DEFAULT_DELTA_MAX = 1.0
DEFAULT_N_STEPS = 12
DEFAULT_PENALTY_FACTOR = 50.0
DEFAULT_MAX_ITER = 35
DEFAULT_TOL = 1e-7
DEFAULT_RELAX = 0.80

# Visualizzazione
DEFAULT_DEFORMATION_SCALE = 1.0
DEFAULT_ZOOM_WIDTH = 30.0
DEFAULT_ZOOM_HEIGHT = 8.0

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

def mesh_block_from_ring(L, h, rnodes, rcontact, h_contact, growth, hmax_factor):
    """Mesh indipendente del blocco, raffinata solo dalla zona di contatto verso destra.

    Le ascisse dei nodi del semianello non vengono proiettate sul blocco: in questo
    modo si evita il secondo addensamento, numericamente inutile, vicino a x = re.
    Il contatto resta nodo-segmento e non richiede nodi contrapposti coincidenti.
    """
    hmax = h_contact * hmax_factor

    dx = graded_partition(L, h_contact, growth, hmax)
    xs = np.r_[0.0, np.cumsum(dx)]
    xs[-1] = L

    dy = graded_partition(h, h_contact, growth, hmax)
    ys = -np.r_[0.0, np.cumsum(dy)][::-1]
    ys[0] = -h
    ys[-1] = 0.0

    nx = len(xs) - 1
    ny = len(ys) - 1
    nodes = np.array([[x, y] for y in ys for x in xs])
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1)+i
            n1 = n0+1
            n2 = n0+nx+1
            n3 = n2+1
            elems += [[n0,n1,n3],[n0,n3,n2]] if (i+j)%2==0 else [[n0,n1,n2],[n1,n3,n2]]

    top = np.arange(ny*(nx+1), (ny+1)*(nx+1))
    bottom = np.arange(nx+1)
    sym = np.arange(0, (ny+1)*(nx+1), nx+1)
    return nodes, np.asarray(elems), top, bottom, sym, xs, ys

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


def slave_tributary_lengths(model, U):
    """Lunghezza tributaria di ogni nodo slave nella configurazione corrente.

    Il nodo sull'asse e il nodo terminale ricevono meta del segmento adiacente;
    un nodo interno riceve meta dei due segmenti adiacenti. Nel mezzo modello
    queste lunghezze sono riferite alla sola meta destra del contatto.
    """
    X = model['nodes'] + U.reshape(-1, 2)
    slave = model['ring_contact']
    w = np.zeros(len(slave))
    if len(slave) < 2:
        return {int(slave[0]): 0.0} if len(slave) else {}
    seg = np.linalg.norm(np.diff(X[slave], axis=0), axis=1)
    w[0] = 0.5 * seg[0]
    w[-1] = 0.5 * seg[-1]
    if len(slave) > 2:
        w[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return {int(n): float(wi) for n, wi in zip(slave, w)}

def contact_matrix_rhs(model, U, kp, active_prev=None):
    """Assembla un contatto penalty integrato sulla superficie slave.

    kp ha unita pressione/lunghezza; p_i = -kp*g_i e la forza nodale
    equivalente e p_i moltiplicata per la lunghezza tributaria del nodo.
    """
    ndof = 2*len(model['nodes'])
    Kc = np.zeros((ndof, ndof))
    rhs = np.zeros(ndof)
    active, data = [], []
    X0 = model['nodes']
    weights = slave_tributary_lengths(model, U)
    for sn, a, b, xi, n, gap in contact_candidates(model, U):
        if gap < 0.0:
            dofs = np.array([2*sn,2*sn+1,2*a,2*a+1,2*b,2*b+1])
            G = np.r_[n, -(1-xi)*n, -xi*n]
            g0 = float((X0[sn]-((1-xi)*X0[a]+xi*X0[b])) @ n)
            tributary = weights[sn]
            pressure = -kp*gap
            nodal_force = pressure*tributary
            Kc[np.ix_(dofs,dofs)] += kp*tributary*np.outer(G,G)
            rhs[dofs] += -kp*tributary*g0*G
            active.append((sn,a,b))
            data.append((sn,gap,pressure,xi,tributary,nodal_force))
    return Kc, rhs, tuple(active), data

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
    amap={q[0]:q for q in cdata}; diag=[]
    for sn,a,b,xi,n,gap in contact_candidates(model,U):
        info=amap.get(sn)
        diag.append({'slave':sn,'a':a,'b':b,'xi':xi,'gap':gap,
                     'pressure':info[2] if info else 0.0,
                     'tributary':info[4] if info else slave_tributary_lengths(model,U).get(sn,0.0),
                     'contact_force':info[5] if info else 0.0,
                     'active':sn in amap})
    row=[delta,force,sum(q['active'] for q in diag),it,converged,min(q['gap'] for q in diag),max(q['pressure'] for q in diag)]
    S['U']=U; S['step']=step; S['history'].append(row); S['shots'].append({'U':U.copy(),'delta':delta,'force':force,'diag':diag,'log':log,'converged':converged})
    S['finished']=step>=p['n_steps'] or not converged
    return S

def signature(p): return tuple((k,float(v)) for k,v in p.items())

def contact_pressure_results(model, shot):
    """Restituisce pressione media e distribuzione discreta nel mezzo modello."""
    X = model['nodes'] + shot['U'].reshape(-1, 2)
    D = sorted(shot['diag'], key=lambda q: X[q['slave'], 0])
    x = np.array([X[q['slave'],0] for q in D], float)
    gap = np.array([q['gap'] for q in D], float)
    pressure = np.array([q['pressure'] for q in D], float)
    active = np.array([q['active'] for q in D], bool)

    # Bordo di contatto per interpolazione lineare del passaggio g<0 -> g>=0.
    a = 0.0
    ids = np.where(active)[0]
    if len(ids):
        j = ids[-1]
        if j+1 < len(D) and gap[j] < 0.0 and gap[j+1] >= 0.0:
            den = gap[j+1]-gap[j]
            a = x[j] + (-gap[j]/den)*(x[j+1]-x[j]) if abs(den)>1e-15 else x[j]
        else:
            # Estremo tributario se tutti i nodi successivi non sono disponibili.
            a = x[j] + 0.5*D[j].get('tributary',0.0)

    total_width = 2.0*a
    mean_pressure = shot['force']/total_width if total_width > 0 else 0.0
    integrated_half_force = sum(q.get('contact_force',0.0) for q in D)
    balance_error = (2*integrated_half_force-shot['force'])/max(abs(shot['force']),1e-15)
    return dict(x=x, gap=gap, pressure=pressure, active=active,
                half_width=a, total_width=total_width,
                mean_pressure=mean_pressure,
                integrated_total_force=2*integrated_half_force,
                balance_error=balance_error)


def fig_pressure_distribution(model, shot):
    import plotly.graph_objects as go
    R = contact_pressure_results(model, shot)
    mask = R['active']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=R['x'][mask], y=R['pressure'][mask],
                             mode='lines+markers', name='Pressione locale penalty integrata',
                             line=dict(color='#d84315', width=3), marker=dict(size=8)))
    if R['total_width'] > 0:
        fig.add_trace(go.Scatter(x=[0,R['half_width']],
                                 y=[R['mean_pressure'],R['mean_pressure']],
                                 mode='lines', name='Pressione media equivalente',
                                 line=dict(color='#1565c0', width=3, dash='dash')))
    fig.update_layout(title='Distribuzione della pressione nella zona di contatto (mezzo modello)',
                      height=430, dragmode='zoom', hovermode='x unified',
                      xaxis_title='Coordinata x sulla superficie di contatto',
                      yaxis_title='Pressione di contatto',
                      margin=dict(l=20,r=20,t=55,b=20), legend=dict(orientation='h'))
    return fig

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
    fig=go.Figure(); fig.add_bar(x=i,y=p,name='Pressione locale penalty',marker_color='orange'); fig.add_scatter(x=i,y=g,name='Gap',mode='lines+markers',yaxis='y2')
    fig.update_layout(title='Gap e pressione locale lungo i nodi slave',height=400,dragmode='zoom',yaxis={'title':'Pressione'},yaxis2={'title':'Gap','overlaying':'y','side':'right'},xaxis_title='Indice nodo slave',hovermode='x unified'); return fig

def fig_history(H):
    import plotly.graph_objects as go
    A=np.asarray(H,object); fig=go.Figure(go.Scatter(x=A[:,0].astype(float),y=A[:,1].astype(float),mode='lines+markers'))
    fig.update_layout(title='Curva forza-spostamento fino allo step selezionato',height=400,dragmode='zoom',xaxis_title='δ',yaxis_title='Reazione totale / profondità'); return fig

def csv_history(H):
    out='delta,reazione,contatti,iterazioni,convergenza,gap_minimo,pressione_massima\n'
    return out+''.join(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]}\n' for r in H)

st.title('FEM 2D didattico: contatto semianello-blocco')
st.caption('Esecuzione incremento per incremento con visualizzazione interattiva dell’active set, del gap e della pressione penalty.')
# I widget sono in un form: la modifica dei campi non rigenera il modello.
# I nuovi valori diventano attivi solo premendo il pulsante Aggiorna modello.
with st.sidebar:
    with st.form('model_parameters_form'):
        st.header('Geometria')
        ri_input = st.number_input('Raggio interno ri', .1, value=DEFAULT_RI)
        re_input = st.number_input('Raggio esterno re', .2, value=DEFAULT_RE)
        L_input = st.number_input('Semilarghezza L', 1., value=DEFAULT_L)
        h_input = st.number_input('Altezza h', 1., value=DEFAULT_H)

        st.header('Materiali')
        E_ring_input = st.number_input('E semianello', 1., value=DEFAULT_E_RING)
        nu_ring_input = st.number_input('nu semianello', 0., .499, value=DEFAULT_NU_RING)
        E_block_input = st.number_input('E blocco', 1., value=DEFAULT_E_BLOCK)
        nu_block_input = st.number_input('nu blocco', 0., .499, value=DEFAULT_NU_BLOCK)

        st.header('Mesh')
        h_contact_input = st.number_input('Elemento al contatto', .05, value=DEFAULT_H_CONTACT, step=.25)
        mesh_growth_input = st.slider('Crescita elementi', 1.02, 1.6, DEFAULT_MESH_GROWTH, .01)
        hmax_factor_input = st.slider('h max / h contatto', 2., 20., DEFAULT_HMAX_FACTOR, .5)

        st.header('Soluzione')
        delta_max_input = st.number_input('Spostamento massimo', .0001, value=DEFAULT_DELTA_MAX)
        n_steps_input = st.slider('Incrementi', 1, 50, DEFAULT_N_STEPS)
        penalty_factor_input = st.number_input('Fattore penalty', 1., value=DEFAULT_PENALTY_FACTOR)
        max_iter_input = st.slider('Iterazioni massime', 5, 100, DEFAULT_MAX_ITER)
        tol_input = st.number_input('Tolleranza', value=DEFAULT_TOL, format='%.1e')
        relax_input = st.slider('Rilassamento', .1, 1., DEFAULT_RELAX, .05)

        update_model = st.form_submit_button('Aggiorna modello', type='primary', use_container_width=True)

    st.header('Vista')
    scale = st.slider('Scala deformata', 0., 20., DEFAULT_DEFORMATION_SCALE, .5)
    zoom_w = st.number_input('Ampiezza zoom contatto', 1., value=DEFAULT_ZOOM_WIDTH)
    zoom_h = st.number_input('Semialtezza zoom', .1, value=DEFAULT_ZOOM_HEIGHT)

if 'active_parameters' not in st.session_state:
    initial_Eeff = 2.0 / (1.0/DEFAULT_E_RING + 1.0/DEFAULT_E_BLOCK)
    st.session_state.active_parameters = dict(
        ri=DEFAULT_RI, re=DEFAULT_RE, L=DEFAULT_L, h=DEFAULT_H,
        E_ring=DEFAULT_E_RING, nu_ring=DEFAULT_NU_RING,
        E_block=DEFAULT_E_BLOCK, nu_block=DEFAULT_NU_BLOCK,
        h_contact=DEFAULT_H_CONTACT, mesh_growth=DEFAULT_MESH_GROWTH,
        hmax_factor=DEFAULT_HMAX_FACTOR, delta_max=DEFAULT_DELTA_MAX,
        n_steps=DEFAULT_N_STEPS,
        kp=DEFAULT_PENALTY_FACTOR*initial_Eeff/DEFAULT_H_CONTACT,
        max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, relax=DEFAULT_RELAX
    )

if update_model:
    if not (re_input > ri_input and L_input > re_input and h_input > 0):
        st.sidebar.error('Richiesto: re > ri, L > re e h > 0.')
    else:
        Eeff_input = 2.0 / (1.0/E_ring_input + 1.0/E_block_input)
        st.session_state.active_parameters = dict(
            ri=ri_input, re=re_input, L=L_input, h=h_input,
            E_ring=E_ring_input, nu_ring=nu_ring_input,
            E_block=E_block_input, nu_block=nu_block_input,
            h_contact=h_contact_input, mesh_growth=mesh_growth_input,
            hmax_factor=hmax_factor_input, delta_max=delta_max_input,
            n_steps=n_steps_input,
            kp=penalty_factor_input*Eeff_input/h_contact_input,
            max_iter=max_iter_input, tol=tol_input, relax=relax_input
        )
        # Una nuova geometria/materiale rende incompatibile l'eventuale soluzione precedente.
        st.session_state.pop('S', None)
        st.session_state.pop('sig', None)
        st.sidebar.success('Modello aggiornato.')

p = st.session_state.active_parameters
try:
    model = build_model(p)
except Exception as e:
    st.error(str(e))
    st.stop()

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
        m=st.columns(5); m[0].metric('Step',f"{k}/{p['n_steps']}"); m[1].metric('δ',f"{shot['delta']:.5g}"); m[2].metric('Reazione',f"{shot['force']:.5g}"); m[3].metric('Contatti',row[2]); m[4].metric('Iterazioni',row[3])
        pressure_result = contact_pressure_results(model, shot)
        pm = st.columns(4)
        pm[0].metric('Ampiezza contatto totale', f"{pressure_result['total_width']:.5g}")
        pm[1].metric('Pressione media', f"{pressure_result['mean_pressure']:.5g}")
        pm[2].metric('Forza integrata dal contatto', f"{pressure_result['integrated_total_force']:.5g}")
        pm[3].metric('Errore equilibrio contatto', f"{100*pressure_result['balance_error']:.3g}%")
        st.markdown('**Rombi arancioni:** contatti attivi. **Linee tratteggiate:** proiezione nodo-segmento. Un gap negativo attiva la molla penalty.')
        x,y=st.columns(2)
        with x: st.plotly_chart(fig_mesh(model,shot['U'],scale,True,f'Deformata step {k}, scala ×{scale:g}'),use_container_width=True,config=config)
        with y: st.plotly_chart(fig_contact(model,shot,zoom_w,zoom_h,scale),use_container_width=True,config=config)
        st.plotly_chart(fig_pressure_distribution(model, shot), use_container_width=True, config=config)
        st.caption('La curva locale deriva dalla formulazione penalty integrata con lunghezze tributarie. È la distribuzione discreta coerente con questo modello; non è una ricostruzione analitica di Hertz.')
        st.plotly_chart(fig_gap(shot),use_container_width=True,config=config)
        st.plotly_chart(fig_history(S['history'][:k]),use_container_width=True,config=config)
        with st.expander('Iterazioni interne dello step'): st.dataframe(shot['log'],use_container_width=True,hide_index=True)
        st.download_button('Scarica storia CSV',csv_history(S['history']),'storia_incrementale.csv','text/csv')
with st.expander('Uso dello zoom'): st.write('Rotella: zoom. Trascinamento: rettangolo di zoom. Doppio clic: reset. La barra Plotly consente pan, autoscale e salvataggio PNG.')
