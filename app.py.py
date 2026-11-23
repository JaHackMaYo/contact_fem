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
# Grafici
# -----------------------------
def plot_model(model, U=None, field=None, title='', scale=1.0, mesh=True):
    X=model['nodes'].copy() if U is None else model['nodes']+scale*U.reshape(-1,2)
    fig,ax=plt.subplots(figsize=(9,5.6))
    polys=X[model['elems']]
    if field is None:
        colors=np.where(model['mats']==0,0.2,0.85)
        pc=PolyCollection(polys,array=colors,cmap='coolwarm',edgecolors='k' if mesh else 'none',linewidths=.25)
    else:
        pc=PolyCollection(polys,array=np.asarray(field),cmap='turbo',edgecolors='k' if mesh else 'none',linewidths=.15)
    ax.add_collection(pc); ax.autoscale(); ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(alpha=.15)
    if field is not None: fig.colorbar(pc,ax=ax,shrink=.8)
    fig.tight_layout(); return fig


def csv_history(hist):
    out='delta,reazione_totale_per_unita_profondita,contatti_attivi,iterazioni,convergenza\n'
    for r in hist: out += f'{float(r[0]):.12g},{float(r[1]):.12g},{int(r[2])},{int(r[3])},{bool(r[4])}\n'
    return out

# -----------------------------
# UI Streamlit
# -----------------------------
st.title('FEM 2D da zero: contatto semianello–blocco')
st.caption('Plane strain, materiali elastici lineari, CST, contatto frictionless penalty, mezzo modello per simmetria.')

with st.sidebar:
    st.header('Geometria')
    ri=st.number_input('Raggio interno ri',0.1,value=40.0)
    re=st.number_input('Raggio esterno re',0.2,value=60.0)
    L=st.number_input('Semilarghezza blocco L',1.0,value=240.0)
    h=st.number_input('Altezza blocco h',1.0,value=180.0)
    st.header('Materiali')
    E_ring=st.number_input('E semianello',1.0,value=210000.0)
    nu_ring=st.number_input('nu semianello',0.0,0.499,value=0.30,format='%.3f')
    E_block=st.number_input('E blocco',1.0,value=30000.0)
    nu_block=st.number_input('nu blocco',0.0,0.499,value=0.20,format='%.3f')
    st.header('Mesh guidata dal contatto')
    h_contact=st.number_input('Dimensione elemento al contatto',0.05,value=2.0,step=0.25,
                              help='Lunghezza obiettivo sul bordo esterno del semianello e sulla superficie del blocco.')
    mesh_growth=st.slider('Fattore crescita elementi',1.02,1.60,1.22,.01)
    hmax_factor=st.slider('Dimensione massima / dimensione contatto',2.0,20.0,8.0,.5)
    st.header('Soluzione')
    delta_max=st.number_input('Spostamento massimo',0.0001,value=1.0,format='%.4f')
    n_steps=st.slider('Incrementi',1,50,12)
    penalty_factor=st.number_input('Fattore penalty α',1.0,value=50.0)
    max_iter=st.slider('Iterazioni per incremento',5,100,35)
    tol=st.number_input('Tolleranza relativa',value=1e-7,format='%.1e')
    relax=st.slider('Rilassamento',0.1,1.0,0.8,.05)

if not (re>ri and L>re and h>0): st.error('Richiesto: re > ri, L > re e h > 0.'); st.stop()
# scala penalty coerente con la dimensione locale del contatto
Eeff=2.0/(1.0/E_ring+1.0/E_block); kp=penalty_factor*Eeff/h_contact
p=dict(ri=ri,re=re,L=L,h=h,E_ring=E_ring,nu_ring=nu_ring,E_block=E_block,nu_block=nu_block,
       h_contact=h_contact,mesh_growth=mesh_growth,hmax_factor=hmax_factor,
       delta_max=delta_max,n_steps=n_steps,kp=kp,max_iter=max_iter,tol=tol,relax=relax)

try: model=build_model(p)
except Exception as ex: st.error(str(ex)); st.stop()

c1,c2,c3,c4=st.columns(4)
c1.metric('Nodi',len(model['nodes'])); c2.metric('Elementi CST',len(model['elems']))
c3.metric('GDL',2*len(model['nodes'])); c4.metric('Penalty kc',f'{kp:.3g}')
mi=model['mesh_info']; st.caption(f"Mesh derivata: semianello {mi['nt']} settori × {mi['nr']} strati; blocco {mi['nx']} × {mi['ny']} celle. Passi x: {mi['min_dx']:.3g}–{mi['max_dx']:.3g}; passi y: {mi['min_dy']:.3g}–{mi['max_dy']:.3g}.")

tab1,tab2,tab3=st.tabs(['Geometria','Mesh','Soluzione e risultati'])
with tab1:
    st.pyplot(plot_model(model,mesh=False,title='Geometria del mezzo modello'))
    st.info('Il centro del semianello è a y = re; il punto esterno inferiore è inizialmente a y = 0. Lo spostamento è imposto sul bordo radiale orizzontale a destra.')
with tab2:
    st.pyplot(plot_model(model,mesh=True,title='Mesh strutturata parametrica'))
    st.write('La dimensione scelta al contatto genera la discretizzazione dell’arco esterno. Le stesse ascisse sono trasferite alla superficie del blocco; gli elementi crescono poi gradualmente verso destra, in profondità e attraverso lo spessore del semianello.')
with tab3:
    if st.button('Risolvi',type='primary',use_container_width=True):
        prog=st.progress(0,text='Preparazione')
        try:
            U,hist,cache,cdata=solve_model(model,p,prog)
            prog.empty(); vm,syy=element_results(model,U,cache)
            st.session_state['solution']=(p,model,U,hist,vm,syy,cdata)
        except np.linalg.LinAlgError:
            prog.empty(); st.error('Sistema singolare o mal condizionato. Aumentare la mesh, ridurre il penalty o controllare i vincoli.')
        except Exception as ex:
            prog.empty(); st.exception(ex)
    if 'solution' in st.session_state:
        ps,ms,U,hist,vm,syy,cdata=st.session_state['solution']
        if any(ps[k]!=p[k] for k in ['ri','re','L','h','h_contact','mesh_growth','hmax_factor','delta_max','kp']):
            st.warning('I parametri sono cambiati dopo l’ultima soluzione. Premere nuovamente Risolvi.')
        last=hist[-1]; a,b,c,d=st.columns(4)
        a.metric('δ raggiunto',f'{float(last[0]):.5g}')
        b.metric('Reazione totale',f'{float(last[1]):.5g}')
        c.metric('Contatti attivi',int(last[2])); d.metric('Convergenza','Sì' if bool(last[4]) else 'No')
        if not bool(last[4]): st.error('Ultimo incremento non convergente: i risultati vanno considerati non validati.')
        scale=st.slider('Scala deformata',0.0,20.0,1.0,.5)
        left,right=st.columns(2)
        with left: st.pyplot(plot_model(ms,U,title='Deformata',scale=scale,mesh=True))
        with right: st.pyplot(plot_model(ms,U,vm,title='Tensione equivalente 2D indicativa',scale=scale,mesh=False))
        fig,ax=plt.subplots(); ax.plot(hist[:,0].astype(float),hist[:,1].astype(float),'o-')
        ax.set_xlabel('Spostamento imposto δ'); ax.set_ylabel('Reazione totale / profondità'); ax.grid(True,alpha=.3); fig.tight_layout(); st.pyplot(fig)
        st.download_button('Scarica curva forza–spostamento CSV',csv_history(hist),'forza_spostamento.csv','text/csv')
        buf=io.StringIO(); buf.write('node,x,y,ux,uy\n')
        for i,(xy,u) in enumerate(zip(ms['nodes'],U.reshape(-1,2))): buf.write(f'{i},{xy[0]:.12g},{xy[1]:.12g},{u[0]:.12g},{u[1]:.12g}\n')
        st.download_button('Scarica spostamenti nodali CSV',buf.getvalue(),'spostamenti_nodali.csv','text/csv')

with st.expander('Assunzioni e limiti numerici'):
    st.markdown('''
- Piccole deformazioni e deformazione piana; profondità unitaria.
- Contatto unilaterale senza attrito, discretizzato nodo-segmento con penalty.
- Triangoli lineari CST: utile per validare l’algoritmo, ma richiede uno studio di convergenza della mesh.
- La normale di contatto è aggiornata, mentre la tangente trascura i termini geometrici della variazione della normale.
- Il valore della reazione è raddoppiato perché si usa metà geometria.
- La tensione equivalente mostrata è una misura 2D indicativa; la verifica finale in plane strain dovrà includere esplicitamente σzz.
''')
