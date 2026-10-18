"""Independent scalar UTD reference; time dependence exp(+i omega t).
Primary equations: Kouyoumjian & Pathak (1974); Nethercote et al. (2019)
https://arxiv.org/abs/1905.10562 . No source code copied from wave_tracer.
"""
import json, math
import numpy as np
from scipy.special import erfc, erfcx, jv
from scipy.optimize import minimize_scalar
import mpmath as mp

PI=math.pi
EP=np.exp(1j*PI/4)

def transition(x):
    if x<0: raise ValueError('x must be nonnegative')
    return np.sqrt(PI*x)*EP*erfcx(EP*np.sqrt(x))

def transition_portable(x):
    """Double-precision browser-portable reference: series then continued fraction."""
    if x == 0:return 0j
    z=EP*np.sqrt(x)
    if x<4:
        term=1+0j; series=term
        for m in range(1,49):
            term*=(-1j*x)/m
            series+=term/(2*m+1)
        return np.sqrt(PI)*z*np.exp(1j*x)*(1-2*z/np.sqrt(PI)*series)
    cf=0j
    for m in range(48,0,-1):cf=(m/2)/(z+cf)
    return z/(z+cf)

def cot_transition(beta,n,kL,sgn):
    """Stable cot((pi+sgn beta)/(2n))*F(kL a_sgn(beta)).
    At a geometrical boundary, use the symmetric value zero and GO weight 1/2.
    Never set the whole wedge coefficient to zero at such a boundary.
    """
    N=math.floor((beta+sgn*PI)/(2*PI*n)+.5)
    delta=beta-(2*PI*n*N-sgn*PI)
    if abs(delta)<1e-14: return 0j
    t=delta/2
    h=delta/(2*n)
    # |sin(t)| / tan(h), evaluated without 0/0 at the shadow boundary.
    ratio=np.sign(delta)*n*np.sinc(t/PI)*np.cos(h)/np.sinc(h/PI)
    x=2*kL*np.sin(t)**2
    return sgn*np.sqrt(2*PI*kL)*EP*ratio*erfcx(EP*np.sqrt(x))

def coefficient(k,n,phi_i,phi_o,L,sin_beta=1.,boundary='soft'):
    a=sum(cot_transition(phi_o-phi_i,n,k*L,s) for s in (1,-1))
    b=sum(cot_transition(phi_o+phi_i,n,k*L,s) for s in (1,-1))
    R=-1 if boundary=='soft' else 1
    return -np.exp(-1j*PI/4)/(2*n*np.sqrt(2*PI*k)*sin_beta)*(a+R*b)

def step(x):
    return .5 if abs(x)<1e-13 else float(x>0)

def go_plane(k,r,n,pi,po,boundary='soft'):
    R=-1 if boundary=='soft' else 1
    d=po-pi; s=po+pi
    return (np.exp(1j*k*r*np.cos(d))*step(PI-abs(d))
            +R*np.exp(1j*k*r*np.cos(s))*step(PI-s)
            +R*np.exp(1j*k*r*np.cos(s-2*n*PI))*step(s-(2*n*PI-PI)))

def total_plane(k,r,n,pi,po,boundary='soft'):
    return go_plane(k,r,n,pi,po,boundary)+coefficient(k,n,pi,po,r,boundary=boundary)*np.exp(-1j*k*r)/np.sqrt(r)

def sommerfeld(k,r,pi,po,boundary='soft'):
    """Independent exact plane-wave half-plane solution (n=2)."""
    R=-1 if boundary=='soft' else 1
    def f(beta):
        t=np.sqrt(2*k*r)*np.cos(beta/2)
        return .5*np.exp(1j*k*r*np.cos(beta))*erfc(-EP*t)
    return f(po-pi)+R*f(po+pi)

def exact_wedge_series(kr,n,phi_i,phi_o,boundary='soft'):
    """Convergent independent solution, Nethercote et al. Appendix A, conjugated."""
    order=np.arange(1,int(n*(kr+40+20*kr**(1/3))))/n
    terms=(4/n)*np.exp(.5j*PI*order)*jv(order,kr)
    if boundary=='soft':
        return np.sum(terms*np.sin(order*phi_i)*np.sin(order*phi_o))
    return (2/n)*jv(0,kr)+np.sum(terms*np.cos(order*phi_i)*np.cos(order*phi_o))

def fermat(src,dst,a,b):
    src=np.asarray(src,float); dst=np.asarray(dst,float)
    a=np.asarray(a,float);b=np.asarray(b,float)
    length=np.linalg.norm(b-a);e=(b-a)/length
    sa=np.dot(src-a,e);da=np.dot(dst-a,e)
    rs=np.linalg.norm(src-a-sa*e);rd=np.linalg.norm(dst-a-da*e)
    if rs+rd<1e-14:return None
    t=(rd*sa+rs*da)/(rs+rd)
    return None if t<0 or t>length else a+t*e

def fermat_plane(direction,dst,a,b):
    direction=np.asarray(direction,float); direction=direction/np.linalg.norm(direction)
    dst=np.asarray(dst,float); a=np.asarray(a,float); b=np.asarray(b,float)
    length=np.linalg.norm(b-a); e=(b-a)/length
    longitudinal=np.dot(dst-a,e)
    rho=np.linalg.norm(dst-a-longitudinal*e)
    c=np.dot(direction,e); sb=np.sqrt(max(0.,1-c*c))
    if sb<1e-3:return None
    t=longitudinal-rho*c/sb
    return None if t<0 or t>length else a+t*e

def run():
    mp.mp.dps=70
    ftests=[]
    for x in [0,1e-16,1e-10,1e-6,.01,.1,1,5.999,6,10,100,1e4]:
        xm=mp.mpf(x);zm=mp.exp(1j*mp.pi/4)*mp.sqrt(xm)
        ref=mp.sqrt(mp.pi*xm)*mp.exp(1j*mp.pi/4)*mp.exp(1j*xm)*mp.erfc(zm)
        val=transition(x)
        ftests.append({'x':x,'absolute_error':float(abs(val-complex(ref)))})
    portable_error=max(abs(transition_portable(float(x))-transition(float(x))) for x in np.r_[0,np.geomspace(1e-16,1e8,1001)])
    half=[]
    for boundary in ('soft','hard'):
        errors=[]
        for kr in [.1,1,10,100,1000]:
            for pi in [.15,.6,1.2,2.5]:
                angles=list(np.linspace(0,2*PI,81))
                angles += [v+d for v in (PI+pi,PI-pi) for d in (-1e-8,-1e-12,0,1e-12,1e-8)]
                for po in angles:
                    if not 0<=po<=2*PI:continue
                    got=total_plane(kr,1,2,pi,po,boundary)
                    ref=sommerfeld(kr,1,pi,po,boundary)
                    errors.append(abs(got-ref))
        half.append({'boundary':boundary,'max_complex_field_error':float(max(errors)),'cases':len(errors)})
    wedges=[]
    for n in (1.25,1.5,1.75,2.):
        for kr in (10,50,100,500):
            errors=[]
            for boundary in ('soft','hard'):
                for po in np.linspace(.01,n*PI-.01,41):
                    exact=exact_wedge_series(kr,n,.6,po,boundary)
                    approx=total_plane(kr,1,n,.6,po,boundary)
                    errors.append(abs(approx-exact))
            wedges.append({'n':n,'kr':kr,'source_angle':.6,'cases':len(errors),'max_absolute_field_error':float(max(errors))})
    limits=[]
    for n in (1.25,1.5,1.75,2.):
        pi=.6
        for boundary in ('soft','hard'):
            for po in (PI-pi,PI+pi,2*n*PI-PI-pi):
                if not 0<po<n*PI:continue
                exact=total_plane(100,1,n,pi,po,boundary)
                side=[total_plane(100,1,n,pi,po+d,boundary) for d in (-1e-9,1e-9)]
                limits.append({'n':n,'boundary':boundary,'phi_o':po,'left_right_gap':float(abs(side[0]-side[1])),'center_vs_side_mean':float(abs(exact-sum(side)/2))})
    flat=[]
    for bc in ('soft','hard'):
        vals=[abs(coefficient(10,1,pi,po,1,boundary=bc)) for pi in (.2,.8,1.5,2.8) for po in (.1,.7,1.4,2.9)]
        flat.append({'boundary':bc,'max_flat_plane_diffraction_coefficient':float(max(vals))})
    rng=np.random.default_rng(32)
    geo=[]
    for j in range(20):
        a=np.array([0.,0.,-10.]);b=np.array([0.,0.,10.])
        src=rng.normal(size=3)*2;dst=rng.normal(size=3)*2
        p=fermat(src,dst,a,b)
        e=(b-a)/np.linalg.norm(b-a)
        opt=minimize_scalar(lambda t:np.linalg.norm(src-(a+t*e))+np.linalg.norm(dst-(a+t*e)),bounds=(0,20),method='bounded',options={'xatol':1e-12})
        stationarity=abs(np.dot((p-src)/np.linalg.norm(p-src)+(p-dst)/np.linalg.norm(p-dst),e))
        geo.append({'stationarity':float(stationarity),'point_vs_minimizer':float(abs(np.dot(p-a,e)-opt.x))})
    plane=[]
    for j in range(20):
        a=np.array([0.,0.,-10.]);b=np.array([0.,0.,10.])
        direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        dst=rng.normal(size=3)*2
        p=fermat_plane(direction,dst,a,b)
        if p is None:continue
        e=(b-a)/np.linalg.norm(b-a)
        residual=abs(np.dot(direction+(p-dst)/np.linalg.norm(p-dst),e))
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        shift=rng.normal(size=3)
        pr=fermat_plane(rotation@direction,rotation@dst+shift,rotation@a+shift,rotation@b+shift)
        plane.append({'stationarity':float(residual),'rotation_translation_error':float(np.linalg.norm(pr-(rotation@p+shift)))})
    # The source header's 75 r^4 differs from the correct asymptotic coefficient 105.
    asym=[]
    for x in (6,10,100):
        r=1/(2*x)
        old=1+1j*r-3*r*r-15j*r**3+75*r**4
        corrected=1+1j*r-3*r*r-15j*r**3+105*r**4
        asym.append({'x':x,'native_truncated_series_error':float(abs(old-transition(x))),'corrected_truncated_series_error':float(abs(corrected-transition(x)))})
    out={'status':'reference equations verified; renderer and Gaussian bundle coupling not validated','time_convention':'exp(+i omega t), outgoing exp(-ik r); conjugate consistently for exp(-i omega t)','sources':['https://arxiv.org/abs/1905.10562','https://ntrs.nasa.gov/api/citations/19740020594/downloads/19740020594.pdf','https://ssteinberg.xyz/2025/08/28/generalizing_the_path_integral/paper.pdf'],'transition_function':ftests,'portable_transition_max_error':float(portable_error),'half_plane_against_exact_sommerfeld':half,'canonical_wedges_against_exact_eigenfunction_series':wedges,'uniform_boundary_limits':limits,'flat_plane_no_edge':flat,'fermat_geometry':{'cases':len(geo),'max_stationarity_residual':max(x['stationarity'] for x in geo),'max_point_error_vs_numeric_minimizer':max(x['point_vs_minimizer'] for x in geo)},'plane_incidence_fermat':{'cases':len(plane),'max_stationarity':max(v['stationarity'] for v in plane),'max_rigid_transform_error':max(v['rotation_translation_error'] for v in plane)},'large_x_series_comparison':asym,'not_validated':['finite-edge endpoint diffraction','general geometry beyond tested canonical convex wedges','Gaussian region-mode weighting','arbitrary mesh multiple-edge interactions','browser renderer']}
    print(json.dumps(out,indent=2))
    assert max(w['max_absolute_field_error'] for w in wedges)<.004
    assert portable_error<1e-10
    assert max(x['absolute_error'] for x in ftests)<1e-12
    assert max(x['max_complex_field_error'] for x in half)<1e-9
    assert max(x['left_right_gap'] for x in limits)<1e-5
    assert max(x['center_vs_side_mean'] for x in limits)<1e-9
    assert max(x['max_flat_plane_diffraction_coefficient'] for x in flat)<1e-12
    assert out['fermat_geometry']['max_stationarity_residual']<1e-12

if __name__=='__main__':run()
