"""

Extension to GQL of 2D equations with
two-scale Kolmogorov-forcing found in:

Tobias and Marston (2017) Physics of Fluids 29, 111111

"""

import numpy as np

from dedalus import public as de
from dedalus.extras import flow_tools 

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")
logger = logging.getLogger(__name__)

from mpi4py import MPI

import time
import yaml

with open(r'input.yaml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)


Nx,Ny   = (params['nx'],params['ny'])            # modes
Lx,Ly   = (2.0*np.pi,2.0*np.pi)                  # dimensions

cutoff  = params['lambda']

A1      = float(params['a1'])
A4      = float(params['a2'])

nu      = float(params['nu'])
beta    = float(params['beta'])

x_basis = de.Fourier('x', Nx, interval=(0.0,Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0.0,Ly), dealias=3/2)  
domain  = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# zelo/zehi: relative vorticity, silo/sihi: streamfunction
gql = de.IVP(domain, variables=['zelo','zehi','silo','sihi','u','v'],time='t')

gql.parameters['nu']            = nu                            # kinematic viscosity
gql.parameters['beta']          = beta                          # Coriolis paramater

gql.substitutions['J(A,B)']     = "dx(A)*dy(B) - dy(A)*dx(B)"   # Jacobian
gql.substitutions['L(A)']       = "dx(dx(A)) + dy(dy(A))"       # Laplacian

# two-scale Kolmogorov forcing
y                                   = domain.grid(1)
ncc                                 = domain.new_field()
ncc.meta['x']['constant']           = True
ncc['g']                            = A1*np.cos(y) + 4.0*A4*np.cos(4.0*y)
gql.parameters['F']                 = ncc

# low modes
gql.add_equation("dt(zelo) - nu*L(zelo) - beta*dx(silo) = - J(silo,zelo) - J(sihi,zehi) + F", condition = f"(abs(nx) <= {cutoff}) and (ny != 0)") 
gql.add_equation("L(silo) - zelo = 0", condition = f"(abs(nx) <= {cutoff}) and (ny != 0)")

gql.add_equation("zelo = 0", condition = f"(abs(nx) > {cutoff}) and (ny != 0)")
gql.add_equation("silo = 0", condition = f"(abs(nx) > {cutoff}) and (ny != 0)")

gql.add_equation("zelo = 0", condition = "(ny == 0)")
gql.add_equation("silo = 0", condition = "(ny == 0)")

# high modes
gql.add_equation("dt(zehi) - nu*L(zehi) - beta*dx(sihi) = - J(sihi,zelo) - J(silo,zehi) + F", condition = f"(abs(nx) > {cutoff}) and (ny != 0)") 
gql.add_equation("L(sihi) - zehi = 0", condition = f"(abs(nx) > {cutoff}) and (ny != 0)")

gql.add_equation("zehi = 0", condition = f"(abs(nx) <= {cutoff}) and (ny != 0)")
gql.add_equation("sihi = 0", condition = f"(abs(nx) <= {cutoff}) and (ny != 0)")

gql.add_equation("zehi = 0", condition = "(ny == 0)")
gql.add_equation("sihi = 0", condition = "(ny == 0)")

gql.add_equation("u + dy(silo + sihi) = 0")
gql.add_equation("v - dx(silo + sihi) = 0")

ts = de.timesteppers.RK222
solver = gql.build_solver(ts)
logger.info('Solver built')

# initial condition
x = domain.grid(0)
y = domain.grid(1)
zelo = solver.state['zelo']
zehi = solver.state['zehi']

zelo['g'] = np.random.rand(*zelo['g'].shape)
zehi['g'] = np.random.rand(*zehi['g'].shape)

# timestepping and analysis
dt = 0.01
stop_sim_time = 10000
solver.stop_sim_time = stop_sim_time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'v'))

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=100, mode='append')
snapshots.add_system(solver.state)

# profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.01, max_writes=10000, mode='append')
# profiles.add_task("integ(u, 'x')", name='uz')

series = solver.evaluator.add_file_handler('series', sim_dt=1e-1, max_writes=100)
series.add_task("integ(integ(u**2 + v**2,'x'),'y')", name='ke')
series.add_task("integ(integ((zelo + zehi)**2,'x'),'y')", name='enstrophy')

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(zelo + zehi)**2.0", name='enstrophy')
flow.add_property("u**2.0 + v**2.0", name='ke')

# adding on solver loop from dedalus/ivp/RB2D
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Average enstrophy = %f' %flow.grid_average('enstrophy'))
            logger.info('Average kinetic energy = %f' %flow.grid_average('ke'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
