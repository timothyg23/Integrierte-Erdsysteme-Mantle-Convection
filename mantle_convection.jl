using PyPlot
using SparseArrays
using LinearAlgebra
using Printf
using Random
using TimerOutputs
using LoopVectorization

to = TimerOutput()

Random.seed!(2)

function Convection2DC()

    # Written by Paul Tackley October 2012 - March 2022

    # Infinite Prandtl number convection (e.g. mantle convection)
    # using a finite volume discretization (staggered grid)
    #  - velocities at cell faces
    #  - temperature at cell centres
    #  - streamfunction and vorticity (when used) at cell corners

    # Equations are nondimensionalised!

    # Boundary conditions are: 
    #  - free-slip on all boundaries
    #  - dT/dx=0 on sides, T=1 at bottom and T=0 at top

    # If variable_viscosity=true then it uses a DIRECT solver
    # If variable_viscosity=false then it uses an iterative MULTIGRID solver
    #  to solve the streamfunction-vorticity version of the equations
    # 
    # Vrms is the root-mean-square (average) velocity over the whole domain
    # Vs is the average surface velocity

    # ---------SET INPUT PARAMETERS HERE------------

    # number of grid points; nz=vertical. 
    # For variable_viscosity=false use a POWER OF 2 for each

    nx=128
    nz=32   #grid points nz=vertical. Good numbers are 16,32,64,128,256
    nsteps=100   # number of time steps to take (press control-C to stop earlier)

    Ra=1e6   # Rayleigh number
    H=0.0      # Internal heating rate (nondimensional)
    initial_temperature=0.5   # Initial temperature (0 to 1)
    
    variable_viscosity=true  # true or false switch on/off variable viscosity
    viscosity_contrast_temperature=1e6  # factor variation with T. 1=no variation
    viscosity_contrast_depth=1e3      # factor variation with depth. 1=no variation
    yielding=false                # plastic yielding of the lithosphere
    yield_stress=1e2               # nondimensional yield stress

    compositional_field=false  # true or false
    B_composition=1.0      # Compositional buoyancy parameter
    depth_composition=0.2  # initial thickness of dense layer

    draw_colorbar=true    # If graphics doesn"t work properly, try setting this to false
    plot_streamfunction=false
    
    plot_graphs=true   # graphs of heat flux, velocity and temperature
    
    # --------DO NOT EDIT ANYTHING BELOW THIS LINE------



    # ---------------------initialise fields

    A_viscosity=log(viscosity_contrast_temperature)
    B_viscosity=log(viscosity_contrast_depth)

    yielding=yielding & variable_viscosity

    d=1/nz
    dsq=d^2

    s=zeros(nx+1,nz+1)
    w=zeros(size(s)) 
    T=zeros(nx,nz)   
    C=zeros(nx,nz)   
    vx=zeros(size(s))
    vz=zeros(size(s))
    eta=zeros(nx,nz)

    viscosity = zeros(nx,nz)
    rhs = zeros(nx,nz)

    f_top = zeros(nsteps)
    f_bot = zeros(nsteps)
    times = zeros(nsteps)
    Tmean = zeros(nsteps)
    Vrms  = zeros(nsteps) 
    Vsurf = zeros(nsteps)

    T=initial_temperature .+ 0.1*(rand(nx,nz).-0.5)
    if compositional_field
        C[:,1:nz*depth_composition]=1.0
    end
    
    time = 0.
    
    for step=1:nsteps

    # ---------------------take timestep

        if variable_viscosity

            # calculate viscosity field
            @timeit to "getvisc" for j=1:nz
                depth=(0.5+nz-j)*d;
                for i=1:nx
                    viscosity[i,j] = exp(-A_viscosity*(T[i,j]-0.5)+B_viscosity*(depth-0.5))
                end
            end

            # calculate rhs at vz points (=-Ra*T)
            @timeit to "rhs" rhs[1:nx,2:nz] = -Ra * 0.5*((T[1:nx,1:nz-1]+T[1:nx,2:nz]) - B_composition*(C[1:nx,1:nz-1]+C[1:nx,2:nz]))

            # call direct solver        
            @timeit to "solve" (vx, vz, p) = direct_solve_Stokes_sparse(viscosity,rhs,d)

            if yielding
                viscosity0 = viscosity;
                viscosity_last = viscosity;
                for i=1:2
                    edot = calculate_edot_2inv(vx,vz)
                    viscosity = min(viscosity0,yield_stress./(2*edot))
                    change = (viscosity-viscosity_last)./viscosity_last;
                    change_rms = sqrt(sum(sum(change.*change))/(nx*nz))              
                    viscosity_last = viscosity;
                    if change_rms<1.e-2
                        break
                    end
                    (vx, vz, p) = direct_solve_Stokes_sparse(viscosity,rhs,d)
                end
                stress = 2*viscosity.*edot;
            end

        else  # not variable viscosity: use stream-function
            # calculate streamfunction from dT/dx

            #  first, right-hand side = Ra*(dT/dx)
            #    calculated at cell corners

            rhs=zeros(nx+1,nz+1)
            for i=2:nx
                for j=2:nz
                    dT_dx=0.5*(T[i,j]+T[i,j-1]-T[i-1,j]-T[i-1,j-1])/d
                    dC_dx=0.5*(C[i,j]+C[i,j-1]-C[i-1,j]-C[i-1,j-1])/d
                    rhs[i,j] = -Ra*(dT_dx-B_composition*dC_dx)
                end
            end

            #    next, call poisson solver twice to find stream-function

            @timeit to "multigrid1" w = poisson_multigrid(rhs,w)
            @timeit to "multigrid2" s = poisson_multigrid(  w,s)

            #   calculate velocities from streamfunction

            for i=1:nx
                ip=1+mod(i,nx)
                for j=1:nz
                    vx[i,j]=-(s[i,j+1]-s[i,j])/d
                    vz[i,j]= (s[ip,j] -s[i,j])/d
                end
            end

        end

        # determine stable time-step
        dt_diff=0.1*dsq; # diffusive

        vxmax = 0.0
        vzmax = 0.0

        for j=1:nz
            for i=1:nx
                if abs(vx[i,j])>vxmax; vxmax=abs(vx[i,j]); end
                if abs(vz[i,j])>vzmax; vzmax=abs(vz[i,j]); end

                # vmax=max(maximum(abs.(vx)),maximum(abs.(vz)))                
            end
        end

        vmax = max(vxmax, vzmax)

        # @timeit to "vmax" vmax=max(maximum(abs.(vx)),maximum(abs.(vz)))
        


        dt_adv=0.5*d/vmax
        dt=min(dt_diff,dt_adv)

        # time-step temperature
        #    - diffusion & internal heating
        
        del2Ttemp = del2T(T)

        for j = 1:nz
            for i = 1:nx
                T[i,j] += dt*(del2Ttemp[i,j] + H) 
            end
        end

        #    - advection    
        T = donor_cell_advection(T,vx,vz,dt)
        C = donor_cell_advection(C,vx,vz,dt)

        time += dt
        
        # calculate heat flux

        for i = 1:nx
            f_top[step] += T[i,nz]         #  sum(T[:,nz])/length(T[:,nz]) / (0.5*d)
            f_bot[step] += T[i,1]          # (1-sum(T[:,1])/length(T[:,1])) / (0.5*d)

            Vsurf[step] += vx[i,nz]^2

        end

        f_top[step] = f_top[step]/(nx*0.5*d)
        f_bot[step] = (1-f_bot[step]/nx)/(0.5*d)

        # f_top[step] = sum(T[:,nz])/length(T[:,nz]) / (0.5*d)
        # f_bot[step] = (1-sum(T[:,1])/length(T[:,1])) / (0.5*d)
        times[step] = time
        Tmean[step] = sum(T)/length(T)
        Vsurf[step] = sqrt(Vsurf[step]/nx)


        for j = 1:nz
            for i = 1:nx
                Vrms[step] += vx[i,j]*vx[i,j]+vz[i,j]*vz[i,j]
            end
        end
        # Vsurf[step] = sqrt(sum(vx[1:nx,nz].^2)/nx)
        Vrms[step] = sqrt(Vrms[step]/(nx*nz))
        # Vrms[step]  = sqrt(sum(vx[1:nx,1:nz].^2+vz[1:nx,1:nz].^2)/(nx*nz))  


        if mod(step,10000)==0
            @printf("Step %d. Heat flux: Top=%5.2f; Bottom=%5.2f. Mean T=%5.2f. Vrms=%5.2f; Vs=%5.2f\n", round(step),f_top[step],f_bot[step],Tmean[step],Vrms[step],Vsurf[step])
        end


        # finally, plot
        nplots=1
        if compositional_field
            nplots=2
        end
        if (variable_viscosity || plot_streamfunction) 
            nplots=nplots+1
        end


        if (plot_graphs && mod(step,100)==0)
            figure(1)
            clf()
            subplot(nplots,1,1)
            contourf(T',20)
            title("Temperature")
            if draw_colorbar
                colorbar()
            end
            quiver(vx[1:nx,1:nz]', vz[1:nx,1:nz]')
            if compositional_field
                subplot(nplots,1,2)
                contourf(C')
                title("Composition")
                quiver(vx[1:nx,1:nz]', vz[1:nx,1:nz]')
            end
            if variable_viscosity
                subplot(nplots,1,nplots)
                contourf(log10.(viscosity'),20)
                title("log10(Viscosity)")
            else
                if plot_streamfunction
                    subplot(nplots,1,nplots)
                    contourf(s')
                    title("Stream-function")
                end
            end
            if draw_colorbar
                colorbar()
            end
        end
        
        # if (plot_graphs && mod(step,100)==0)
        #     figure(2)
        #     clf()
        #     subplot(4,1,1)
        #     plot(times[1:step],f_top[1:step],"b",times[1:step],f_bot[1:step],"r")
        #     xlabel("Time (nondimensional)")
        #     ylabel("Heat flux (nondimensional)")
        #     # legend("Top","Bottom","Location","northwest")
        #     title("Heat flux")

        #     subplot(4,1,2)
        #     plot(times[1:step],Tmean[1:step])
        #     xlabel("Time (nondimensional)")
        #     ylabel("mean temperature (nondimensional)")
        #     title("Temperature")

        #     subplot(4,1,3)
        #     plot(times[1:step],Vsurf[1:step],times[1:step],Vrms[1:step])
        #     xlabel("Time (nondimensional)")
        #     ylabel("Velocity (nondimensional)")
        #     # legend("Surface","Volume average","Location","northwest")
        #     title("Velocity")

        #     subplot(4,1,4)
        #     plot(times[1:step],Vsurf[1:step]./Vrms[1:step])
        #     xlabel("Time (nondimensional)")
        #     ylabel("Mobility (nondimensional)")
        #     title("Surface Mobility")
        # end
        
        # sleep(0.01)

    end
end


#-----------------------------------------------------------------------

function del2T(T)
    #   calculates del2 of temperature field assuming:
    #    - zero flux side boundaries 
    #    - T=0 at top
    #    - T=1 at bottom

    (nx, nz) = size(T)
    delsqT = zeros(size(T))
    dsq = 1.0/(nz-1)^2
    
    for i=1:nx
        im=max(i-1,1)
        ip=min(i+1,nx)
        
        for j=1:nz
                    
            T_xm=T[im,j]
            T_xp=T[ip,j] 
            if j==1   # near lower boundary where T=1
                T_zm=2-T[i,j]
            else
                T_zm=T[i,j-1]
            end
            if j==nz # near upper boundary where T=0
                T_zp=-T[i,j]
            else
                T_zp=T[i,j+1]
            end
            
            delsqT[i,j]=(T_xm+T_xp+T_zm+T_zp-4*T[i,j])/dsq

        end
    end
    
    return delsqT
end

#-----------------------------------------------------------------------

function donor_cell_advection(T,vx,vz,dt)

    (nx, nz) = size(T)
    Tnew = zeros(size(T))
    d = 1.0/(nz-1)

    for j=1:nz
        for i=1:nx    
            # Donor cell advection (div(VT))
            if vx[i,j]>0
                flux_xm = T[i-1,j]*vx[i,j]  
            else
                flux_xm = T[i  ,j]*vx[i,j]  
            end
            if i<nx
                if vx[i+1,j]>0
                    flux_xp = T[i  ,j]*vx[i+1,j]  
                else
                    flux_xp = T[i+1,j]*vx[i+1,j]  
                end
            else
                flux_xp = 0
            end
    
            if vz[i,j]>0
                flux_zm = T[i,j-1]*vz[i,j]  
            else
                flux_zm = T[i  ,j]*vz[i,j]  
            end
            if j<nz
                if vz[i,j+1]>=0
                    flux_zp = T[i,j]*vz[i,j+1]
                else
                    flux_zp = T[i,j+1]*vz[i,j+1]
                end
            else
                flux_zp = 0
            end

            dT_dt = (flux_xm-flux_xp+flux_zm-flux_zp)/d
            Tnew[i,j] = T[i,j] + dt*dT_dt
        
        end
    end
    return Tnew
end
    
#-----------------------------------------------------------------------

function direct_solve_Stokes_sparse(eta,rhs,h)

    # direct solver for 2D variable viscosity Stokes flow
    # sxx,x + szx,z - p,x = 0
    # szz,z + sxz,x - p,z = rhs        
    # (dimensional rhs=density*g; nondimensionally, rhs=-Ra*T)

    # 2*(eta_xp*(u_xp-u0)/h-eta_xm*(u0-u_xm)/h)/h +
    #   (eta_zp*(u_zp-u0-w_xpzp-w_xmzp)/h-eta_zm*(u0-u_zm-w_xpzm-w_xmzm)/h)/h=0

    # Boundary conditions:
    #  shear stress=0 -> put etaxz=0 on the boundaries
    #  impermeable -> no action needed

    # rhs should have dimensions of the number of cells

    (nx,nz)=size(rhs)
    oh=1/h
    ohsq=1/h^2

    f=zeros(nx*nz*3,1)
    idx=3
    idz=nx*3

    n_non0=(nx*nz*(11+11+4)) # estimated number of non-zero matrix entries
    r=zeros(Int64, n_non0)
    c=zeros(Int64, n_non0)
    s=zeros(Float64, n_non0)
    

    vx = zeros(nx+1,nz+1)
    vz = zeros(nx+1,nz+1)
    p  = zeros(nx+1,nz+1)

    n=0 # number of non-zero point

    for iz=1:nz
        for ix=1:nx
            icell=ix-1+(iz-1)*nx
            ieqx=icell*3+1; vxc=ieqx
            ieqz=ieqx+1;    vzc=ieqz
            ieqc=ieqx+2;     pc=ieqc

            # print(icell, "\n")
            
            # viscosity points
            etaii_c  = eta[ix,iz]
            if ix>1
                etaii_xm = eta[ix-1,iz]
            end
            if iz>1
                etaii_zm = eta[ix,iz-1]
            end
            if (ix>1 && iz>1)
                etaxz_c  = (eta[ix,iz]*eta[ix-1,iz]*eta[ix,iz-1]*eta[ix-1,iz-1])^0.25
            else
                etaxz_c = 0
            end
            if (ix>1 && iz<nz) 
                etaxz_zp = (eta[ix,iz+1]*eta[ix-1,iz+1]*eta[ix,iz]*eta[ix-1,iz])^0.25
            else
                etaxz_zp = 0;
            end
            if (ix<nx && iz>1)   
                etaxz_xp  = (eta[ix+1,iz]*eta[ix,iz]*eta[ix+1,iz-1]*eta[ix,iz-1])^0.25
            else
                etaxz_xp = 0;
            end
            
            xmom_zero_eta = (etaii_c==0 && etaii_xm==0 && etaxz_c==0 && etaxz_zp==0)
            zmom_zero_eta = (etaii_c==0 && etaii_zm==0 && etazx_c==0 && etazx_xp==0)
            
            # x-momentum
           
            if ix>1 && !xmom_zero_eta
                n=n+1; r[n]=ieqx; c[n]=vxc    ; s[n]=-ohsq*(2*etaii_c+2*etaii_xm+etaxz_c+etaxz_zp)
                n=n+1; r[n]=ieqx; c[n]=vxc-idx; s[n]= 2*ohsq*etaii_xm
                n=n+1; r[n]=ieqx; c[n]=vzc    ; s[n]=-ohsq*etaxz_c
                n=n+1; r[n]=ieqx; c[n]=vzc-idx; s[n]= ohsq*etaxz_c
                n=n+1; r[n]=ieqx; c[n]=pc     ; s[n]=-oh
                n=n+1; r[n]=ieqx; c[n]=pc-idx ; s[n]= oh
                if (ix+1<=nx)
                    n=n+1; r[n]=ieqx; c[n]=vxc+idx; s[n]=2*ohsq*etaii_c;
                end
                if (iz+1<=nz)
                    n=n+1; r[n]=ieqx; c[n]=vxc+idz    ; s[n]= ohsq*etaxz_zp;
                    n=n+1; r[n]=ieqx; c[n]=vzc+idz    ; s[n]= ohsq*etaxz_zp;
                    n=n+1; r[n]=ieqx; c[n]=vzc+idz-idx; s[n]=-ohsq*etaxz_zp;
                end
                if (iz>1)
                    n=n+1; r[n]=ieqx; c[n]=vxc-idz; s[n]=ohsq*etaxz_c;
                end
                f[ieqx]             =   0;
            else
                n=n+1; r[n]=ieqx; c[n]=vxc; s[n]=1;
                f[ieqx]             =    0;
            end
            
            # z-momentum
            if iz>1 && !zmom_zero_eta
                n=n+1; r[n]=ieqz; c[n]=vzc    ; s[n]=-ohsq*(2*etaii_c+2*etaii_zm+etaxz_c+etaxz_xp)
                n=n+1; r[n]=ieqz; c[n]=vzc-idz; s[n]= 2*ohsq*etaii_zm;
                n=n+1; r[n]=ieqz; c[n]=vxc    ; s[n]=-ohsq*etaxz_c;
                n=n+1; r[n]=ieqz; c[n]=vxc-idz; s[n]= ohsq*etaxz_c;
                n=n+1; r[n]=ieqz; c[n]=pc     ; s[n]=-oh;
                n=n+1; r[n]=ieqz; c[n]=pc-idz ; s[n]= oh;
                if iz+1<=nz
                    n=n+1; r[n]=ieqz; c[n]=vzc+idz ; s[n]= 2*ohsq*etaii_c;
                end
                if ix+1<=nx
                    n=n+1; r[n]=ieqz; c[n]=vzc+idx    ; s[n]=  ohsq*etaxz_xp;
                    n=n+1; r[n]=ieqz; c[n]=vxc+idx    ; s[n]=  ohsq*etaxz_xp;
                    n=n+1; r[n]=ieqz; c[n]=vxc+idx-idz; s[n]= -ohsq*etaxz_xp;
                end
                if ix>1
                    n=n+1; r[n]=ieqz; c[n]=vzc-idx ; s[n]= ohsq*etaxz_c;
                end
                f[ieqz]             =   rhs[ix,iz]
            else
                n=n+1; r[n]=ieqz; c[n]=vzc ; s[n]=1;
                f[ieqz]             = 0;
            end
            
            # continuity
            if (ix==1 && iz==1) || (xmom_zero_eta && zmom_zero_eta)
                n=n+1; r[n]=ieqc; c[n]=pc; s[n]=1;
            else
                n=n+1; r[n]=ieqc; c[n]=vxc; s[n]=-oh
                n=n+1; r[n]=ieqc; c[n]=vzc; s[n]=-oh
                if ix+1<=nx
                    n=n+1; r[n]=ieqc; c[n]=vxc+idx; s[n]=oh;
                end
                if iz+1<=nz
                    n=n+1; r[n]=ieqc; c[n]=vzc+idz; s[n]=oh;
                end
            end
            f[ieqc]             =   0
            
        end
    end
    size(s)

    m=sparse(r[1:n],c[1:n],s[1:n])
    @timeit to "getsolution" solution = m\f
    for iz=1:nz
        for ix=1:nx
            i0=(ix-1)+(iz-1)*nx
            vx[ix,iz]=solution[i0*3+1] # x-velocity
            vz[ix,iz]=solution[i0*3+2] # z-velocity
            p[ix,iz]=solution[i0*3+3] # pressure
        end
    end
    vx[nx+1,:].=0   #impermeable right boundary
    vz[:,nz+1].=0   #impermeable top boundary
    p[:,:] .= p[:,:] .- sum(p[1:nx,1:nz])/length(p[1:nx,1:nz]) # zero mean pressure

    return (vx, vz, p)

end

#-----------------------------------------------------------------------
 
function poisson_multigrid(R,f)
    #function [f] = poisson_multigrid (R,fguess)
    # calls poisson_vee repeatedly until desired convergence is reached
    # R and f should be a power-of-two in horizontal direction and a 
    #  power-of-two plus one in the vertical direction (second dimension)

    # f=fstart
    Rmean = 0.0
    (nx,nz) = size(R)

    for j=1:nz
        for i = 1:nx
            Rmean += abs(R[i,j])
        end
    end
    Rmean = Rmean/length(R)

    # residue = delsqzp(f)-R
    # resmean=sum(abs.(residue))/length(residue)

    resmean = calc_resmean(f, R)


    while resmean>1e-3*Rmean
        @timeit to "poisson_vee" f=poisson_vee(R,f)

        # residue = delsqzp(f)-R
        # resmean1=sum(abs.(residue))/length(residue)

        @timeit to "resmean2" resmean = calc_resmean(f,R)

        # print(resmean1-resmean, "\n")
    end

    return f
end

#--------------------------------------------------------------------

function poisson_vee(R,f)
    # function [f] = poisson_vee (R)
    # solve poisson equation in 2D using multigrid approach: del2(f)=R
    # Boundary conditions:
    # - sides are periodic
    # - top & bottom are zero
    # This routine calls itself recursively to find the correction to the next
    # coarsest level

    (nx,nz)=size(R)
    #surf(R")title("Rhs")pause
    h=1.0/(nz-1)     # grid spacing, assuming vertical dimension=1

    alpha=0.9

    if nz>5

        for iter=1:2               #  smoothing steps
            f=relaxdelsq(f,R,alpha)
        end
        
        residue = delsqzp(f)-R;    # residue to coarse grid

        
        rescoarse = residue[1:2:nx,1:2:nz]
        


        coarseguess=zeros(size(rescoarse))
        coarsecorrection = poisson_vee(rescoarse,coarseguess) # recursive call of this routine
        
        finecorrection = zeros(size(R))             # linear interpolation to fine grid
        finecorrection[1:2:nx,1:2:nz] = coarsecorrection;

        for j=2:2:nz-1
            for i=1:nx
                finecorrection[i,j]=0.5*(finecorrection[i,j-1]+finecorrection[i,j+1])
            end
        end
        for j=1:nz
            for i=2:2:nx-1
                finecorrection[i,j]=0.5*(finecorrection[i-1,j]+finecorrection[i+1,j])
            end
        end
        for j=1:nz
            for i=1:nx
                f[i,j] = f[i,j] - finecorrection[i,j]/2    # add fine correction to solution
            end
        end
       
        
        for iter=1:2               # more smoothing steps
           f=relaxdelsq(f,R,alpha)
        end
    else
        for iter=1:50               # reach solution using multiple iters
           f=relaxdelsq(f,R,alpha)
        end
    end
    return f
end

#--------------------------------------------------------------------

function relaxdelsq(a,rhs,alpha)
    # Gauss-Seidel relax delsquared assuming sides, top&bottom are zero
    s=size(a)
    nx=s[1]
    nz=s[2]
    ohsq=(nz-1)^2
    afac=alpha/(4*ohsq)
    rmean=0;
    for irb = 0:1 #red-black
        for i=2:nx-1
            for j = 2+mod(i+irb,2):2:nz-1   # don"t relax top & bottom boundaries
                res = (a[i+1,j]+a[i-1,j]+a[i,j-1]+a[i,j+1]-4*a[i,j])*ohsq-rhs[i,j]
                a[i,j] += afac*res
                rmean += abs(res)
            end
        end
    end
    rmean=rmean/(nx*(nz-1))
    f=a

    return f
end

#--------------------------------------------------------------------

function delsqzp(a)
    # delsquared assuming sides, top&bottom are zero
    s=size(a)
    nx=s[1]
    nz=s[2]
    ohsq=(nz-1)^2
    d=zeros(s)
    for j = 2:nz-1
        for i = 2:nx-1
            d[i,j]=(a[i+1,j]+a[i-1,j]+a[i,j-1]+a[i,j+1]-4*a[i,j])*ohsq
        end
    end
    return d
end

function calc_resmean(a, R)
    # delsquared assuming sides, top&bottom are zero
    (nx,nz)=size(a)
    ohsq=(nz-1)^2

    resmean = 0.0
    for j = 2:nz-1
        for i = 2:nx-1
            resmean+= abs((a[i+1,j]+a[i-1,j]+a[i,j-1]+a[i,j+1]-4*a[i,j])*ohsq - R[i,j])
        end
    end
    return resmean / length(R)
end

#--------------------------------------------------------------------

function calculate_stress_2inv(vx,vz,eta)
    (nx,nz)=size(eta)
    oh=nz
    sxy=zeros(nx+1,nz+1)
    stress=zeros(s)
    for j = 2:nz
        for i = 2:nx
            etaxz  = (eta[i,j]*eta[i-1,j]*eta[i,j-1]*eta[i-1,j-1])^0.25
            sxy[i,j] = etaxz*(vx[i,j]-vx[i,j-1]+vz[i,j]-vz[i-1,j])*oh
        end
    end
    for j = 1:nz
        for i = 1:nx
            sxx=2*eta[i,j]*(vx[i+1,j]-vx[i,j])*oh
            szz=2*eta[i,j]*(vz[i,j+1]-vz[i,j])*oh
            sxyc=0.25*(sxy[i,j]+sxy[i+1,j]+sxy[i,j+1]+sxy[i+1,j+1])
            stress[i,j]=sqrt(0.5*(sxx^2+szz^2+1*sxyc^2))
        end
    end
    return stress
end

#--------------------------------------------------------------------

function calculate_edot_2inv(vx,vz)
    s=size(vx)
    nx=s[1]-1
    nz=s[2]
    oh=nz
    exy=zeros(nx+1,nz+1)
    edot=zeros(nx,nz)
    for j = 2:nz
        for i = 2:nx
            exy[i,j] = 0.5*(vx[i,j]-vx[i,j-1]+vz[i,j]-vz[i-1,j])*oh
        end
    end
    for j = 1:nz
        for i = 1:nx
            exx=(vx[i+1,j]-vx[i,j])*oh
            ezz=(vz[i,j+1]-vz[i,j])*oh
            exyc=0.25*(exy[i,j]+exy[i+1,j]+exy[i,j+1]+exy[i+1,j+1])
            edot[i,j]=sqrt(0.5*(exx^2+ezz^2+1*exyc^2))
        end
    end

    return edot
end

@timeit to "main" Convection2DC()

show(to)