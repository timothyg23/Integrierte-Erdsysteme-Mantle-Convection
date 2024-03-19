using PyPlot
using SparseArrays
using LinearAlgebra
using Printf
using TimerOutputs
using Random
using LoopVectorization

Random.seed!(2)

to = TimerOutput()


function Convection3DC()
    
    # Written by Paul Tackley April 2018 - March 2022

    # Infinite Prandtl number convection (e.g. mantle convection) in 3D
    # using a finite volume discretization (staggered grid)
    #  - velocities at cell faces
    #  - temperature at cell centres
    #  - streamfunction and vorticity (when used) at cell corners

    # Equations are nondimensionalised!

    # Boundary conditions are: 
    #  - free-slip on all boundaries
    #  - dT/dx=0 on sides, T=1 at bottom and T=0 at top

    # Constant viscosity only.
    # Uses an iterative MULTIGRID solver
    #  to solve the poloidal mass flux potential version of the equations
    #
    # ---------SET INPUT PARAMETERS HERE------------

    # number of grid points; nz=vertical. 
    # For variable_viscosity=false use a POWER OF 2 for each

    nx=64; ny=64; nz=32;   # #grid points nz=vertical. Good numbers are 16,32,64,128,256
    nsteps=1000;   # number of time steps to take (press control-C to stop earlier)

    Ra=1e6;   # Rayleigh number
    H=0;      # Internal heating rate (nondimensional)
    initial_temperature=0.5;   # Initial temperature (0 to 1)
    
    compositional_field=false;  # true or false
    B_composition=1.0;    # Compositional buoyancy parameter
    depth_composition=0.2;  # initial thickness of dense layer
    
    plot_residual=false;
    plot_graphs=true;   # graphs of heat flux, velocity and temperature
    
    # --------DO NOT EDIT ANYTHING BELOW THIS LINE------



    # ---------------------initialise fields

    d=1/nz; dsq=d^2;

    s=zeros(nx,ny,nz+1); w=zeros(size(s)); 
    T=zeros(nx,ny,nz);   
    C=zeros(nx,ny,nz);   
    vx=zeros(nx+1,ny,nz); vy=zeros(nx,ny+1,nz); vz=zeros(nx,ny,nz+1);

    f_top  = zeros(nsteps)
    f_bot  = zeros(nsteps)
    times  = zeros(nsteps)
    Tmean  = zeros(nsteps)
    Vrms   = zeros(nsteps)
    Vsurf  = zeros(nsteps)

    rhs=zeros(nx,ny,nz+1);

    dsdz = zeros(nx,ny,nz)

    T=initial_temperature .+ 0.1*(rand(nx,ny,nz).-0.5);
    if compositional_field
        C[:,:,1:nz*depth_composition]=1.0;
    end
    
    time = 0.
    
    for step=1:nsteps

    # ---------------------take timestep

      # constant viscosity: use poloidal mass flux potential

            #  first, right-hand side = Ra*T
            #    calculated at centres of z faces
            #  See Travis 1990 GRL

         # rhs=zeros(nx,ny,nz+1);
         @timeit to "Tc Cc" for k=2:nz
             for j=1:ny
                @turbo for i=1:nx        
                     Tc = 0.5*(T[i,j,k]+T[i,j,k-1]);
                     Cc = 0.5*(C[i,j,k]+C[i,j,k-1]);
                     rhs[i,j,k] = Ra*(Tc-B_composition*Cc);
                 end
             end
         end

         #    next, call poisson solver twice to find poloidal mass flux

         @timeit to "multigrid1" w = poisson_multigrid(rhs,w);
         @timeit to "multigrid2" s = poisson_multigrid(  w,s);
    

         #   calculate velocities from poloidal mass flx ptl.

         # @timeit to "dsdz" dsdz[:,:,1:nz] = (s[:,:,2:nz+1]-s[:,:,1:nz])/d;

        @timeit to "dsdz" for k=1:nz
            for j=1:ny
                @turbo for i=1:nx
                    dsdz[i,j,k] = (s[i,j,k+1] - s[i,j,k])/d
                end
            end
        end


        @timeit to "vx vy" for k=1:nz
            for j=1:ny
                jm=max(j-1,1); jp=min(j+1,ny);
                @turbo for i=1:nx
                    im=max(i-1,1); ip=min(i+1,nx);
                    vx[i,j,k]=(dsdz[i,j,k]-dsdz[im,j,k])/d;
                    vy[i,j,k]=(dsdz[i,j,k]-dsdz[i,jm,k])/d;
                    vz[i,j,k]=-(s[ip,j,k]+s[im,j,k]+s[i,jp,k]+s[i,jm,k]-4*s[i,j,k])/d^2;
                 end
             end
         end
        

        # determine stable time-step
        dt_diff=0.1*dsq; # diffusive

        vxmax = 0.0
        vymax = 0.0
        vzmax = 0.0
        @timeit to "vxmax" for k = 1:nz
            for j=1:ny
                @inbounds for i=1:nx
                    if abs(vx[i,j,k])>vxmax; vxmax=abs(vx[i,j,k]); end
                    if abs(vy[i,j,k])>vymax; vymax=abs(vy[i,j,k]); end
                    if abs(vz[i,j,k])>vzmax; vzmax=abs(vz[i,j,k]); end
                end
                # vmax=max(maximum(abs.(vx)),maximum(abs.(vz)))                
            end
        end
        vmax = max(vxmax,vymax,vzmax)

        # @timeit to "vxmax" vmax=max(maximum(abs.(vx)),maximum(abs.(vy)),maximum(abs.(vz)));

        dt_adv=0.5*d/vmax;
        dt=min(dt_diff,dt_adv);

        # time-step temperature
        #    - diffusion & internal heating

        del2Ttemp = del2T(T)

        # @timeit to "tempstep" for k = 1:nz
        #     for j=1:ny
        #         @turbo for i=1:nx
        #             T[i,j,k] += dt*(del2Ttemp[i,j,k] + H)
        #         end
        #     end
        # end

        @timeit to "tempstep" @inbounds T .= T .+ dt .* (del2Ttemp .+ H)
  
        #    - advection    
        T = donor_cell_advection(T,vx,vy,vz,dt);
        if compositional_field
            C = donor_cell_advection(C,vx,vy,vz,dt);
        end

        time += dt;
        
        # calculate heat flux
            

        for j = 1:ny
            @inbounds for i = 1:nx
                f_top[step] += T[i,j,nz]         #  sum(T[:,nz])/length(T[:,nz]) / (0.5*d)
                f_bot[step] += T[i,j,1]          # (1-sum(T[:,1])/length(T[:,1])) / (0.5*d)
                Vsurf[step] += vx[i,j,nz]*vx[i,j,nz]
            end
        end

        f_top[step] = f_top[step]/(nx*ny*0.5*d)
        f_bot[step] = (1-f_bot[step]/(nx*ny))/(0.5*d)

        # f_top[step] = sum(T[:,nz])/length(T[:,nz]) / (0.5*d)
        # f_bot[step] = (1-sum(T[:,1])/length(T[:,1])) / (0.5*d)
        times[step] = time
        Tmean[step] = sum(T)/length(T)
        Vsurf[step] = sqrt(Vsurf[step]/(nx*ny))

        for k = 1:nz
            for j = 1:ny
                @inbounds for i = 1:nx
                    Vrms[step] += vx[i,j,k]*vx[i,j,k]+vz[i,j,k]*vz[i,j,k]
                end
            end
        end
        # Vsurf[step] = sqrt(sum(vx[1:nx,nz].^2)/nx)
        Vrms[step] = sqrt(Vrms[step]/(nx*ny*nz))
        # Vrms[step]  = sqrt(sum(vx[1:nx,1:nz].^2+vz[1:nx,1:nz].^2)/(nx*nz))  


        # f_top[step] = sum(T[:,:,nz])/length(T[:,:,nz]) / (0.5*d);
        # f_bot[step] = (1-sum(T[:,:,1])/length(T[:,:,nz])) / (0.5*d);
        # times[step] = time;
        # Tmean[step] = sum(T)/length(T);
        # Vrms[step] = sqrt(sum(vx[1:nx,1:ny,1:nz].^2+vy[1:nx,1:ny,1:nz].^2+vz[1:nx,1:ny,1:nz].^2)/(nx*ny*nz));
        # Vsurf[step] = sqrt(sum(vx[1:nx,1:ny,nz].^2)/(nx*ny));
        

        if mod(step,100)==0
            @printf("Step %d. Heat flux: Top=%5.2f; Bottom=%5.2f. Mean T=%5.2f. Vrms=%5.2f; Vs=%5.2f\n", round(step),f_top[step],f_bot[step],Tmean[step],Vrms[step],Vsurf[step]);
        end
        # finally, plot
        


        nplots=1




        # if (plot_graphs && mod(step,100)==0)
        #     figure(1)
        #     clf()
        #     slice = reshape(T[:,div(ny,2),:],nx,nz);
        #     subplot(nplots,1,1); contourf(slice',20); title("Temperature");
        #     if plot_residual
        #         Tres=T;
        #         for k=1:nz
        #             Tres[:,:,k] = T[:,:,k] - sum(T[:,:,k])/length(T[:,:,k]);
        #         end
        #         # contourf()

        #         # contour3D(Tres,-0.1);
        #         # contour3D(Tres, 0.1);
        #     else
        #         # contour3D(T,0.4);
        #         # contour3D(T,0.6);
        #     end
        #     # axis([1 nx 1 ny 1 nz])
        #     title("Temperature");
        #     if compositional_field
        #         figure(2)

        #         # contour3D()

        #         contourf(C[:,:,1])

        #         # contour3D(C,0.5);
        #         # axis([1 nx 1 ny 1 nz])
        #         title("Composition");
        #     end
        # end
        
        # if (plot_graphs && mod(step,100)==0)
        #     figure(3)
        #     clf()
        #     subplot(4,1,1); plot(times[1:step],f_top[1:step],"b",times[1:step],f_bot[1:step],"r")
        #     xlabel("Time (nondimensional)"); ylabel("Heat flux (nondimensional)")
        #     # legend("Top","Bottom","Location","northwest")
        #     title("Heat flux")

        #     subplot(4,1,2); plot(times[1:step],Tmean[1:step])
        #     xlabel("Time (nondimensional)"); ylabel("mean temperature (nondimensional)")
        #     title("Temperature")

        #     subplot(4,1,3); plot(times[1:step],Vsurf[1:step],times[1:step],Vrms[1:step])
        #     xlabel("Time (nondimensional)"); ylabel("Velocity (nondimensional)")
        #     # legend("Surface","Volume average","Location","northwest")
        #     title("Velocity")

        #     subplot(4,1,4); plot(times[1:step],Vsurf[1:step]./Vrms[1:step])
        #     xlabel("Time (nondimensional)"); ylabel("Mobility (nondimensional)")
        #     title("Surface Mobility")
        # end
        
        # pause(0.01)

    end
end


#-----------------------------------------------------------------------

function del2T(T)
#   calculates del2 of temperature field assuming:
#    - zero flux side boundaries 
#    - T=0 at top
#    - T=1 at bottom

    (nx, ny, nz) = size(T);
    delsqT = zeros(size(T));
    dsq = 1.0/(nz-1)^2;
    
    for k=1:nz
        for j=1:ny
            jm=max(j-1,1); jp=min(j+1,ny);
            @inbounds for i=1:nx
                im=max(i-1,1); ip=min(i+1,nx);
        
                T_xm=T[im,j,k]; T_xp=T[ip,j,k]; 
                T_ym=T[i,jm,k]; T_yp=T[i,jp,k]; 
                if k==1   # near lower boundary where T=1
                    T_zm=2-T[i,j,k];
                else
                    T_zm=T[i,j,k-1];
                end
                if k==nz # near upper boundary where T=0
                    T_zp=-T[i,j,k];
                else
                    T_zp=T[i,j,k+1];
                end
            
                delsqT[i,j,k]=(T_xm+T_xp+T_ym+T_yp+T_zm+T_zp-6*T[i,j,k])/dsq;

            end
        end
    end
    
    return delsqT
end

#-----------------------------------------------------------------------

function donor_cell_advection(T,vx,vy,vz,dt)

    (nx, ny, nz) = size(T);
    Tnew = zeros(size(T));
    d = 1.0/(nz-1);

    for k=1:nz
        for j=1:ny  
            @inbounds for i=1:nx        
                # Donor cell advection (div(VT))
                if vx[i,j,k]>0
                    flux_xm = T[i-1,j,k]*vx[i,j,k];  
                else
                    flux_xm = T[i  ,j,k]*vx[i,j,k];  
                end
                if i<nx
                    if vx[i+1,j,k]>0
                        flux_xp = T[i  ,j,k]*vx[i+1,j,k];  
                    else
                        flux_xp = T[i+1,j,k]*vx[i+1,j,k];  
                    end
                else
                    flux_xp = 0;
                end

                
                if vy[i,j,k]>0
                    flux_ym = T[i,j-1,k]*vy[i,j,k];  
                else
                    flux_ym = T[i,j  ,k]*vy[i,j,k];  
                end
                if j<ny
                    if vy[i,j+1,k]>0
                        flux_yp = T[i,j  ,k]*vy[i,j+1,k];  
                    else
                        flux_yp = T[i,j+1,k]*vy[i,j+1,k];  
                    end
                else
                    flux_yp = 0;
                end

                if vz[i,j,k]>0
                    flux_zm = T[i,j,k-1]*vz[i,j,k];  
                else
                    flux_zm = T[i  ,j,k]*vz[i,j,k];  
                end
                if k<nz
                    if vz[i,j,k+1]>=0
                        flux_zp = T[i,j,k]*vz[i,j,k+1];  
                    else
                        flux_zp = T[i,j,k+1]*vz[i,j,k+1];  
                    end
                else
                    flux_zp = 0;
                end

                dT_dt = (flux_xm-flux_xp+flux_ym-flux_yp+flux_zm-flux_zp)/d;
                Tnew[i,j,k] = T[i,j,k] + dt*dT_dt;
        
            end
        end
    end

    return Tnew
 
end
    

#-----------------------------------------------------------------------
 
function poisson_multigrid(R,f)
    #function [f] = poisson_multigrid (R,fguess)
    # calls poisson_vee repeatedly until desired convergence is reached
    # R and f should be a power-of-two in horizontal direction and a 
    #  power-of-two plus one in the vertical direction (second dimension)

    @timeit to "rmean" Rmean=sum(abs.(R))/length(R);
    # residue = delsqzp(f)-R; resmean=sum(abs.(residue))/length(residue);
    @timeit to "calc_resmean" resmean = calc_resmean(f, R)

    while resmean>1e-3*Rmean
        
        @timeit to "poisson_vee" f=poisson_vee(R,f);
        # residue = delsqzp(f)-R; resmean=sum(abs.(residue))/length(residue);

        @timeit to "calc_resmean" resmean = calc_resmean(f, R)
    end
    return f

end

#--------------------------------------------------------------------

function poisson_vee(R,fguess)
    # function [f] = poisson_vee (R)
    # solve poisson equation in 2D using multigrid approach: del2(f)=R
    # Boundary conditions:
    # - sides are periodic
    # - top & bottom are zero
    # This routine calls itself recursively to find the correction to the next
    # coarsest level

    s=size(R); nx=s[1];ny=s[2];nz=s[3]; #surf(R");title("Rhs");pause
    h=1/(nz-1);     # grid spacing, assuming vertical dimension=1

    f=fguess;
    alpha=0.9;

    if nz>5

        for iter=1:2               #  smoothing steps
            f=relaxdelsq(f,R,alpha);
        end
        
        # residue = delsqzp(f)-R;    # residue to coarse grid
        residue = calc_res(f, R)
        rescoarse = (residue[1:2:nx-1,1:2:ny-1,1:2:nz]+residue[2:2:nx,1:2:ny-1,1:2:nz]+residue[1:2:nx-1,2:2:ny,1:2:nz]+residue[2:2:nx,2:2:ny,1:2:nz])/4; 
        coarseguess=zeros(size(rescoarse));
        
        coarsecorrection = poisson_vee(rescoarse,coarseguess); # recursive call of this routine
        
        finecorrection = zeros(size(R));             # linear interpolation to fine grid
        for k=1:2:nz
            kc=div(k+1,2);
            for j=1:2:ny-1
                jc=div(j+1,2); jcm=max(1,jc-1); jcp=min(div(ny,2),jc+1);
                for i=1:2:nx-1
                    ic=div(i+1,2); icm=max(1,ic-1); icp=min(div(nx,2),ic+1);
                    finecorrection[i,j,k]     = (9*coarsecorrection[ic,jc,kc] + 3*coarsecorrection[icm,jc,kc] + 3*coarsecorrection[ic,jcm,kc] + coarsecorrection[icm,jcm,kc])/16;
                    finecorrection[i+1,j,k]   = (9*coarsecorrection[ic,jc,kc] + 3*coarsecorrection[icp,jc,kc] + 3*coarsecorrection[ic,jcm,kc] + coarsecorrection[icp,jcm,kc])/16;
                    finecorrection[i,j+1,k]   = (9*coarsecorrection[ic,jc,kc] + 3*coarsecorrection[icm,jc,kc] + 3*coarsecorrection[ic,jcp,kc] + coarsecorrection[icm,jcp,kc])/16;
                    finecorrection[i+1,j+1,k] = (9*coarsecorrection[ic,jc,kc] + 3*coarsecorrection[icp,jc,kc] + 3*coarsecorrection[ic,jcp,kc] + coarsecorrection[icp,jcp,kc])/16;
                end
            end
        end

        for k=2:2:nz-1
            for j=1:ny
                @turbo for i=1:nx
                    finecorrection[i,j,k]=0.5*(finecorrection[i,j,k-1]+finecorrection[i,j,k+1]);
                end
            end
        end
        
        for k=1:nz
            for j=1:ny
                @turbo for i=1:nx
                    f[i,j,k] -= finecorrection[i,j,k]
                end
            end
        end

        for iter=1:2               # more smoothing steps
           f=relaxdelsq(f,R,alpha);
        end
        
    else
        
        for iter=1:50               # reach solution using multiple iters
           f=relaxdelsq(f,R,alpha);
        end

    end

    return f

end

#--------------------------------------------------------------------

function relaxdelsq(a,rhs,alpha)
    # Gauss-Seidel relax delsquared assuming sides, top&bottom are zero
    s=size(a); nx=s[1]; ny=s[2]; nz=s[3]; ohsq=(nz-1)^2; afac=alpha/(6*ohsq);
    rmean=0;
    for irb = 0:1 #red-black
        for i=1:nx
            im=max(1,i-1); ip=min(nx,i+1);
            for j=1:ny
                jm=max(1,j-1); jp=min(ny,j+1);
                for k = 2+mod(i+j+irb,2):2:nz-1   # don"t relax top & bottom boundaries
                    res = (a[ip,j,k]+a[im,j,k]+a[i,jm,k]+a[i,jp,k]+a[i,j,k-1]+a[i,j,k+1]-6*a[i,j,k])*ohsq-rhs[i,j,k];
                    a[i,j,k] += afac*res;
                    rmean += abs(res);
                end
            end
        end
    end
    rmean=rmean/(nx*ny*(nz-1));
    return a
end

#--------------------------------------------------------------------

function delsqzp(a)
    # delsquared assuming sides, top&bottom are zero
    s=size(a); nx=s[1]; ny=s[2]; nz=s[3]; ohsq=(nz-1)^2;
    d=zeros(s);
    for k = 2:nz-1
        for j = 1:ny
            jm=max(1,j-1); jp=min(ny,j+1);
            for i = 1:nx
                im=max(1,i-1); ip=min(nx,i+1);
                d[i,j,k]=(a[ip,j,k]+a[im,j,k]+a[i,jm,k]+a[i,jp,k]+a[i,j,k-1]+a[i,j,k+1]-6*a[i,j,k])*ohsq;
            end
        end
    end
    return d
end

#--------------------------------------------------------------------


function calc_res(a, R)
    # delsquared assuming sides, top&bottom are zero
    s=size(a); nx=s[1]; ny=s[2]; nz=s[3]; ohsq=(nz-1)^2;
    d=zeros(s);
    for k = 2:nz-1
        for j = 1:ny
            jm=max(1,j-1); jp=min(ny,j+1);
            for i = 1:nx
                im=max(1,i-1); ip=min(nx,i+1);
                d[i,j,k]=(a[ip,j,k]+a[im,j,k]+a[i,jm,k]+a[i,jp,k]+a[i,j,k-1]+a[i,j,k+1]-6*a[i,j,k])*ohsq - R[i,j,k];
            end
        end
    end
    return d
end


function calc_resmean(a,R)
    # delsquared assuming sides, top&bottom are zero
    s=size(a); nx=s[1]; ny=s[2]; nz=s[3]; ohsq=(nz-1)^2;
    resmean = 0.0
    for k = 2:nz-1
        for j = 1:ny
            jm=max(1,j-1); jp=min(ny,j+1);
            @turbo for i = 1:nx
                im=max(1,i-1); 
                ip=min(nx,i+1);
                resmean += abs((a[ip,j,k]+a[im,j,k]+a[i,jm,k]+a[i,jp,k]+a[i,j,k-1]+a[i,j,k+1]-6*a[i,j,k])*ohsq - R[i,j,k])
            end
        end
    end
    return resmean/length(R)
end

@timeit to "main" Convection3DC()

show(to)