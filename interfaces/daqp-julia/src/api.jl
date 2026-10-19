"""
# Example calls
    xstar, fval, exitflag, info = DyadDAQP.quadprog(H,f,A,bupper)
    xstar, fval, exitflag, info = DyadDAQP.quadprog(H,f,A,bupper,blower,sense;primal_start,dual_start

finds the solution `xstar` to the quadratic program

```
 minimize	   0.5 x' H x + f' x
subject to   blower <= A x <= bupper
```
If `bupper` and `blower` have more elements than rows of `A`, the first
elements are interpreted as simple bounds. For example:

```
    A = [7.0 8.0]
    blower = [-4.0; -5.0; -6.0]
    bupper = [ 1.0;  2.0;  3.0]
```
is interpreted as

```
        -4.0 <= x₁ <= 1.0
        -5.0 <= x₂ <= 2.0
    -6.0 <= 7 x₁ + 8 x₂ <= 3.0

```

# Input
* `H`           - cost matrix
* `f`           - cost vector
* `A`           - linear constraint matrix
* `bupper`       - upper bounds for constraints
* `blower`      - lower bounds for constraints (default: -Inf)
* `sense`       - constraint types, as a vector of Cints (default: 0). Example types:
  * `0 ` : inequality
  * `1 ` : active inequality (used as warm start)
  * `5 ` : equality
  * `8 ` : soft (allowed to be violated if necessary)
  * `16` : binary (either upper or lower bound should hold with equality)
* `primal_start` - Initial guess of primal iterate
* `dual_start`   - Initial guess of dual iterate

# Output
* `xstar`       - solution
* `fval`        - objective function value for `xstar`.
* `exitflag`    - flag from solver (>0 success, <0 failure)
* `info`        - tuple containing profiling information from the solver.

"""
function quadprog(H::Union{Matrix{Float64}, Cholesky},f::Vector{Float64},
        A::Matrix{Float64},bupper::Vector{Float64},blower::Vector{Float64}=Float64[],
        sense::Vector{Cint}=Cint[];A_rowmaj=false,settings=nothing,
        primal_start::Vector{Cdouble}=Cdouble[], dual_start::Vector{Cdouble}=Cdouble[])
    d = DyadDAQP.Model()
    !isnothing(settings) && DyadDAQP.settings(d,settings)
    exitflag,setup_time = DyadDAQP.setup(d,QPj(H,f,A,bupper,blower,sense;A_rowmaj);
        primal_start,dual_start,
        init_mask=DAQP_UPDATE_unconstrained|DAQP_UPDATE_eliminate)
    return DyadDAQP.solve(d;setup_time);
end

# XXX Just kept for test for now
function quadprog(qpj::QPj;settings=nothing)
    # Setup QP
    qp = QPc(qpj);

    # Setup output struct
    xstar = zeros(Float64,qp.n);
    lam= zeros(Float64,qp.m);
    result= Ref(DAQPResult(xstar,lam));
    ptr_settings = settings isa DAQPSettings ? Ref(settings) : Ptr{DyadDAQP.DAQPSettings}(C_NULL)


    ccall((:daqp_quadprog, DyadDAQP.libdaqp), Nothing,
          (Ref{DyadDAQP.DAQPResult},Ref{DyadDAQP.QPc},Ref{DyadDAQP.DAQPSettings}),
          result,Ref(qp),ptr_settings)

    info = (x = xstar, λ=lam, fval=result[].fval,
            exitflag=result[].exitflag,
            status = DyadDAQP.flag2status[result[].exitflag],
            solve_time = result[].solve_time,
            setup_time = result[].setup_time,
            iterations= result[].iter, nodes = result[].nodes)
    return xstar,result[].fval,result[].exitflag,info
end

"""
    xstar, fval, exitflag, info = DyadDAQP.linprog(f,A,bupper)
    xstar, fval, exitflag, info = DyadDAQP.linprog(f,A,bupper,blower,sense;primal_start,dual_start)

finds the solution `xstar` to the linear program

```
min_x	f' x
subject to
    blower[1:ms]	<= x[1:ms] <= bupper[1:ms]
    blower[ms+1:m]  <= A*x 	   <= bupper[ms+1:m],
```
where `m = length(bupper)` and `ms = m-size(A,2)`.

# Input
* `f`  			- linear term in objective function, n-vector
* `A`  			- constraint normals, (`(m-ms) x n`)-matrix
* `bupper` 		- upper bounds for constraints, `m`-vector
* `blower` 		- lower bounds for constraints, `m`-vector (default: -Inf)
* `sense` 		- constraint types,  `m`-vector of Cints (default: 0). Example types:
  * `0` : inequality
  * `5` : equality
* `primal_start` - Initial guess of primal iterate
* `dual_start`   - Initial guess of dual iterate

# Output
* `xstar` 		- solution provided by solver
* `fval` 		- objective function value for `xstar`.
* `exitflag` 	- flag from solver (>0 success, <0 failure)
* `info` 		- tuple containing profiling information from the solver.

"""
function linprog(f::Vector{Float64},
        A::Matrix{Float64},bupper::Vector{Float64},blower::Vector{Float64}=Float64[],
        sense::Vector{Cint}=Cint[];A_rowmaj=false, primal_start::Vector{Cdouble}=Cdouble[],
        dual_start::Vector{Cdouble}=Cdouble[])
    d = DyadDAQP.Model()
    exitflag,setup_time = DyadDAQP.setup(
        d, QPj(zeros(0,0),f,A,bupper,blower,sense;A_rowmaj);
        primal_start,dual_start,init_mask=DAQP_UPDATE_eliminate)
    return  DyadDAQP.solve(d;setup_time);
end
"""
# Example calls
    xstar, _, exitflag, info = DyadDAQP.avi(H,f,A,bupper)
    xstar, _, exitflag, info = DyadDAQP.avi(H,f,A,bupper,blower,sense;primal_start,dual_start)

finds the primal solution `xstar` and dual solution `lambda` to the affine variational inequality

Find `xstar` ∈ C ≜ {x : blower <= A x <= bupper}
```
⟨H*xstar f, y-xstar⟩ ≥ 0 for all y ∈ C
```
If `bupper` and `blower` have more elements than rows of `A`, the first
elements are interpreted as simple bounds. For example:

```
    A = [7.0 8.0]
    blower = [-4.0; -5.0; -6.0]
    bupper = [ 1.0;  2.0;  3.0]
```
is interpreted as

```
        -4.0 <= x₁ <= 1.0
        -5.0 <= x₂ <= 2.0
    -6.0 <= 7 x₁ + 8 x₂ <= 3.0

```

# Input
* `H`           - linear transform
* `f`           - offset
* `A`           - linear constraint matrix
* `bupper`       - upper bounds for constraints
* `blower`      - lower bounds for constraints (default: -Inf)
* `sense`       - constraint types, as a vector of Cints (default: 0). Example types:
  * `0 ` : inequality
  * `1 ` : active inequality (used as warm start)
  * `5 ` : equality
  * `8 ` : soft (allowed to be violated if necessary)
  * `16` : binary (either upper or lower bound should hold with equality)
* `primal_start` - Initial guess of primal iterate
* `dual_start`   - Initial guess of dual iterate

# Output
* `xstar`       - primal solution
* `_`           - placeholder
* `exitflag`    - flag from solver (>0 success, <0 failure)
* `info`        - tuple containing profiling information from the solver.

"""
function avi(H::Matrix{Float64},f::Vector{Float64},
        A::Matrix{Float64},bupper::Vector{Float64},blower::Vector{Float64}=Float64[],sense::Vector{Cint}=Cint[];A_rowmaj=false,settings=nothing, primal_start::Vector{Cdouble}=Cdouble[],
        dual_start::Vector{Cdouble}=Cdouble[])
    d = DyadDAQP.Model()
    !isnothing(settings) && DyadDAQP.settings(d,settings)
    exitflag,setup_time = DyadDAQP.setup(d,QPj(H,f,A,bupper,blower,sense;A_rowmaj,is_avi=true);
        primal_start,dual_start,init_mask=DAQP_UPDATE_eliminate)
    return  DyadDAQP.solve(d;setup_time);
end
"""
    d = DyadDAQP.Model()
creates an empty optimization model `d`.


The following functions acts on such models:

* `setup(d,H,f,A,bupper,blower,sense;primal_start, dual_start)`: setup a problem (see `DyadDAQP.quadprog` or DyadDAQP.avi for details)
* `solve(d)`: solve a populated model
* `update(d,H,f,A,bupper,blower,sense)`: update an existing model. Omitting
  `sense` reuses the previous state; supplying it overrides the warm start.
* `dict = DyadDAQP.settings(d)`: return a Dictionary with the current settings for the model `d`
* `settings(d,dict)`: update the settings for `d` with the Dictionary `dict`
"""
mutable struct Model
    work::Ptr{DyadDAQP.Workspace}
    qpj::QPj
    qpc::QPc
    qpc_ptr::Ptr{DyadDAQP.QPc}
    has_model::Bool
    has_solved::Bool
    x::Vector{Float64}
    λ::Vector{Float64}
    function Model()
        # Setup initial model
        work = Libc.calloc(1,sizeof(DyadDAQP.Workspace))
        daqp= new(Ptr{DyadDAQP.Workspace}(work))
        daqp.qpc_ptr = Libc.calloc(1,sizeof(DyadDAQP.QPc))
        ccall((:allocate_daqp_settings,DyadDAQP.libdaqp),Nothing,(Ptr{DyadDAQP.Workspace},),work)
        finalizer(DyadDAQP.delete!, daqp)
        daqp.has_model=false
        daqp.has_solved=false
        return daqp
    end
end


function delete!(daqp::DyadDAQP.Model)
    if(daqp.work != C_NULL)
        ccall((:free_daqp_workspace,DyadDAQP.libdaqp),Nothing,(Ptr{DyadDAQP.Workspace},),daqp.work)
        ccall((:free_daqp_ldp,DyadDAQP.libdaqp),Nothing,(Ptr{DyadDAQP.Workspace},),daqp.work)
        Libc.free(daqp.work);
        daqp.work = C_NULL
        Libc.free(daqp.qpc_ptr);
        daqp.qpc_ptr = C_NULL
    end
end

function setup(daqp::DyadDAQP.Model, qp::DyadDAQP.QPj;primal_start::Vector{Cdouble}=Cdouble[],
        dual_start::Vector{Cdouble}=Cdouble[],init_mask=Cint(0))
    # Own sense so warm starts do not modify the caller's QP.
    daqp.qpj = QPj(qp.n,qp.m,qp.ms,qp.H,qp.f,qp.A,qp.bupper,qp.blower,
                   copy(qp.sense),qp.break_points,qp.nh,qp.is_avi,qp.is_factorized)
    daqp.qpc = DyadDAQP.QPc(daqp.qpj)
    daqp.has_solved = false
    old_settings = settings(daqp); # in case setup fails
    unsafe_store!(daqp.qpc_ptr,daqp.qpc)
    setup_time = Ref{Cdouble}(0.0);

    # Set starting AS if dual or primal iterate provided
    if !isempty(dual_start)
        ccall((:daqp_dual_init_active,DyadDAQP.libdaqp),Nothing,
              (Ptr{DyadDAQP.QPc},Ptr{Cdouble}),daqp.qpc_ptr,dual_start)
    elseif !isempty(primal_start)
        ccall((:daqp_primal_init_active,DyadDAQP.libdaqp),Nothing,
              (Ptr{DyadDAQP.QPc},Ptr{Cdouble}),daqp.qpc_ptr,primal_start)
    end

    if(isempty(qp.H) && !isempty(qp.f))# LP
        # ensure there is no binary constraint
        @assert(!any((qp.sense.&BINARY).==BINARY),
                "DAQP requires the objective to be strictly convex to support binary variables")
    end

    # Handle AVI with other setup
    exitflag = ccall((:setup_daqp_main,DyadDAQP.libdaqp),Cint,
        (Ptr{DyadDAQP.QPc}, Ptr{DyadDAQP.Workspace}, Ptr{Cdouble},Cint),
        daqp.qpc_ptr, daqp.work, setup_time, Cint(init_mask))
    if(exitflag < 0)
        # XXX: if setup fails DAQP currently clears settings
        ccall((:allocate_daqp_settings,DyadDAQP.libdaqp),Nothing,(Ptr{DyadDAQP.Workspace},),daqp.work)
        settings(daqp,old_settings)
    else
        daqp.has_model = true
        daqp.x = Vector{Float64}(undef, qp.n)
        daqp.λ = Vector{Float64}(undef, qp.m)
        if !isempty(primal_start)
            ccall((:daqp_set_primal_start,DyadDAQP.libdaqp),Nothing,(Ptr{DyadDAQP.Workspace},Ptr{Cdouble}),
                  daqp.work,primal_start)
        end
    end

    return exitflag, setup_time[]
end

function setup(daqp::DyadDAQP.Model, H::Union{Matrix{Cdouble},Cholesky},f::Vector{Cdouble},
        A::Matrix{Cdouble},bupper::Vector{Cdouble},blower::Vector{Cdouble}=Cdouble[],
        sense::Vector{Cint}=Cint[];A_rowmaj=false,break_points = Cint[], is_avi=false,
        primal_start::Vector{Cdouble}=Cdouble[],dual_start::Vector{Cdouble}=Cdouble[],
        init_mask=Cint(0))
    return setup(
        daqp, QPj(H,f,A,bupper,blower,sense;A_rowmaj,break_points,is_avi);
        primal_start,dual_start,init_mask)
end

function solve(daqp::DyadDAQP.Model;setup_time=0.0)
    if(!daqp.has_model) return  zeros(0), NaN, -10, [] end
    result_ptr= Ref(DAQPResult(daqp.x,daqp.λ));

    ccall((:daqp_solve, DyadDAQP.libdaqp), Nothing,
          (Ref{DyadDAQP.DAQPResult},Ref{DyadDAQP.Workspace}),
          result_ptr,daqp.work)
    daqp.has_solved = true

    result = unsafe_load(Base.unsafe_convert(Ptr{DAQPResult}, result_ptr))
    info = (x = daqp.x, λ=daqp.λ, fval=result.fval,
            exitflag=result.exitflag,
            status = DyadDAQP.flag2status[result.exitflag],
            solve_time = result.solve_time,
            setup_time = setup_time,
            iterations= result.iter, nodes = result.nodes)
    return copy(daqp.x),result.fval,result.exitflag,info
end


function settings(p::Ptr{DyadDAQP.Workspace})
    workspace = unsafe_load(p);
    if(workspace.settings != C_NULL)
        return unsafe_load(workspace.settings)
    end
end

settings(daqp::DyadDAQP.Model) = settings(daqp.work)

function settings(p::Ptr{DyadDAQP.Workspace}, new_settings::DyadDAQP.DAQPSettings)
    workspace = unsafe_load(p);
    if(workspace.settings != C_NULL)
        old_settings = unsafe_load(workspace.settings)
        unsafe_store!(workspace.settings,new_settings)
        if old_settings.rho_soft != new_settings.rho_soft ||
                old_settings.w_soft != new_settings.w_soft
            ccall((:daqp_refresh_soft_weights,DyadDAQP.libdaqp),Cvoid,
                  (Ptr{DyadDAQP.Workspace},),p)
        end
    end
    return new_settings
end

settings(daqp::DyadDAQP.Model, new_settings::DyadDAQP.DAQPSettings) = settings(daqp.work,new_settings)

function settings(daqp::DyadDAQP.Model,changes::Dict{Symbol,<:Any})
    return settings(daqp.work,changes)
end
function settings(p::Ptr{DyadDAQP.Workspace},changes::Dict{Symbol,<:Any})
    workspace = unsafe_load(p);
    if(workspace.settings == C_NULL) return end
    old_settings = unsafe_load(workspace.settings)
    new = [haskey(changes,f) ? changes[f] : getfield(old_settings,f)
           for f in fieldnames(DyadDAQP.DAQPSettings)];
    new_settings = DyadDAQP.DAQPSettings(new...)
    return settings(p,new_settings)
end

function update(daqp::DyadDAQP.Model, H,f,A,bupper,blower,sense=nothing,break_points=nothing,
        init_mask=Cint(0))
    update_mask = Cint(init_mask);
    work = unsafe_load(daqp.work);
    # The workspace dimensions can differ from the problem's if equality
    # constraints have been eliminated, so validate against the problem.
    qp = unsafe_load(work.qp);
    if(!isnothing(H) && qp.n == size(H,1) && qp.n == size(H,2))
        daqp.qpj.H.=H
        update_mask |= DAQP_UPDATE_Rinv
    end
    if(!isnothing(A) && size(A,1)==(qp.m-qp.ms) && size(A,2)==qp.n)
        daqp.qpj.A.=A'
        update_mask |= DAQP_UPDATE_M
    end

    if(!isnothing(f) && length(f)==qp.n)
        daqp.qpj.f.=f
        update_mask |= DAQP_UPDATE_v
    end

    if(!isnothing(bupper) && !isnothing(blower) &&
       length(bupper)==qp.m && length(blower)==qp.m)
        daqp.qpj.bupper.=bupper
        daqp.qpj.blower.=blower
        update_mask |= DAQP_UPDATE_d
    end

    if(!isnothing(sense) && length(sense)== qp.m)
        daqp.qpj.sense .= sense
        update_mask |= DAQP_UPDATE_sense
    end

    # Omitted sense reuses DAQP's state; explicit sense overrides it.
    if isnothing(sense) && (update_mask & (DAQP_UPDATE_Rinv | DAQP_UPDATE_M)) != 0
        if daqp.has_solved
            work = unsafe_load(daqp.work)
            unsafe_copyto!(pointer(daqp.qpj.sense), work.sense, Int(qp.m))
        end
        update_mask |= DAQP_UPDATE_sense
    end

    if(!isnothing(break_points) && length(break_points)== qp.nh)
        daqp.qpj.break_points .= break_points
        update_mask |= DAQP_UPDATE_hierarchy
    end
    daqp.qpc = QPc(daqp.qpj);
    unsafe_store!(daqp.qpc_ptr, daqp.qpc)

    exitflag = ccall((:daqp_update_ldp,DyadDAQP.libdaqp),Cint,(Cint,Ptr{DyadDAQP.Workspace},Ptr{DyadDAQP.QPc}),
                     update_mask, daqp.work, daqp.qpc_ptr);
end

function reset(p::Ptr{DyadDAQP.Workspace})
    ccall((:daqp_deactivate_constraints,libdaqp),Cvoid,(Ptr{DyadDAQP.Workspace},),p);
    ccall((:reset_daqp_workspace,libdaqp),Cvoid,(Ptr{DyadDAQP.Workspace},),p);
end

function reset(d::DyadDAQP.Model)
    reset(d.work)
end

# Weights of the soft constraints, one entry per constraint (nothing leaves
# the weight at its default: settings.rho_soft and settings.w_soft)
function soft_weights(d::DyadDAQP.Model; rho_l=nothing, rho_u=nothing,
        w_l=nothing, w_u=nothing)
    function arg(v, name)
        isnothing(v) && return nothing
        x = convert(Vector{Cdouble},v)
        length(x) == d.qpj.m || throw(DimensionMismatch(
            "$name must have one entry per constraint"))
        return x
    end
    rl, ru = arg(rho_l, "rho_l"), arg(rho_u, "rho_u")
    wl, wu = arg(w_l, "w_l"), arg(w_u, "w_u")
    ok = GC.@preserve rl ru wl wu ccall(
        (:daqp_set_soft_weights,DyadDAQP.libdaqp),Cint,
        (Ptr{DyadDAQP.Workspace},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble},Ptr{Cdouble}),
        d.work, isnothing(rl) ? C_NULL : rl, isnothing(ru) ? C_NULL : ru,
        isnothing(wl) ? C_NULL : wl, isnothing(wu) ? C_NULL : wu)
    ok == 0 && error("libdaqp was built without support for individual soft weights")
    return nothing
end

using Downloads
function codegen(d::DyadDAQP.Model; fname="daqp_workspace", dir="codegen", prefix="daqp_", src=false)
    @assert(d.has_model, "setup the model before code generation")

    dir[end] != '/' && (dir*="/") ## Make sure it is correct directory path
    isdir(dir) || mkdir(dir)

    reset(d) # Make sure workspace is cleared

    exitflag = ccall((:render_daqp_workspace, libdaqp),Cvoid,
                     (Ptr{DyadDAQP.Workspace},Cstring,Cstring,Cstring), d.work,fname,dir,prefix);
    if src
        cfiles = ["daqp.c","auxiliary.c","factorization.c"]
        hfiles = ["daqp.h","auxiliary.h","factorization.h","constants.h", "types.h"]

        # Append BnB code if there are any binary variables
        if any((d.qpj.sense.&BINARY).==BINARY)
            push!(cfiles,"bnb.c")
            push!(hfiles,"bnb.h")
        end
        if d.qpj.nh>1
            push!(cfiles,"hierarchical.c")
            push!(hfiles,"hierarchical.h")
        end

        # The sources of the library that rendered the workspace, if its DyadDAQP_jll artifact ships them
        # (share/daqp/src, as the builds of DyadDAQP_jll do); otherwise from the branch dyad of JuliaComputing/daqp
        art_src = joinpath(DyadDAQP_jll.artifact_dir, "share", "daqp", "src")
        art_inc = joinpath(DyadDAQP_jll.artifact_dir, "include", "daqp")
        if startswith(DyadDAQP_jll.libdaqp, DyadDAQP_jll.artifact_dir) &&
                all(f -> isfile(joinpath(art_src, f)), cfiles) && all(f -> isfile(joinpath(art_inc, f)), hfiles)
            for (from, files) in ((art_src, cfiles), (art_inc, hfiles)), f in files
                cp(joinpath(from, f), dir*f; force = true)
                chmod(dir*f, 0o644) # the artifact is read-only
            end
        else
            for f in cfiles
                Downloads.download("https://raw.githubusercontent.com/JuliaComputing/daqp/dyad/src/"*f, dir*f)
            end
            for f in hfiles
                Downloads.download("https://raw.githubusercontent.com/JuliaComputing/daqp/dyad/include/"*f, dir*f)
            end
        end
    end
end

function setup_c_workspace(n)::Ptr{DyadDAQP.Workspace}
    p = Libc.calloc(1,sizeof(DyadDAQP.Workspace));
    ccall((:allocate_daqp_workspace,libdaqp), Cvoid, (Ptr{Cvoid},Cint,Cint),p, n, 0);
    ccall((:allocate_daqp_settings,libdaqp), Cvoid, (Ptr{Cvoid},),p);
    return p
end

function free_c_workspace(p::Ptr{DyadDAQP.Workspace})
    ccall((:free_daqp_workspace,libdaqp), Cvoid, (Ptr{Cvoid},),p)
    Libc.free(p)
end

function init_c_workspace_ldp(p::Ptr{DyadDAQP.Workspace},A::Matrix{Cdouble},bupper::Vector{Cdouble},blower::Vector{Cdouble},sense::Vector{Cint}; max_radius=nothing)
    # Set fval_bound to maximal radius for early termination
    if(!isnothing(max_radius))
        d_work = unsafe_load(Ptr{DyadDAQP.Workspace}(p));
        unsafe_store!(Ptr{Cdouble}(d_work.settings+fieldoffset(DAQPSettings,8)),max_radius);
    end

    # Pass pointers for LDP to DAQP
    unsafe_store!(Ptr{Ptr{Cdouble}}(p+fieldoffset(DyadDAQP.Workspace,5)),pointer(A))
    unsafe_store!(Ptr{Ptr{Cdouble}}(p+fieldoffset(DyadDAQP.Workspace,6)),pointer(bupper))
    unsafe_store!(Ptr{Ptr{Cdouble}}(p+fieldoffset(DyadDAQP.Workspace,7)),pointer(blower))
    unsafe_store!(Ptr{Ptr{Cint}}(p+fieldoffset(DyadDAQP.Workspace,10)),pointer(sense))
end

function isfeasible(p::Ptr{DyadDAQP.Workspace}, m=nothing, ms=nothing ;validate=false)::Bool
    # Update A and bupper/blower dimensions
    !isnothing(m)  && unsafe_store!(Ptr{Cint}(p+fieldoffset(DyadDAQP.Workspace,3)),m)
    !isnothing(ms) && unsafe_store!(Ptr{Cint}(p+fieldoffset(DyadDAQP.Workspace,4)),ms)

    exitflag =ccall((:daqp_ldp,DyadDAQP.libdaqp), Int32, (Ptr{Cvoid},),p);

    if(validate && exitflag == -1)
        daqp_ws = unsafe_load(Ptr{DyadDAQP.Workspace}(p))
        m,ms,n = daqp_ws.m, daqp_ws.ms, daqp_ws.n
        A = unsafe_wrap(Matrix{Cdouble}, daqp_ws.M, (n,m), own=false)
        b= unsafe_wrap(Vector{Cdouble}, daqp_ws.dupper, m+ms, own=false)
        AS = copy(unsafe_wrap(Vector{Cint}, daqp_ws.WS, daqp_ws.n_active, own=false))
        AS .+= 1 # Offset for C -> Julia
        lam_star = unsafe_wrap(Vector{Cdouble}, daqp_ws.lam_star, daqp_ws.n_active, own=false)
        err  = dot(b[AS],lam_star)+norm(A[:,AS]*lam_star)
        if(err>0)
            @warn "Couldn't validate infeas. with Frakas (err:$(err), fval=$(daqp_ws.fval))"
        end
    end
    # Make sure workspace is clean for next solve
    reset(p)
    return exitflag == 1
end
