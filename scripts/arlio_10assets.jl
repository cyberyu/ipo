## iop_online
#  Shi Yu, Chaosheng Dong
#  shi_yu@vanguard.com
#  chaosheng@pitt.edu
#  All rights reserved 2017-03-17
# function
@everywhere using JuMP, Gurobi, Distributions, Clustering, Distances, CSV, NPZ
@everywhere function f(Q,c,r,x)
    1/2*r*vecdot(x, Q*x) - vecdot(c,x)
end

@everywhere function GenerateData(n_f,Sigma_f,c,Af,b,T,ra)
   x = SharedArray{Float64,2}((n_f,T));
   @parallel (+) for i = 1:T
       genData = Model( solver=GurobiSolver(OutputFlag=0, MIPGap = 1e-6,TimeLimit = 600 ) )
       @variable(genData, wf[1:n_f])
       @objective(genData, Min, f(Sigma_f,c,ra,wf))
       @constraint(genData, Af*wf .>= b )
       solve(genData)
       x[:,i] = getvalue(wf)
   end
   return x
end

@everywhere function GenerateData2(n,Q,c,A,b,r)
    genData = Model( solver=GurobiSolver( OutputFlag=0 ) )
    @variable(genData, y[1:n] )
    @objective(genData, Min, f(Q,c,r,y) )
    @constraint(genData, A*y .>= b)
    solve(genData)
    return getvalue(y)
end

@everywhere function qp_r_online_ver2(m,nf,M,eta,Sigma_f,c,Af,b,wf,ra)
   problem = Model(solver=GurobiSolver(OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 600) )
   @variable(problem, 0.000001<= ra_t<= 100) # risk aversion value at time t
   @variable(problem, x[1:nf])
#    @variable(problem, 0<=x[1:n]<=1 )
   @variable(problem, u[1:m] >= 0 )
   @variable(problem, z[1:m], Bin )
   @objective(problem, Min, vecdot(ra - ra_t,ra - ra_t)/2 + eta*vecdot(wf - x,wf - x) )
   @constraint(problem, Af*x .>= b )
   @constraint(problem, u .<= M*z )
   @constraint(problem, Af*x - b .<= M*(1-z))
   @constraint(problem, Sigma_f*x - ra_t*c - Af'*u .== zeros(nf) )
   solve(problem)
   return getvalue(ra_t), getvalue(x)
end

@everywhere function qp_r_online_ver3(m,nf,M,eta,Sigma_f,c,Af,b,wf,ra)
   problem = Model(solver=GurobiSolver(OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 3000) )
   @variable(problem, 0.000001<= ra_t<= 100) # risk aversion value at time t
   @variable(problem, x[1:nf])
#  @variable(problem, 0<=x[1:n]<=1 )
   @variable(problem, u[1:m] >= 0 )
   @variable(problem, z[1:m], Bin )
   #@variable(problem, e<=0.0001)
   @objective(problem, Min, vecdot(ra - ra_t,ra - ra_t)/2 + eta*vecdot(wf - x,wf - x))
   @constraint(problem, Af*x .>= b)
   @constraint(problem, u .<= M*z )
   @constraint(problem, Af*x - b .<= M*(1-z))
   @constraint(problem, Sigma_f*x - ra_t*c - Af'*u .== zeros(nf) )
   solve(problem)
   return getvalue(ra_t), getvalue(x)
end

@everywhere function qp_r_online_ver4(m,nf,M,eta,Sigma_f,c,Af,b,x,ra)
   problem = Model(solver=GurobiSolver(OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 300) )
   @variable(problem, 0.000001<= ra_t<= 100) # risk aversion value at time t
#    @variable(problem, 0<=x[1:n]<=1 )
   @variable(problem, u[1:m] >= 0 )
   @variable(problem, z[1:m], Bin )
   @objective(problem, Min, vecdot(ra - ra_t,ra - ra_t)/2)
   @constraint(problem, Af*x .>= b )
   @constraint(problem, u .<= M*z )
#   @constraint(problem, Af*x - b .<= M*(1-z))
   @constraint(problem, ra_t*Sigma_f*x - c - Af'*u .== zeros(nf) )
   solve(problem)
   return getvalue(ra_t)
end

@everywhere function qp_r_online_ver5(m,lambda,Q,c,A,b,x,r)
    problem = Model(solver=GurobiSolver(OutputFlag = 0, MIPGap = 1e-3, TimeLimit = 300, InfUnbdInfo=1) )
    @variable(problem, -10 <= r_t <= 10)
    @variable(problem, u[1:m] >= 0)
    @variable(problem, t[1:m],Bin)
    @variable(problem, eta1)
    @objective(problem, Min, eta1/2 )
    @constraint(problem, norm([2*(r - r_t); eta1-1]) <= eta1 + 1 )
    #@constraint(problem, A*x .>= b )
    @constraint(problem, u.<= M*t )
    #@constraint(problem, A*x - b .<= M*(1-t) )
    @constraint(problem, r_t*Q*x - c - A'*u .== zeros(m) )
    solve(problem)
    return getvalue(r_t)
end


@everywhere function qp_r_online_ver2(m,nf,M,eta,Sigma_f,c,Af,b,wf,ra)
   problem = Model(solver=GurobiSolver(OutputFlag = 0, MIPGap = 1e-6, TimeLimit = 300) )
   @variable(problem, 0.000001<= ra_t<= 100) # risk aversion value at time t
   @variable(problem, x[1:nf])
#    @variable(problem, 0<=x[1:n]<=1 )
   @variable(problem, u[1:m] >= 0 )
   @variable(problem, z[1:m], Bin )
   @objective(problem, Min, vecdot(ra - ra_t,ra - ra_t)/2 + eta*vecdot(wf - x,wf - x) )
   @constraint(problem, Af*x .>= b )
   @constraint(problem, u .<= M*z )
   @constraint(problem, Af*x - b .<= M*(1-z))
   @constraint(problem, Sigma_f*x - ra_t*c - Af'*u .== zeros(nf) )
   solve(problem)
   return getvalue(ra_t), getvalue(x)
end


@everywhere function decompose(A_return, num_comp)
    # spectral decompositon of the covariance matrix
    cov_mat = cor(A_return);
    Q = cov_mat;
    Q[isnan(Q)] = 0;
    Q = Q + 0.0001*I;
    
    factor = eigfact(Q);
    eigvalues = factor[:values];
    eigvectors = factor[:vectors];
    
    large_eigs = eigvalues[length(eigvalues)-num_comp+1:length(eigvalues)];
    
    F = eigvectors[:,length(eigvalues)-num_comp+1:length(eigvalues)];
    
    Sigma_f = Diagonal(large_eigs);
    
   return F, Sigma_f 
end

@everywhere function compute_c(A_return)
#    A_return: matrix of size (num of time steps, num of assets) 

    num_asset = size(A_return)[2];
    c = SharedArray{Float64,1}(num_asset);
    for ind_asset = 1:num_asset
        A_vec = A_return[:, ind_asset];
        nonzeroInd = find(x->x!=0, A_vec);
        non_zeroA_vec = A_vec[nonzeroInd];
        c[ind_asset] = mean(non_zeroA_vec);
    end
   return c 
end

@everywhere function compute_c_max(A_return)
#    A_return: matrix of size (num of time steps, num of assets) 

    num_asset = size(A_return)[2];
    #print("num_asset ",num_asset)
    c = SharedArray{Float64,1}(num_asset);
    for ind_asset = 1:num_asset
        A_vec = A_return[:, ind_asset];
        nonzeroInd = find(x->x>=0, A_vec);
        non_zeroA_vec = A_vec[nonzeroInd];
        c[ind_asset] = median(non_zeroA_vec);
    end
   return c 
end

using CSV, DataFrames, DataFramesMeta, Base.Dates
@everywhere using JuMP, Gurobi, Distributions, Clustering, Distances, Plots, JLD

fund_name="arlio";

A = zeros(10, 425);

icount = 1;

lambda =10;
M=10;

    

#X_obs = CSV.read("../data_shiyu/"string(fund_name)"_random_asset_bigvariance/"string(fund_name)"_X.txt",delim=",",datarow=1);
X_obs = CSV.read("/home/syu/Documents/Projects/ipo_kdd/ipo/data/drl/drl_10_assets_target1.1/arlio_X.txt",delim=",",datarow=1);
X_obs = Matrix(X_obs);
X_obs = X_obs[:,1:end]
Xobsmat = Matrix(X_obs);
Xobsmat = transpose(Xobsmat);


for sind=1:10:100

    universe_results = Vector{Vector{Float64}}(); 
    universe_results_ratio = Vector{Vector{Float64}}(); 

    all_results = Vector{Vector{Float64}}();
    all_results2 = Vector{Vector{Float64}}();
    Time = Float64[];  
    
    for ind=1:1:425
        A_return = CSV.read("/home/syu/Documents/Projects/ipo_kdd/ipo/data/drl/drl_10_assets_target1.1/arlio_A_"*string(ind)*".txt",datarow=1);
        A_return = Matrix(A_return); 
        println("A matrix size ", size(A_return));
        #sind=size(A_return)[1]-lookback; 


        A_return = A_return[sind:size(A_return)[1], :]; 
        #A_return = A_return/12;

        #A_return = remove_extreme_negative_return(A_return);
        #A_return = A_return./100;
        #c = maximum(A_return,2);

        ind_results = Vector{Float64}();

        ra =1;
        c = compute_c(A_return);
        #c = compute_c_max(A_return);
        #println(c);

        c[isnan.(c)] .= 0;
        #print(c);

        n=10;

        Q = cov(A_return);
        Q = Q + 0.001*I;


        xx_obs = X_obs[:, 1:ind]; 
        est_return = Vector{Float64}();
        est_return2 = Vector{Float64}();


        #println(size(xx_obs)[2]);

        tic();

        if size(xx_obs)[2]<=10
            s=1
        else
            s=size(xx_obs)[2]-10
        end

        for t = s:size(xx_obs)[2]

            #println(ind,ra,t);

            y = xx_obs[:,t];
            y = y/sum(y);

            y[isnan.(y)] .= 0;


            #A = -[eye(n);ones(1,n);-ones(1,n);-eye(n)]; 
            #A_simple= [c';eye(n)]

            A_complex=-[eye(n);ones(1,n);-ones(1,n);-eye(n)]

            #b = -[ones(n,1);1;-1;zeros(n,1)];
            #b_simple = zeros(n+1,1);

            b_complex= -[ones(n,1);1;-1;zeros(n,1)];

            (m,n) = size(A_complex);
            eta = lambda*t^(-1/2);
            #print(ra, y,c)
                
            try
                ra, x = qp_r_online_ver2(m,n,M,eta,Q,c,A_complex,b_complex,y,ra);
            catch e
                println(e);
                push!(est_return, 0);
                push!(est_return2, transpose(y)*Q*y./(transpose(c)*y));                
                continue
            end

            push!(est_return, ra);
            push!(est_return2, transpose(y)*Q*y./(transpose(c)*y));
        end
        print(est_return);
        push!(all_results, est_return); 
        push!(all_results2, est_return2); 
        t = toc();
        Time = push!(Time, t);
    end

    temp_result = Vector{Float64}();
    temp_result2 = Vector{Float64}();

    for i=1:425
        push!(temp_result, all_results[i][end]);
        push!(temp_result2, all_results2[i][end]);
    end
    push!(universe_results, temp_result);
    push!(universe_results_ratio, temp_result2);
        
    for i=1:425
        A[icount, i]=universe_results[1][i];
    end
        
    icount = icount+1;    
end

save("/home/syu/Documents/Projects/ipo_kdd/ipo/results/A_10assetnew_10_obs_10L_10M.jld","A",A)
npzwrite("/home/syu/Documents/Projects/ipo_kdd/ipo/results/A_10assetnew_10_obs_10L_10M.npy", A)