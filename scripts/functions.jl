using CSV, DataFrames, DataFramesMeta, Base.Dates, Plots
using JuMP, Gurobi, Distributions, Clustering, Distances, JLD


@everywhere function f(Sigma_f,c,wf,r)
    1/2*vecdot(wf, Sigma_f*wf) - r*vecdot(c,wf)
end


@everywhere function f_pos(Sigma_f,c,wf,r)
    1/2*vecdot(wf, Sigma_f*wf) + r*vecdot(c,wf)
end

@everywhere function GenerateData(n_f,Sigma_f,c,Af,b,T,ra)
   x = SharedArray{Float64,2}((n_f,T));
   @parallel (+) for i = 1:T
       genData = Model( solver=GurobiSolver( OutputFlag=0, MIPGap = 1e-6, TimeLimit = 600 ) )
       @variable(genData, wf[1:n_f] )
       @objective(genData, Min, f(Sigma_f,c,wf,ra) )
       @constraint(genData, Af*wf .>= b )
       solve(genData)
       x[:,i] = getvalue(wf)
   end
   return x
end



@everywhere function GenerateData_QEP(n_f,Sigma_f,c,Af,b,T,ra)
   x = SharedArray{Float64,2}((n_f,T));
   @parallel (+) for i = 1:T
       genData = Model( solver=GurobiSolver(OutputFlag=0, MIPGap = 1e-6, TimeLimit = 300 ) )
       @variable(genData, wf[1:n_f] )
       @objective(genData, Min, f_pos(Sigma_f,c,wf,ra) )
       @constraint(genData, Af*wf .== b )
       solve(genData)
       x[:,i] = getvalue(wf)
   end
   return x
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
    c = SharedArray{Float64,1}(num_asset);
    for ind_asset = 1:num_asset
        A_vec = A_return[:, ind_asset];
        nonzeroInd = find(x->x>=0, A_vec);
        non_zeroA_vec = A_vec[nonzeroInd];
        c[ind_asset] = median(non_zeroA_vec);
    end
   return c 
end

@everywhere function remove_extreme_negative_return(A_return)
    for i=1:size(A_return)[1]
        for j=1:size(A_return)[2]
           if A_return[i,j]<-10
#                 println(string(i));
#                 println(string(j));
#                 println(A_return[i,j]);
                A_return[i,j] = 0 ;
            end
        end
    end
    A_return[isnan.(A_return)] .= 0;
    return A_return
end

@everywhere function remove_allzeros_rows(A_return)
    
    if sum(broadcast(abs, A_return[:,1]))>0.0
        R = A_return[:,1]
        for i=2:size(A_return)[2]
            if sum(broadcast(abs, A_return[:,i]))>0.0
                R = hcat(R, A_return[:,i]);
            end    
        end
    else
        R = A_return[:,2]
        for i=3:size(A_return)[2]
            if sum(broadcast(abs, A_return[:,i]))>0.0
                R = hcat(R, A_return[:,i]);
            end            
        end
    end            
    return R
end

@everywhere function remove_allzeros_rows_withobs(A_return, X_obs)
    if (size(A_return,2)!=size(X_obs,2))
        println("Unequal sample size");
        return None, None
    else
        if sum(broadcast(abs, A_return[:,1]))>0.0
            R = A_return[:,1];
            Xr = X_obs[:,1];
            for i=2:size(A_return)[2]
                if sum(broadcast(abs, A_return[:,i]))>0.0
                    R = hcat(R, A_return[:,i]);
                    Xr =hcat(Xr, X_obs[:,i]);        
                end    
            end
        else
            R = A_return[:,2];
            Xr = X_obs[:,2];                        
            for i=3:size(A_return)[2]
                if sum(broadcast(abs, A_return[:,i]))>0.0
                    R = hcat(R, A_return[:,i]);
                    Xr =hcat(Xr, X_obs[:,i]);        
                end            
            end
        end            
        return R, Xr
    end
        
end    

@everywhere function decompose(A_return, num_comp)
    # spectral decompositon of the covariance matrix
    cov_mat = cov(A_return);
    Q = cov_mat;
    
    factor = eigfact(Q);
    eigvalues = factor[:values];
    eigvectors = factor[:vectors];
    
    large_eigs = eigvalues[length(eigvalues)-num_comp+1:length(eigvalues)];
    
    F = eigvectors[:,length(eigvalues)-num_comp+1:length(eigvalues)];
    
    Sigma_f = Diagonal(large_eigs);
    
   return F, Sigma_f 
end
            
            
@everywhere function decompose_new(A_return, num_comp)
    # spectral decompositon of the covariance matrix
    cov_mat = cov(A_return);
    Q = cov_mat;
    Q[isnan(Q)] = 0;
                
    da = diag(Q);
    da[da.==0]=0;
    n=size(Q,1);            
    Q[1:n+1:end]=da;            
    
                
    factor = eigfact(Q);
    eigvalues = factor[:values];
    eigvectors = factor[:vectors];
    
    large_eigs = eigvalues[length(eigvalues)-num_comp+1:length(eigvalues)];
    
    F = eigvectors[:,length(eigvalues)-num_comp+1:length(eigvalues)];
    
    Sigma_f = Diagonal(large_eigs);
    
   return F, Sigma_f 
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

    
@everywhere function fixdiagonal(sig)
    d = diag(sig);
        if d[length(d)]>5*d[length(d)-1]
            println("eigen value is too large");
            d[length(d)] =  d[length(d)-1]+0.02;
        end
    return Diagonal(d)
            
end    
            
@everywhere function normalize_doublesigns(vec)
    outvec = deepcopy(vec);
    pos_ind=find(x->x>0, outvec);
    neg_ind =find(x->x<0, outvec);
    sum_neg=sum(outvec[neg_ind]);
    sum_pos=sum(outvec[pos_ind]); 

    outvec[pos_ind]=outvec[pos_ind]./sum_pos;
    outvec[neg_ind]=-outvec[neg_ind]./sum_neg;
    return outvec;
end                
            
            
@everywhere function remove_allzeros_rows_withobs(A_return, X_obs)
        
    if (size(A_return,2)!=size(X_obs,2))
        println("Unequal sample size");
        return None, None
    else
        if sum(broadcast(abs, A_return[:,1]))>0.0
            R = A_return[:,1];
            Xr = X_obs[:,1];
            for i=2:size(A_return)[2]
                if sum(broadcast(abs, A_return[:,i]))>0.0
                    R = hcat(R, A_return[:,i]);
                    Xr =hcat(Xr, X_obs[:,i]);        
                end    
            end
        else
            R = A_return[:,2];
            Xr = X_obs[:,2];                        
            for i=3:size(A_return)[2]
                if sum(broadcast(abs, A_return[:,i]))>0.0
                    R = hcat(R, A_return[:,i]);
                    Xr =hcat(Xr, X_obs[:,i]);        
                end            
            end
        end            
        return R, Xr
    end
end 
                    
@everywhere function validation(all_configs, Sigma_f, caf, k, n, F)
    
    T = 200;
    return_x_err = SharedArray{Float64,2}((size(all_configs,1),T));
    return_r_err = SharedArray{Float64,2}((size(all_configs,1),T));
    
    for i = 1:size(all_configs,1)
        
        # obtaint the parameters from the all_configs vector
        M=all_configs[i][1];
        lambda=all_configs[i][2];
        err=all_configs[i][3];
        ra_guess=all_configs[i][4];
        ra_true=all_configs[i][5];
        
        Af = -[eye(n);ones(1,n);-ones(1,n);-eye(n)]*F; 
        
        bf = -[ones(n,1);1;-1;err*ones(n,1)];    
        mf = length(bf);        
        
        x_err_vec = SharedArray{Float64}(T);
        r_err_vec = SharedArray{Float64}(T);
        
        for j = 1:T
            
            x_recon = GenerateData(k,Sigma_f,caf,Af,bf,1,ra_true);
            ra_guess, x2 = qp_r_online_ver3(mf,k,M,lambda*j^(-1/2),Sigma_f,caf,Af,bf,x_recon,ra_guess); 
            
            x_err_vec[j]=norm(x2-x_recon,2)/norm(x_recon,2)
            r_err_vec[j]=(ra_guess-ra_true).^2./(ra_true.^2)
            
        end

        return_x_err[i,:]=x_err_vec;
        return_r_err[i,:]=r_err_vec;
        
    end     
    
    return return_x_err, return_r_err;
end

function plot_asset_eigenspace(outfile)  
    c=ColorGradient([:red,:yellow,:blue]);
    plot(ef_riskvec_ineq,ef_profitvec_ineq, linewidth = 4, lc=c, line_z=log(steps), label="inequality constraint directly in eigenspace", color =:red, xtickfontsize=5, ytickfontsize=5, size = (500, 500), legendfontsize = 8, titlefontsize=7,legend=:bottomright, title="inequality constraiant directly in eigenspace", fmt = :png);
    plot!(ef_riskvec_asset_projected,ef_profitvec_asset_projected, seriestype = :scatter, markersize=1, legend = false, color = [:black], size = (500, 500), fmt = :png);
    plot!(ef_riskvec_fund_projected,ef_profitvec_fund_projected, seriestype = :scatter, markersize=5, marker=:cross, legend = false, color = [:cyan], size = (500, 500), fmt = :png);                    
    png(outfile);                    
end            

function plot_asset_originalspace(outfile) 
    c=ColorGradient([:red,:yellow,:blue]);                    
    plot(ef_riskvec_recon_ineq, ef_profitvec_recon_ineq, linewidth = 4, lc=c, line_z=log(steps), label="inequality constraint mapped to original space", color =:red, xtickfontsize=5, ytickfontsize=5, size = (500, 500), legendfontsize = 8, titlefontsize=7, legend=:bottomright, title="inequality constraint mapped to original space", fmt = :png);
    plot!(ef_riskvec_asset_original,ef_profitvec_asset_original, seriestype = :scatter, markersize=1, legend = false, color = [:black], size = (500, 500), fmt = :png);
    plot!(ef_riskvec_fund_original,ef_profitvec_fund_original, seriestype = :scatter, markersize=5, marker=:cross, legend = false, color = [:cyan], size = (500, 500), fmt = :png);
    png(outfile);                    
end            

function plot_asset_ori(outfile) 
    c=ColorGradient([:red,:yellow,:blue]);
    plot(ef_riskvec_ori, ef_profitvec_ori, linewidth = 4, lc=c, line_z=log(steps), label="inequality constraint mapped to original space \n double unnormalized", color = :red, xtickfontsize=5, ytickfontsize=5, size = (500, 500), legendfontsize = 8, titlefontsize=7, legend=:bottomright, title="inequality constraint original", fmt = :png);
    plot!(ef_riskvec_asset_original,ef_profitvec_asset_original, seriestype = :scatter, markersize=1, legend = false, color = [:black], size = (500, 500), fmt = :png);
    plot!(ef_riskvec_fund_original,ef_profitvec_fund_original, seriestype = :scatter, markersize=5, marker=:cross, legend = false, color = [:cyan], size = (500, 500), fmt = :png);
    png(outfile);
end                      

                    
function plot_asset_sector(outfile)     
    c=ColorGradient([:red,:yellow,:blue]);
    plot(ef_riskvec_sector, ef_profitvec_sector, linewidth = 4, lc=c, line_z=log(steps), label="inequality constraint sector space", color = :red, xtickfontsize=5, ytickfontsize=5, size = (500, 500), legendfontsize = 8, titlefontsize=7, legend=:bottomright, title="inequality constraint sector space", fmt = :png);
    plot!(ef_riskvec_fund_sector,ef_profitvec_fund_sector, seriestype = :scatter, markersize=1, legend = false, color = [:black], size = (500, 500), fmt = :png);
    png(outfile);
end    
       
                    
                    
function plot_asset_sector_withRandomPortfolio(outfile)
    x = range(1,1,10);
    labels = string.(collect(x));
    c=ColorGradient([:red,:yellow,:blue]);
                        
    # varying risk aversion value to generate portfolio                             
    p1=plot(ef_riskvec_sector, ef_profitvec_sector, linewidth=4, lc=c, line_z=log(steps), label="inequality constraint sector space", color = :red, xtickfontsize=10, ytickfontsize=10, size=(800, 800), legendfontsize=8, titlefontsize=7, legend=:bottomright, title="Efficient Frointer in Sector Space", fmt = :png);
            
    #plot random portfolio <risk,profit> points                            
    #plot!(ef_riskvec_random, ef_profitvec_random, text=labels, seriestype = :scatter, label="random portfolio", fmt = :png);
    plot!(ef_riskvec_random, ef_profitvec_random, seriestype = :scatter, legend=:bottomright, label="random portfolio", fmt = :png);
    
    #plot real fund portfolio on sector level                        
    plot!(ef_riskvec_fund_sector, ef_profitvec_fund_sector, seriestype = :scatter, legend=:bottomright, label="VFINX portfolio",  fmt = :png);
                        
    # guess from random portfolios                    
    p2=bar(all_r,color="red",alpha=0.4, titlefontsize=7, title="Estimated Risk Aversion \n from Random Portfolios");
                            
    # guess from real fund sector portfolios                         
    p3=bar(all_fund_r,color="green",alpha=0.4, titlefontsize=7, title="Estimated Risk Aversion \n from Fund Portfolios"); 
                        
    l = @layout [a{0.8w} [b{0.5h}; b{0.5h}]];
                        
    display(plot(p1, p2, p3, layout = l, legend = false));
    png(outfile);
end                        
                    

# generate portfolio directly in eigen space
function plot_diagonal_rguess(outfile)
    #i_rand is the 
    i_rand = sortperm(r_rand);
    i_benchmark = sortperm(all_benchmark_r);
    plot(i_rand,i_benchmark, seriestype = :scatter, xlabel="True Risk Aversion Order", ylabel="Estimated Risk Aversion Order", xtickfontsize=20, ytickfontsize=20,  titlefontsize=30, title="Order Rank", markersize=r_rand[i_rand]*10, legend = false, color = [:blue], size = (500, 500), fmt = :png);
    plot!([0,30],[0,30], color=[:black], linewidth=0.8);                     
    png(outfile);
end
        
                    
                    
# generate portfolio directly in eigen space
function plot_diagonal_rguess_eigen(outfile)
    #i_rand is the 
    i_rand = sortperm(r_rand);
    i_benchmark = sortperm(all_benchmark_r);
    plot(i_rand,i_benchmark, seriestype = :scatter, xlabel="True Risk Aversion Order", ylabel="Estimated Risk Aversion Order", xtickfontsize=10, ytickfontsize=10,  titlefontsize=10, title="Order Rank", markersize=r_rand[i_rand]/10, legend = false, color = [:orange], size = (500, 500), fmt = :png);
    plot!([0,30],[0,30], color=[:black], linewidth=0.8);                     
    png(outfile);
end
                    
                    
# generate portfolio directly in eigen space
function plot_diagonal_rguess_truevalue(outfile)
    plot(r_rand,all_benchmark_r, seriestype = :scatter, xlabel="True Risk Aversion Value", ylabel="Estimated Risk Aversion Value", xtickfontsize=20, ytickfontsize=20,  titlefontsize=30, title="Comparison Rank", markersize=r_rand/2, legend = false, color = [:blue], size = (500, 500), fmt = :png);
                        
    plot!([0,80],[0,80], color=[:black], linewidth=0.8);
    png(outfile);
end
             
                    
# generate portfolio in the original space, and project it to eigenspace
function plot_diagonal_rguess_projection(outfile)
    i_rand_p = sortperm(r_rand_ori);
    i_benchmark_p = sortperm(all_benchmark_projected_r);
    plot(i_rand_p,i_benchmark_p, seriestype = :scatter, markersize=r_rand_ori[i_rand_p],  xlabel="True Risk Aversion Order", ylabel="Estimated Risk Aversion Order", legend = false, color = [:orange], size = (500, 500), fmt = :png);
    png(outfile);
end
                    
# generate portfolio in the original space, and project it to eigenspace                        
function plot_diagonal_rguess_projection_truevalue(outfile)
    plot(r_rand, all_benchmark_r, seriestype = :scatter, title="Comparison Rank", markersize=r_rand/10, xtickfontsize=10, ytickfontsize=10,  titlefontsize=10, xlabel="True Risk Aversion Value", ylabel="Estimated Risk Aversion Value", legend = false, color = [:orange], size = (500, 500), fmt = :png);
    plot!([0,80],[0,80], color=[:black], linewidth=0.8);
    png(outfile);
end

                    
# plot the EFs                    
function plot_rand_portfolio_onef(outfile)
    c=ColorGradient([:red,:yellow,:blue]);
    p1=plot(ef_riskvec_sector, ef_profitvec_sector, linewidth = 4, lc=c, line_z=log(steps), label="efficient frontier", color = :red, xtickfontsize=15, ytickfontsize=15, size = (500, 500), legendfontsize = 8, titlefontsize=30, legend=:bottomright, title="Efficient frontier", fmt = :png);
    plot!(ef_riskvec_random_onef, ef_profitvec_random_onef, lc=c, markersize=r_rand*15, seriestype = :scatter, label="random portfolio", fmt = :png);
    png(outfile);                    
end
                    
                    
                    
function plot_asset_eigenspace_mono(outfile)   
    c=ColorGradient([:red,:yellow,:blue]);
    # plot EF directly generated in eigenspace
    # color is red 
    plot(ef_riskvec_ineq,ef_profitvec_ineq, linewidth = 4, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, title="Efficient frontier for factor-based portfolio (k=5)", fmt = :png);
                        
    plot!(ef_riskvec_random_onef, ef_profitvec_random_onef, seriestype=:scatter, markersize=r_rand./5, marker=:circle, legend=:bottomright,  label="random portfolio generated in factor space", color = [:orange], size = (500, 500), fmt=:png);
                        

                        
    png(outfile);                    
end     
                  
function plot_asset_eigenspace_k(outfile) 
    c=ColorGradient([:red,:yellow,:blue]);
    plot(ef_riskvec_ineq_5,ef_profitvec_ineq_5, linewidth = 1, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", label="k=5", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, title="inequality constriant directly in eigenspace", fmt = :png);
    plot!(ef_riskvec_ineq_10,ef_profitvec_ineq_10, linewidth = 2, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", label="k=10", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, fmt = :png); 
    plot!(ef_riskvec_ineq_15,ef_profitvec_ineq_15, linewidth = 2, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", label="k=15", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, fmt = :png); 
    plot!(ef_riskvec_ineq_20,ef_profitvec_ineq_20, linewidth = 3, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", label="k=20", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, fmt = :png); 
    plot!(ef_riskvec_ineq_25,ef_profitvec_ineq_25, linewidth = 4, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", label="k=25", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, fmt = :png);                         
    plot!(ef_riskvec_ineq_30,ef_profitvec_ineq_30, linewidth = 5, lc=c, line_z=log(steps), xlabel="Risk", ylabel="Profit", label="k=30", color =:red, titlefontsize=10, xtickfontsize=10, ytickfontsize=10, size = (500, 500), legendfontsize = 8, legend=:bottomright, fmt = :png);                         
    png(outfile); 
end
        