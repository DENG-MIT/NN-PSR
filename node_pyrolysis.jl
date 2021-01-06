using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using DiffEqSensitivity
using Zygote
# using ForwardDiff
using Interpolations
using LinearAlgebra
using Random
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DelimitedFiles

is_restart = false;
n_epoch = 10000;
batch_size = 500;
n_plot = 1;

opt = ADAMW(0.001, (0.9, 0.999), 1.f-6);

lb = 1.e-6;
ub = 1.e5;
llb = 1.e-12;
ode_solver = AutoTsit5(Rosenbrock23(autodiff=false));
# ode_solver = Tsit5();

rawdata = readdlm("cantera/data_T.txt")';
tsteps = rawdata[1, :];
Tlist = rawdata[2, :];
# Plist = rawdata[3, :];
ylabel = rawdata[2:end, :];
t_end = tsteps[end]
tspan = (0.0, tsteps[end]);
ns = size(ylabel)[1];
ntotal = size(ylabel)[2];
batch_size = minimum([batch_size, ntotal]);
varnames = ["H2", "CO", "CH4", "O2", "N2", "CO2"];

ymax = maximum(ylabel, dims=2);
ymin = minimum(ylabel, dims=2);

ymean = mean(ylabel, dims=2);
ystd = std(ylabel, dims=2);

yscale = ymax - ymin;
normdata = (ylabel .- ymin) ./ yscale;

# normdata = @. log(ylabel + llb);
# yscale = std(normdata, dims=2)[:, 1];

u0 = ylabel[:, 1];

nr = 4
dudt2 = Chain(x -> x,
            Dense(ns, ns * nr, relu),
            Dense(ns * nr, ns * nr, relu),
            Dense(ns * nr, ns * nr, relu),
            Dense(ns * nr, ns + 1))

p, re = Flux.destructure(dudt2);

uin = readdlm("cantera/TYin.txt");
Q = 8.e2;
t_cycle = 1e3;
Ta = 760;
tres = 1.0;

function dudt!(du, u, p, t)

    nnout = re(p)((u .- ymean) ./ ystd)
    TYdot = nnout[1:ns] / t_cycle
    rhocp = abs(nnout[ns + 1]) + 100

    dT = (uin[1] - u[1]) / tres + TYdot[1] + Q * (Ta - u[1]) / rhocp
    dY = (uin[2:end] - u[2:end]) / tres + TYdot[2:end]

    du .= vcat(dT, dY)
end

prob = ODEProblem(dudt!, u0, tspan)

# sense = InterpolatingAdjoint(autojacvec=ZygoteVJP(); checkpointing=true)
sense = BacksolveAdjoint(checkpointing=true)
function predict_n_ode(p, sample)
    # global rep = re(p)
    _prob = remake(prob, p=p, tspan=[0, tsteps[sample]])
    pred = Array(solve(_prob, ode_solver, saveat=tsteps[1:sample],
                 atol=lb, sensalg=sense))
end
pred = predict_n_ode(p, ntotal)

function loss_n_ode(p, sample)
    pred = predict_n_ode(p, sample)
    loss = mae(pred[1, :], Tlist[1:size(pred)[2]])
    return loss
end
loss_n_ode(p, ntotal)

list_loss = []
list_grad = []
iter = 1
cb = function (p, loss_mean, g_norm; doplot=true)
    global list_loss, list_grad, iter
    push!(list_loss, loss_mean)
    push!(list_grad, g_norm)

    if doplot & iter % n_plot == 0
        pred = predict_n_ode(p, ntotal)

        list_plt = []
        for i in 1:ns
            plt = plot(tsteps, ylabel[i,:], label="data", legend=:outertopright, size=(900, 500))
            plot!(plt, tsteps, pred[i,:], label="pred")
            ylabel!(plt, "$(varnames[i])")
            xlabel!(plt, "Time [ms]")
            push!(list_plt, plt)
        end
        plt_all = plot(list_plt..., legend=true)
        png(plt_all, "figs/pred.png")

        println("update plot for i_exp")

        plt_loss = plot(list_loss, xscale=:identity, yscale=:log10, label="loss", legend=:outertopright)
        plt_grad = plot(list_grad, xscale=:identity, yscale=:log10, label="grad_norm", legend=:outertopright)
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Grad Norm")
        plt_all = plot([plt_loss, plt_grad]..., legend=true)
        png(plt_all, "figs/loss_grad")


        @save "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    end
    iter += 1
    return false
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    iter += 1
end

epochs = ProgressBar(iter:n_epoch);
for epoch in epochs
    global p

    sample = rand(batch_size:ntotal)

    loss, back = Zygote.pullback(x -> loss_n_ode(x, sample), p)
    grad = back(one(loss))[1]

    grad_norm = norm(grad, 2)
    # grad = grad ./ grad_norm .* 1.e0

    update!(opt, p, grad)
    
    set_description(epochs, string(@sprintf("Loss: %.4e grad: %.2e", loss, grad_norm)))
    cb(p, loss, grad_norm)
end



# function f(p)
#     return loss_n_ode(p, ntotal)
# end

# function g!(G, p)
#     # G .= ForwardDiff.gradient(f, x);
#     loss, back = Zygote.pullback(x -> f(x), p)
#     G .= back(one(loss))[1]
# end

# G = zeros(size(p));
# loss = f(p)
# g!(G, p);
# grad_norm = norm(G, 2)

# pp = p;
# for ii in 1:20
#     global pp
#     res = optimize(f, g!, pp,
#                     BFGS(),
#                     Optim.Options(g_tol=1e-12, iterations=5,
#                                   store_trace=true, show_trace=true))
#     pp = res.minimizer
#     loss = f(pp)
#     g!(G, pp)
#     grad_norm = norm(G, 2)
#     println("bfgs iter $ii loss $loss grad $grad_norm")
#     cb(pp, loss, grad_norm; doplot=true)
# end