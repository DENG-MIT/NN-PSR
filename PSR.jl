using Arrhenius
using LinearAlgebra
using DifferentialEquations
using ForwardDiff
using DiffEqSensitivity
using Sundials
using Plots
using DelimitedFiles
using Profile

cantera_data = readdlm("cantera/data_T.txt");
ct_ts = cantera_data[:, 1];
ct_T = cantera_data[:, 2];
ct_Y = cantera_data[:, 3:end];

gas = CreateSolution("./cantera/gri30.yaml");
ns = gas.n_species;

# 300, P, 'H2:9.5023,CO:1.7104,CH4:5.7014,O2:17.0090,N2:66.0769'
TYin = readdlm("cantera/TYin.txt")
Yin = zeros(ns)
Tin = 300.0

for (i, s) in enumerate(["H2", "CO", "CH4", "O2", "N2"])
    Yin[species_index(gas, s)] = TYin[i + 1]
end

Y0 = ct_Y[1, :]
T0 = ct_T[1]
P = one_atm

@inbounds function dudt!(du, u, p, t)
    Tin = p[1]  # 300.0
    Ta = p[2]  # 760.0
    Q = p[3] # 8.e2
    tres = p[4] # 1.0
    T = u[end]
    Y = @view(u[1:ns])
    mean_MW = 1. / dot(Y, 1 ./ gas.MW)
    ρ_mass = P / R / T * mean_MW
    X = Y2X(gas, Y, mean_MW)
    C = Y2C(gas, Y, ρ_mass)
    cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
    h_mole = get_H(gas, T, Y, X)
    S0 = get_S(gas, T, P, X)
    wdot = wdot_func(gas.reaction, T, C, S0, h_mole)
    Ydot = @. wdot / ρ_mass * gas.MW + (Yin - Y) / tres
    Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass +
            (Tin - T) / tres +
            Q * (Ta - T) / ρ_mass / cp_mass
    du .= vcat(Ydot, Tdot)
end

u0 = vcat(Y0, T0)
tspan = [0.0, 1.5];
p = [300.0, 760.0, 8.e2, 1.0];
prob = ODEProblem(dudt!, u0, tspan, p);
sol = solve(prob, TRBDF2(), reltol=1e-6, abstol=1e-9);

plt = plot(sol.t, sol[species_index(gas, "CH4"), :], lw=2, label="Arrhenius.jl");
plot!(plt, ct_ts, ct_Y[:, species_index(gas, "CH4")], label="Cantera");
ylabel!(plt, "Mass Fraction of CH4");
xlabel!(plt, "Time [s]");
plt_CO = plot(sol.t, sol[species_index(gas, "CO"), :], lw=2, label="Arrhenius.jl");
plot!(plt_CO, ct_ts, ct_Y[:, species_index(gas, "CO")], label="Cantera");
ylabel!(plt_CO, "Mass Fraction of CO");
xlabel!(plt_CO, "Time [s]");
pltT = plot(sol.t, sol[end, :], lw=2, label="Arrhenius.jl");
plot!(pltT, ct_ts, ct_T, label="Cantera");
ylabel!(pltT, "Temperature [K]");
xlabel!(pltT, "Time [s]");
# title!(plt, "JP10 pyrolysis @1200K/1atm")
pltsum = plot(plt, plt_CO, pltT, legend=true, framestyle=:box, layout=(3, 1), size=(1200, 1200));
png(pltsum, "figs/PSR.png");


# sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(false))
# sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(false))
# sensealg = ForwardDiffSensitivity()
# sensealg = SensitivityADPassThrough()
sensealg = ForwardSensitivity(autojacvec=true)
function fsol(p)
    sol = solve(prob, p=p, TRBDF2(), tspan=[0.0, 0.5],
                reltol=1e-6, abstol=1e-9, sensealg=sensealg)
    return sol[end, end]
end
println("timing ode solver ...")
@time fsol(p)
@time fsol(p)
@time ForwardDiff.gradient(fsol, p)
@time ForwardDiff.gradient(fsol, p)
