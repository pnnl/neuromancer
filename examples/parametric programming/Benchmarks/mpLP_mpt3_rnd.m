%% multiparametric linear program 
% using YALMIP and MPT3
% https://yalmip.github.io/tutorial/multiparametricprogramming/

yalmip('clear')
clear all

%% LP problem matrices
nx = 1; % number of decision variables
ntheta = 2; % Number of parameters
ncon = 5; % Number of constraints
A = randn(ncon,nx);
b = randn(ncon,1);
E = randn(ncon,ntheta);
c = randn(1,nx);    

%% mpLP problem formulation 
x = sdpvar(nx,1);     % decision variable
theta = sdpvar(2,1);  % problem parameter

theta_min = -2;
theta_max = 2;

C = [A*x <= b+E*theta];
% obj = c*x;    
obj = (x)'*(x);

%  [mptsol,diagnostic,u,J,U] = solvemp(Constraints,Objective,Options,Parameters)
[solution,diagnostics,aux,Valuefunction,Optimal_x] = solvemp(C,obj, [], theta);
% test optimal decision at specific parameter value
assign(theta,[0.1;0.2]);
value(Optimal_x)

% plp = Opt(C, obj, theta, x);
% solution = plp.solve();

%% Plots
plot(Valuefunction);
figure
plot(Optimal_x);

% figure;
% solution.xopt.fplot('primal');
% figure;
% solution.xopt.fplot('obj');
% xlabel('t');
% ylabel('J(t)');


[theta1,theta2] = meshgrid(theta_min:.5:theta_max);
X = nan(length(theta1),length(theta2));

for i=1:length(theta1)
    for j=1:length(theta2)
        theta_k = [theta1(i,j); theta2(i,j)];
        assign(theta,theta_k);
        X(i,j) = value(Optimal_x);
    end
end

figure
s = surf(theta1,theta2,X);
xlabel('theta_2')
ylabel('theta_1')
zlabel('x')
s.EdgeColor = 'none';



