%% multiparametric linear program 
% using MPT3
% https://www.mpt3.org/ParOpt/ParOpt

yalmip('clear')
clear all

rng(1337,'twister')

%% LP problem matrices
nx = 1; % number of decision variables
ntheta = 2; % Number of parameters
ncon = 5; % Number of constraints
A = randn(ncon,nx);
b = randn(ncon,1);
E = randn(ncon,ntheta);

%% mpLP problem formulation 
x = sdpvar(nx,1);     % decision variable
theta = sdpvar(2,1);  % problem parameter

theta_min = -2;
theta_max = 2;

C = [A*x <= b+E*theta];
C = C + [theta_min<= theta <= theta_max];

obj = (x)'*(x);

% MPT3 format
plp = Opt(C, obj, theta, x);
solution = plp.solve();

%% Plots

% for i = 1:nx
%     figure;
%     solution.xopt.fplot('primal', 'position', i);
%     xlabel('t');
%     ylabel(sprintf('x_%d(t)', i));
% end
% 
% figure;
% solution.xopt.fplot('obj');
% xlabel('t');
% ylabel('J(t)');

[theta1,theta2] = meshgrid(theta_min:.05:theta_max);
X = nan(length(theta1),length(theta2));
J = nan(length(theta1),length(theta2));

for i=1:length(theta1)
    for j=1:length(theta2)
        theta_k = [theta1(i,j); theta2(i,j)];
        X(i,j) = solution.xopt.feval(theta_k, 'primal'); 
        J(i,j) = solution.xopt.feval(theta_k, 'obj');
    end
end

figure
s = surf(theta1,theta2,X);
xlabel('theta_2')
ylabel('theta_1')
zlabel('x')
s.EdgeColor = 'none';


figure
s = surf(theta1,theta2,J);
xlabel('theta_2')
ylabel('theta_1')
zlabel('J')
s.EdgeColor = 'none';




