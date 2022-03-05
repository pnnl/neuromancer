
%% Double integrator model
% system from slide 72
% https://engineering.utsa.edu/ataha/wp-content/uploads/sites/38/2017/10/MPC_Intro.pdf

yalmip('clear')
clear all
% Model data
A = [1.2 1;0 1];
B = [1;0.5];  
% C = [1 1];
C = [1 0; 0 1];
nx = 2; % Number of states
nu = 1; % Number of inputs
ny = 2; % Number of outputs
% Prediction horizon
N = 10;

% constraints
umin = -1;
umax = 1;
ymin = 5;
ymax = 5;
xmin = -10;
xmax = 10;

% objective weights
Qy = 1;
Qu = 1;

% variables
x = sdpvar(nx, N+1, 'full');        % MPC parameter
u = sdpvar(nu, N, 'full');          % decision variables
y = sdpvar(ny, N, 'full');          % internal variable

%% Optimization problem

% compute explicit MPC policy
run_eMPC = 0;

con = [];
obj = 0;
for k = 1:N
    con = con + [x(:, k+1) == A*x(:, k) + B*u(:, k)];       % state update model
%     con = con + [y(:, k) == C*x(:, k+1) ];         % output model
    con = con + [ xmin <= x(:, k) <= xmax ];
%     con = con + [ ymin <= y(:, k) <= ymax ];
    con = con + [ umin <= u(:, k) <= umax ];                    % input constraints
    
        % objective function
    obj = obj + x(:, k)'*Qy*x(:, k) + u(:, k)'*Qu*u(:, k);
end

options = sdpsettings('verbose', 0,'solver','QUADPROG','QUADPROG.maxit',1e6);


if run_eMPC
    % eMPC optimizer
    plp = Opt(con, obj, [x(:, 1)], u(:, 1));
    solution = plp.solve();
    filename = strcat('eMPC_di');
    solution.xopt.toMatlab(filename, 'primal', 'obj')  %  generate standalone m-file with the eMPC policy
    
%     figure;
%     solution.xopt.plot()
    figure;
    solution.xopt.fplot('primal');
    figure;
    solution.xopt.fplot('obj');
    xlabel('t');
    ylabel('J(t)');
    
%     [sol,diagn,Z,Valuefcn,Optimizer] = solvemp(con,obj ,[],[x(:, 1)], u(:, 1));
%     figure;plot(Valuefcn)
%     figure;plot(Optimizer)
else
    % % iMPC optimizer
    opt = optimizer(con, obj, options, x(:, 1), u);
end


%% Simulation Setup


% initial conditions
x0 = 1.5*ones(nx,1);
Nsim = 50;
Xsim = x0;
Usim = [];
Ysim = [];
for k = 1:Nsim
    
    x_k = Xsim(:, end);
%     uopt = eMPC_policy(x_k);    
    if run_eMPC
        uopt = solution.xopt.feval(x_k, 'primal');
    else
        [u, problem, info] = opt{x_k};  
        uopt = value(u(:, 1));                      
    end
    
    xn = A*x_k + B*uopt;
    yn = C*x_k;
    Usim = [Usim, uopt];
    Xsim = [Xsim, xn];
    Ysim = [Ysim, yn];
end

%% Plots
% close all
t = 0:Nsim-1;

figure
subplot(2,1,1)
plot(t,Xsim(:,1:end-1),'LineWidth',2)
legend('x_1', 'x_2')
xlabel('time')
subplot(2,1,2)
stairs(t,Usim,'LineWidth',2)
title('Input')
legend('u')
xlabel('time')

% TODO: plots with multiple initial conditions
figure
% solution.xopt.plot()
% hold on
plot(Xsim(1,:),Xsim(2,:),'LineWidth',2)
grid on
xlim([-1 7])
ylim([-4 4])

[X1,X2] = meshgrid(-5:.1:5);
U = nan(length(X1),length(X1));
Alpha = zeros(length(X1),length(X1));
X1_feasible = [];
X2_feasible = [];
for i=1:length(X1)
    for j=1:length(X2)
        x_k = [X1(i,j); X2(i,j)];
%         U(i,j) = solution.xopt.feval(x_k, 'primal'); 
        
        if run_eMPC
            U(i,j) = solution.xopt.feval(x_k, 'primal'); 
        else
            [u, problem, info] = opt{x_k};  
            U(i,j) = value(u(:, 1));                      
        end
        
        xn = A*x_k + B*U(i,j);
%         xn = A*x_k + B*uopt;
        if ~norm(x_k)==0
            Alpha(i,j) = norm(xn)/norm(x_k);
        else
            Alpha(i,j) = 0;
        end
        if ~isnan(U(i,j))
            X1_feasible = [X1_feasible; X1(i,j)];
            X2_feasible = [X2_feasible; X2(i,j)];
        end
    end
end
figure
s = surf(X1,X2,U)
xlabel('x_2')
ylabel('x_1')
zlabel('u')
s.EdgeColor = 'none';

figure
s = surf(X1,X2,Alpha)
xlabel('x_2')
ylabel('x_1')
zlabel('u')
s.EdgeColor = 'none';

figure
imagesc(rot90(Alpha))
xlabel('x_2')
ylabel('x_1')
colorbar

P = [X1_feasible, X2_feasible];
[k,av] = convhull(P);
figure
plot(P(:,1),P(:,2),'*')
hold on
plot(P(k,1),P(k,2))

