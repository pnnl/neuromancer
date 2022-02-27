
%% Double integrator model
% system from slide 72
% https://engineering.utsa.edu/ataha/wp-content/uploads/sites/38/2017/10/MPC_Intro.pdf

yalmip('clear')
clear all
% Model data
A = [ 1.    ,  0.    ,  0.    ,  0.2   ,  0.    ,  0.    ;
      0.    ,  1.    ,  0.    ,  0.    ,  0.2   ,  0.    ;
      0.    ,  0.    ,  1.    ,  0.    ,  0.    ,  0.2   ;
      0.    ,  0.    , -1.96  ,  0.9975,  0.    ,  0.    ;
      0.    ,  0.    ,  0.    ,  0.    ,  0.9975,  0.    ;
      0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ];
B = [ 0.        ,  0.        ;
      0.        ,  0.        ;
      0.        ,  0.        ;
      0.05      , -0.        ;
      0.        ,  0.05      ;
      1.05263158,  0.        ];
  
C = eye(6);
nx = 6; % Number of states
nu = 2; % Number of inputs
ny = 6; % Number of outputs
% Prediction horizon
N = 10;

% constraints
umin = -5;
umax = 5;
xmin = -5;
xmax = 5;

% objective weights
Qx = 3;
Qu = 0.1;

% variables
x = sdpvar(nx, N+1, 'full');        % MPC parameter
u = sdpvar(nu, N, 'full');          % decision variables
y = sdpvar(ny, N, 'full');          % internal variable

%% Optimization problem

% explicit MPC policy
run_eMPC = 0;

con = [];
obj = 0;
for k = 1:N
    con = con + [x(:, k+1) == A*x(:, k) + B*u(:, k)];       % state update model
    con = con + [ xmin <= x(:, k) <= xmax ];
    con = con + [ umin <= u(:, k) <= umax ];                    % input constraints
    
        % objective function
    obj = obj + x(:, k)'*Qx*x(:, k) + u(:, k)'*Qu*u(:, k);
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
x0 = 0.7*ones(nx,1);
Nsim = 60;
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
hold on
plot(t,xmin.*ones(size(Xsim(:,1:end-1))),'k--','LineWidth',2)
plot(t,xmax.*ones(size(Xsim(:,1:end-1))),'k--','LineWidth',2)
title('States')
legend('x_1', 'x_2')
xlabel('time')
subplot(2,1,2)
stairs(t, Usim.','LineWidth',2)
hold on
plot(t,umin.*ones(size(Xsim(:,1:end-1))),'k--','LineWidth',2)
plot(t,umax.*ones(size(Xsim(:,1:end-1))),'k--','LineWidth',2)
title('Input')
legend('u')
xlabel('time')

%% Generate samples for approximate MPC

generate_samples = false;

if generate_samples
    nsim = 15000;
    % sample initial conditions x_0
    samples_x = 0.5*randn(nx,nsim);
    tStart = tic;          
    T = zeros(1,nsim);
    samples_u = zeros(nu,nsim);
    for k  = 1:nsim
        fprintf('k = %d\n', k)
        tic
        [u, problem, info] = opt{samples_x(:,k)};  
        uopt = value(u(:, 1));  
        if sum(isnan(uopt)) > 0
            [u, problem, info] = opt{0.5*samples_x(:,k)};  
            uopt = value(u(:, 1));  
            if sum(isnan(uopt)) > 0
                fprintf('NaN alert')
            end
        end
        samples_u(:,k) = uopt;
        T(k)= toc;
    end
    tMul = sum(T)
    save aMPC_dataset.mat samples_x samples_u
end


