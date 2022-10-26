clc
clear all
close all


%% Set up optimization environment
opti = casadi.Opti();

x = opti.variable();
y = opti.variable();

b = 1;

opti.minimize((1-x)^2+b*(y-x^2)^2);

opti.solver('ipopt');
sol = opti.solve();

figure
hold on
plot(sol.value(x),sol.value(y),'o');

[X,Y] = meshgrid(linspace(0,1.5),linspace(-0.5,1.5));
contour(X,Y,(1-X).^2+b*(Y-X.^2).^2,100);

print('rosenbrock1','-dpng')

%% Set up optimization environment
opti = casadi.Opti();

x = opti.variable();
y = opti.variable();

opti.minimize((1-x)^2+b*(y-x^2)^2);
opti.subject_to(x^2+y^2==1);

opti.solver('ipopt');
sol = opti.solve();

figure
hold on
plot(sol.value(x),sol.value(y),'o');

[X,Y] = meshgrid(linspace(0,1.5),linspace(-0.5,1.5));
contour(X,Y,(1-X).^2+b*(Y-X.^2).^2,100);
ts = linspace(0,2*pi);
xlim([0    1.5])
ylim([-0.5 1.5])
plot(cos(ts),sin(ts),'r','linewidth',2)


print('rosenbrock2','-dpng')

%% Set up optimization environment
opti = casadi.Opti();

x = opti.variable();
y = opti.variable();

opti.minimize((1-x)^2+b*(y-x^2)^2);
opti.subject_to(x^2+y^2==1);
opti.subject_to(y>=x);

opti.solver('ipopt');
sol = opti.solve();

figure
hold on
plot(sol.value(x),sol.value(y),'o');

[X,Y] = meshgrid(linspace(0,1.5),linspace(-0.5,1.5));
contour(X,Y,(1-X).^2+b*(Y-X.^2).^2,100);
ts = linspace(0,2*pi,1000);
plot(cos(ts),sin(ts),'r','linewidth',2)
plot(ts,ts,'r','linewidth',2)
xlim([0    1.5])
ylim([-0.5 1.5])

print('rosenbrock3','-dpng')

%% parametric

opti = casadi.Opti();

x = opti.variable();
y = opti.variable();
r = opti.parameter();

f = (1-x)^2+(y-x^2)^2;
opti.minimize(f);
con = x^2+y^2<=r;
opti.subject_to(con);

opti.solver('ipopt');


figure
hold on
for r_value=linspace(1,3,25)
    opti.set_value(r,r_value)
    sol = opti.solve();
    plot(r_value,sol.value(f),'ko')
    
    % Plot the lagrange multipliers
    lam = sol.value(opti.dual(con));
    ts = linspace(-0.02,0.02);
    plot(ts+r_value,-lam*ts+sol.value(f),'r')
end

xlabel('Value of r')
ylabel('Objective value at solution')
print('rosenbrock4','-dpng')
