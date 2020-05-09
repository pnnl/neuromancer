% Benchmark Matlab system ID toolbox
% https://www.mathworks.com/help/ident/examples/two-tank-system-single-input-single-output-nonlinear-arx-and-hammerstein-wiener-models.html

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% load data
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
load NLIN_two_tank_SISO
z = iddata(y, u, 0.2, 'Name', 'Two tank system');
plot(z)
z1 = z(1:1000); z2 = z(1001:2000); z3 = z(2001:3000); % data split 

% model order selection
V = arxstruc(z1,z2,struc(1:5, 1:5,1:5));
nn = selstruc(V,'aic') % selection of nn=[na nb nk] by Akaike's information criterion (AIC)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% nonlinear ARX wavenet 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
mw1 = nlarx(z1,[5 1 3], wavenet);
mw1.Nonlinearity.NumberOfUnits
% specifying number of units in the WAVENET estimator,
mw2 = nlarx(z1,[5 1 3], wavenet('NumberOfUnits',8));
mw3 = nlarx(z1,[5 1 3], wavenet, 'nlreg', [1 2 6]);  % using short-hand notation
getreg(mw3, 'nonlinear')  % get nonlinear regressors
mw4 = nlarx(z1,[5 1 3],wavenet,'nlreg','input');
mw4.nlreg    % 'nlreg' is the short-hand of 'NonlinearRegressors'
getreg(mw4,'nonlinear')  % get nonlinear regressor
% exhausitve search
opt = nlarxOptions('Display','on');
mws = nlarx(z1,[5 1 3], wavenet, 'nlreg', 'search', opt);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % linear ARX model
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
mlin = arx(z1,[5 1 3]);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Nonlinear ARX Model with SIGMOIDNET Nonlinearity Estimator
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
ms1 = nlarx(z1,[5 1 3], sigmoidnet('NumberOfUnits', 8));

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Hammerstein-Wiener Model with the Piecewise Linear Nonlinearity Estimator
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
mhw1 = nlhw(z1, [1 5 3], pwlinear, pwlinear);


% evaluate model
figure; compare(z1,mlin,mw2,mw3,mw4,mws,ms1,mhw1);
figure; compare(z2,mlin,mw2,mw3,mw4,mws,ms1,mhw1);
figure; compare(z3,mlin,mw2,mw3,mw4,mws,ms1,mhw1);




