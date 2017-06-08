%% YALMIP : Circular Trajectory Tracking using MPC 
clc;
clear;
close all;
yalmip('clear')

%% MPC Parameters definition
% Model Parameters
params.Ts = 0.01;   % Sampling time 
params.nstates = 3; % No of States in the system [x, y,theta]
params.ninputs = 1; % No of Inputs in the system [delta_f], steering angle
params.l_f = 0.17; % Distance from the CG to the front wheel
params.l_r = 0.16; % Distance from the CG to the rear wheel
params.l_q = params.l_r / (params.l_r + params.l_f);

% Control Parameters
params.Np = 30;   % The horizon
params.Q = [1 0 0; 0 1 0; 0 0 1]; % The Q matrix for running cost for error in trajectory tracking
params.Q_N = [1 0 0; 0 1 0; 0 0 1];% The Q matrix for terminal cost for error in trajectory tracking
params.R = 1; %  The R matrix for running cost for input

% Initial conditions for the state of the vehicle       
params.x0 = 1.5;  
params.y0 = 0;  
params.v0 = 1; 
params.psi0 = pi/2; 

% Number of iterations
params.N = 1000;

% Every point on the circle is a reference to be followed by the vehicle
% x0 and y0 are the coordinates of the circle's center,r is its radius.
trajectory_params.x0 = 0;
trajectory_params.y0 = 0;
trajectory_params.r = 1.5;

%% Simulation environment
% Initialization
z(1,:)  = [params.x0, params.y0, params.psi0]; % State representation
u(1) = 0; % Control Input

% Generate the trajectory to be followed
traj = trajectory_circular(trajectory_params);

% Obtain the symbolic form of the matrices A, B
syms xkn ykn psikn xk yk psik Ts lr lf delta
[symbolic_linear_A, symbolic_linear_B] = Symbolic_Kinematic_Model(params);

k = 0;
while (k < params.N)
  k = k+1;  
  fprintf("Iteration number = %d ",k);
  
  % The distances of all the points in the trajectory to the vehicle
  d = sqrt((traj(:,1)-z(k,1)).^2 + (traj(:,2)-z(k,2)).^2);
  
  % The index of the point that is the closest to the vehicle
  [~, idx] = min(d);
  
  % The first reference point for x, y, theta
  z_ref(k,:) = traj(idx, :);   
  
  refs = zeros(params.Np+1, 3);
  
  for i = 1:params.Np+1
    theta = z_ref(k,3) +(params.v0 / trajectory_params.r) * params.Ts * i;
    x = trajectory_params.x0 + trajectory_params.r * cos(theta - pi/2);
    y = trajectory_params.y0 + trajectory_params.r * sin(theta - pi/2);
    refs(i,:) = [x, y, theta];
  end
% The first case
if (k == 1)
  params.A = subs(symbolic_linear_A, [Ts lr lf psik delta], [params.Ts params.l_r params.l_f z(k,3) 0]);
  params.B = subs(symbolic_linear_B, [Ts lr lf psik delta], [params.Ts params.l_r params.l_f z(k,3) 0]);
  params.A = double(params.A);
  params.B = double(params.B);
end
% The remaining cases
if (k > 1)
  % Completing the circle
  if (z_ref(k,3) < z_ref(k-1,3))    
    refs(:,3) = refs(:,3) + 2*pi;
    z_ref(k,3) = z_ref(k,3) + 2*pi;
    traj(idx,3) = traj(idx,3) + 2*pi;
  end
  params.A = subs(symbolic_linear_A, [Ts lr lf psik delta], [params.Ts params.l_r params.l_f z(k,3) u(k-1)]);
  params.B = subs(symbolic_linear_B, [Ts lr lf psik delta], [params.Ts params.l_r params.l_f z(k,3) u(k-1)]);
  params.A = double(params.A);
  params.B = double(params.B);
end

% Ensure stability : Ricatti Equation Solver is used to ensure stability 
% finding stable Q matrix
  [params.Qf,~,~,err] = dare(params.A, params.B, params.Q, params.R);
  if (err == -1 || err == -2)
    params.Qf = params.Q;
  end
 yalmip('clear')

  % The predicted state and control variables
  z_mpc = sdpvar(params.Np+1, params.nstates);
  u_mpc = sdpvar(params.Np, params.ninputs);

  % Reset constraints and objective function
  constraints = [];
  J = 0;

  for i = 1:params.Np
    
    if i < params.Np
      J = J + (z_mpc(i,:)-refs(i,:)) * params.Qf * (z_mpc(i,:)-refs(i,:))' +  ...
        u_mpc(i,:) * params.R * u_mpc(i,:)';


      % Model constraints
      constraints = [constraints, ...
        z_mpc(i+1,:)' == params.A * z_mpc(i,:)' + params.B * u_mpc(i,:)'];

      % Input constraints
      constraints = [constraints, ...
        -pi/4 <= u_mpc(i) <= pi/4];
    else
      J = J + ...
        (z_mpc(i,:)-refs(i,:)) * params.Q_N * (z_mpc(i,:)-refs(i,:))' +  ...
        u_mpc(i,:) * params.R * u_mpc(i,:)';


      % Model constraints
      constraints = [constraints, ...
        z_mpc(i+1,:)' == params.A * z_mpc(i,:)' + params.B * u_mpc(i,:)'];

      % Input constraints
      constraints = [constraints, ...
        -pi/4 <= u_mpc(i) <= pi/4];
    end
  end

  
  %  terminal cost
  J = J + (z_mpc(params.Np+1, :) - refs(params.Np+1,:)) * params.Qf * (z_mpc(params.Np+1, :)-refs(params.Np+1,:))';

  % terminal constraints
    constraints = [constraints, ...
       z_mpc(params.Np+1, 3) - refs(params.Np+1, 3) == 0];
        
  assign(z_mpc(1,:), z(k,:));

  % Options
  ops = sdpsettings('solver', 'quadprog');

  % Optimize
  optimize([constraints, z_mpc(1,:) == z(k,:)], J, ops);

  % Take the first predicted input 
  u(k) = value(u_mpc(1));

  % simulate the vehicle
  z(k+1,:) = vehicle_model(z(k,:), u(k), params);
  plot(traj(:,1), traj(:,2))
  hold on
  plot(z(2:end,1), z(2:end,2));
  plot(refs(:,1), refs(:,2), '*', 'Color','b')
  plot(z(k,1), z(k,2), '*', 'Color', 'r')
  axis equal
  axis([-2 2 -2 2])
  drawnow
  title('Trajectory Tracking')
  hold off

end
% Plot state deviations and inputs
figure
subplot(2,2,1)
plot(traj(:,1), traj(:,2),'-*r')
hold on
plot(z(:,1), z(:,2))
title('Trajectory Tracking')
legend ('Given Trajectory','My Trajectory','location','southeast');
axis equal
hold off

subplot(2,2,2)
plot((z(1:end-1, 1) - z_ref(:,1)).^2 + (z(1:end-1, 2) - z_ref(:,2)).^2)
xlabel('No of iterations');
ylabel('Deviation from the reference trajectory');
title('Distance deviation')


subplot(2,2,3)
plot((z_ref(:,3) - z(1:end-1,3))*180/pi)
xlabel('No of iterations');
ylabel('Deviation from the reference trajectory');
title('Orientation deviation')

figure
plot(u * 180 / pi)
xlabel('No of iterations');
ylabel('Variation in steering angle');
title('Input Steering Angle')

%% Trajectory Generator

%Creates the Trajectory to Follow
function traj = trajectory_circular(trajectory_params)
% Center of the circular trajectory 
  x0 = trajectory_params.x0;
  y0 = trajectory_params.y0;
  r = trajectory_params.r;

% Creates the circular trajectory and corresponding angle
  for i = 0:1:360
      x = x0 + r*cos(pi * i / 180);
      y = y0 + r*sin(pi * i / 180);
      theta = pi - atan2(x - x0, y - y0);

      traj(i+1, 1) = x;
      traj(i+1, 2) = y;
      traj(i+1, 3) = theta;
  end
  
end

%% Kinematic Bicycle Model

function [A, B] = Symbolic_Kinematic_Model(params)

  syms xkn ykn psikn xk yk psik Ts lr lf delta
  
  xkn = xk + Ts*params.v0*cos(psik + atan((lr/(lr+lf))*tan(delta)));
  ykn = yk + Ts*params.v0*sin(psik + atan((lr/(lr+lf))*tan(delta)));
  psikn = psik + Ts*(params.v0/lr)*sin(atan((lr/(lr+lf))*tan(delta)));

  A(1,1) = diff(xkn, xk);
  A(1,2) = diff(xkn, yk);
  A(1,3) = diff(xkn, psik);

  A(2,1) = diff(ykn, xk);
  A(2,2) = diff(ykn, yk);
  A(2,3) = diff(ykn, psik);

  A(3,1) = diff(psikn, xk);
  A(3,2) = diff(psikn, yk);
  A(3,3) = diff(psikn, psik);

  B(1,1) = diff(xkn, delta);
  B(2,1) = diff(ykn, delta);
  B(3,1) = diff(psikn, delta);

end

%% Vehicle Model
function z_new = vehicle_model(z_curr, u_curr, params)
    
  delta = u_curr;
  
  % vehicle dynamics equations
  z_new(1) = z_curr(1) + (params.v0*cos(z_curr(3)+atan((params.l_r/(params.l_r+params.l_f))*tan(delta))))*params.Ts;
  z_new(2) = z_curr(2) + (params.v0*sin(z_curr(3)+atan((params.l_r/(params.l_r+params.l_f))*tan(delta))))*params.Ts;
  z_new(3) = z_curr(3) + (params.v0 / params.l_r * sin(atan((params.l_r/(params.l_r+params.l_f))*tan(delta))))*params.Ts;

end

