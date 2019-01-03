clear

step_per_epoch = 30000;
num_epoch = 10;
step = 1:step_per_epoch*num_epoch;

lambda_min = 10;
lambda_base = 1000;
gamma = 0.05;
lambda_power = 0.8;

% gamma = 0.001;
% lambda_power = 1;

lambda = max(lambda_min, lambda_base * (1 + gamma * step).^(-lambda_power));
plot(step, lambda);
xlim([0 step_per_epoch*num_epoch])
ylim([0 20])